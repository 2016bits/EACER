from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionOutput:
    pooled: torch.Tensor          # (B, D)
    tokens: torch.Tensor          # (B, 1 + L_t, D)


class EvidenceFusion(nn.Module):
    """CIEA-style fusion: concat [start_emb; tilde_v; end_emb; text tokens] then transformer.

    The pooled output is the first token's hidden state.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.start_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.end_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.visual_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.text_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        weighted_visual: torch.Tensor,        # (B, D)
        text_tokens: torch.Tensor,            # (B, L_t, D)
        text_attention_mask: torch.Tensor,    # (B, L_t)
    ) -> FusionOutput:
        B = weighted_visual.size(0)
        v = (weighted_visual.unsqueeze(1) + self.visual_type)            # (B, 1, D)
        start = self.start_emb.expand(B, -1, -1)
        end = self.end_emb.expand(B, -1, -1)
        text = text_tokens + self.text_type
        seq = torch.cat([start, v, end, text], dim=1)                    # (B, 3+L_t, D)
        # Build mask: positions to KEEP have value 1
        keep = torch.cat(
            [
                weighted_visual.new_ones(B, 3),                          # start, v, end
                text_attention_mask.float(),
            ],
            dim=1,
        )
        key_padding_mask = ~keep.bool()                                  # True = ignore
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        out = self.out_norm(out)
        pooled = out[:, 0, :]
        return FusionOutput(pooled=pooled, tokens=out[:, 1:, :])
