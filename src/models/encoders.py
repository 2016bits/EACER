from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel


@dataclass
class TextEncoderOutput:
    token_embeddings: torch.Tensor       # (B, L, D)
    pooled: torch.Tensor                  # (B, D)
    attention_mask: torch.Tensor          # (B, L)


class TextEncoder(nn.Module):
    """Wraps a HuggingFace text encoder.

    ``pooling`` selects how the sentence representation is built:
      - "mean": attention-mask-aware mean over token embeddings (default; what
        the XLM-RoBERTa-base / BERT runs used).
      - "cls":  ``last_hidden_state[:, 0, :]`` — the canonical choice for
        retrieval-tuned encoders like BGE-M3, which were trained with CLS
        contrastive heads.

    Token-level outputs are returned regardless so downstream modules (S/D
    weights, fusion) keep their per-token view.
    """

    def __init__(self, model_name: str, freeze: bool = False, pooling: str = "mean"):
        super().__init__()
        assert pooling in {"mean", "cls"}, f"unknown pooling: {pooling}"
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextEncoderOutput:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        if self.pooling == "cls":
            pooled = tokens[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (tokens * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom
        return TextEncoderOutput(token_embeddings=tokens, pooled=pooled, attention_mask=attention_mask)


class VisualEncoder(nn.Module):
    """Frozen-by-default CLIP vision encoder returning patch tokens (CLS dropped)."""

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        patches = out.last_hidden_state[:, 1:, :]   # drop CLS
        return patches                               # (B, P, D_vis)


class PatchProjector(nn.Module):
    """Maps CLIP patch features to the text-encoder hidden space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
