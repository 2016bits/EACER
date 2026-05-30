from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WeightingOutput:
    weighted_visual: torch.Tensor          # (B, D) - aggregated tilde_v
    alpha: torch.Tensor                    # (B, P)
    W: torch.Tensor                        # (B, P) - final patch weights pre-softmax
    S: Optional[torch.Tensor]              # (B, P)
    D: Optional[torch.Tensor]              # (B, P)
    R: Optional[torch.Tensor]              # (B, P)


def _masked_max_cos(patches: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Max cosine similarity between each patch and any (non-pad) token.

    patches: (B, P, D); tokens: (B, L, D); token_mask: (B, L) {0,1}
    returns: (B, P)
    """
    p = F.normalize(patches, dim=-1)
    t = F.normalize(tokens, dim=-1)
    sim = torch.einsum("bpd,bld->bpl", p, t)                       # (B, P, L)
    mask = token_mask.unsqueeze(1).bool()                          # (B, 1, L)
    sim = sim.masked_fill(~mask, float("-inf"))
    return sim.max(dim=-1).values                                  # (B, P)


class EntropyAwareComplementaryWeighting(nn.Module):
    """Combines S_j, D_j, R_j into W_j; outputs weighted aggregated visual tokens.

    Each of S/D/R can be toggled on/off (multiplicative neutral = 1.0 when off).
    """

    def __init__(
        self,
        use_claim_relevance: bool = True,
        use_complementarity: bool = True,
        use_entropy_reliability: bool = True,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_S = use_claim_relevance
        self.use_D = use_complementarity
        self.use_R = use_entropy_reliability
        self.tau_w = softmax_temperature

    def forward(
        self,
        patches: torch.Tensor,                # (B, P, D) projected
        claim_tokens: torch.Tensor,            # (B, L_c, D)
        claim_mask: torch.Tensor,              # (B, L_c)
        text_tokens: torch.Tensor,             # (B, L_t, D)
        text_mask: torch.Tensor,               # (B, L_t)
        reliability: Optional[torch.Tensor] = None,   # (B, P) precomputed R
    ) -> WeightingOutput:
        B, P, _ = patches.shape

        if self.use_S:
            S = _masked_max_cos(patches, claim_tokens, claim_mask)
            S = ((S + 1.0) * 0.5).clamp(0.0, 1.0)                  # map [-1,1] -> [0,1]
        else:
            S = None

        if self.use_D:
            sim_pt = _masked_max_cos(patches, text_tokens, text_mask)   # (B, P), in [-1, 1]
            D = (1.0 - sim_pt) * 0.5                                    # in [0, 1]
            D = D.clamp(0.0, 1.0)
        else:
            D = None

        R = reliability if self.use_R else None
        if R is not None:
            R = R.clamp(0.0, 1.0)

        components = [c for c in (S, D, R) if c is not None]
        if components:
            W = components[0]
            for c in components[1:]:
                W = W * c
        else:
            # No weighting -> uniform attention.
            W = patches.new_ones(B, P)

        # Numerical safety: at least one positive value per row.
        W = W.clamp(min=1e-6)

        alpha = F.softmax(W / self.tau_w, dim=-1)                  # (B, P)
        weighted = torch.einsum("bp,bpd->bd", alpha, patches)      # (B, D)

        return WeightingOutput(
            weighted_visual=weighted,
            alpha=alpha,
            W=W,
            S=S,
            D=D,
            R=R,
        )
