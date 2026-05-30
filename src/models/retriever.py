from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import PatchProjector, TextEncoder, VisualEncoder
from .entropy import EntropyReliability
from .fusion import EvidenceFusion
from .weighting import EntropyAwareComplementaryWeighting


@dataclass
class RetrieverOutput:
    claim_repr: torch.Tensor              # (B, D)
    evidence_repr: torch.Tensor           # (B + B*neg, D)
    claim_comp_repr: Optional[torch.Tensor]   # (B, D)
    weighted_visual: torch.Tensor         # (B + B*neg, D)
    alpha: torch.Tensor                   # (B + B*neg, P)
    aux: Dict[str, Any]


class ECERRetriever(nn.Module):
    """Top-level Entropy-aware Complementary Evidence Retriever."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        m_cfg = cfg["model"]
        w_cfg = cfg["weighting"]

        self.text_encoder = TextEncoder(
            m_cfg["text_encoder"], freeze=m_cfg.get("freeze_text", False)
        )
        self.visual_encoder = VisualEncoder(
            m_cfg["visual_encoder"], freeze=m_cfg.get("freeze_visual", True)
        )
        self.projector = PatchProjector(
            in_dim=self.visual_encoder.hidden_dim,
            hidden_dim=m_cfg.get("projector_hidden_dim", 768),
            out_dim=m_cfg.get("projector_out_dim", self.text_encoder.hidden_dim),
        )

        self.weighting = EntropyAwareComplementaryWeighting(
            use_claim_relevance=w_cfg["use_claim_relevance"],
            use_complementarity=w_cfg["use_complementarity"],
            use_entropy_reliability=w_cfg["use_entropy_reliability"],
            softmax_temperature=w_cfg.get("softmax_temperature", 1.0),
        )

        if w_cfg["use_entropy_reliability"]:
            self.entropy = EntropyReliability(
                clip_model_name=m_cfg["visual_encoder"],
                concepts_path=w_cfg.get("entropy_concepts_path"),
                K=w_cfg.get("entropy_K", 1000),
                scheme=w_cfg.get("entropy_scheme", "A"),
                mu=w_cfg.get("entropy_mu", 4.0),
                sigma=w_cfg.get("entropy_sigma", 1.5),
            )
        else:
            self.entropy = None

        self.fusion = EvidenceFusion(
            hidden_dim=m_cfg.get("fusion_hidden_dim", self.text_encoder.hidden_dim),
            num_layers=m_cfg.get("fusion_num_layers", 2),
            num_heads=m_cfg.get("fusion_num_heads", 8),
        )

        # Projection head for claim->fusion-space (claim already in text-encoder space).
        self.claim_head = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_dim, self.fusion.start_emb.shape[-1]),
            nn.LayerNorm(self.fusion.start_emb.shape[-1]),
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_claim(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out

    def encode_evidence_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out

    def encode_visual(self, pixel_values):
        raw = self.visual_encoder(pixel_values)            # (B, P, D_clip)
        projected = self.projector(raw)                    # (B, P, D)
        return raw, projected

    # ------------------------------------------------------------------
    # Forward (training): claim aligned 1-1 with positive evidence; negatives flat
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> RetrieverOutput:
        claim_out = self.encode_claim(batch["claim_input_ids"], batch["claim_attention_mask"])

        evi_text_out = self.encode_evidence_text(
            batch["evidence_input_ids"], batch["evidence_attention_mask"]
        )

        raw_patches, projected_patches = self.encode_visual(batch["pixel_values"])

        # Build claim_tokens aligned per-evidence (positives + negatives).
        # Each sample contributes 1 positive + n_neg negatives. We tile the claim
        # tokens across its own positive+negatives so that S_j is computed
        # against the *correct* claim for every evidence.
        B_claim = claim_out.token_embeddings.size(0)
        nneg = batch["num_negatives_per_sample"]                           # (B_claim,)
        B_evi = evi_text_out.token_embeddings.size(0)

        # repeat indices: for sample i contribute (1 + nneg_i) copies of i
        repeats = (nneg + 1).tolist()
        rep_index = torch.cat(
            [torch.full((r,), i, dtype=torch.long) for i, r in enumerate(repeats)]
        ).to(claim_out.token_embeddings.device)
        assert rep_index.numel() == B_evi, "evidence batch size mismatch"

        claim_tokens_tiled = claim_out.token_embeddings.index_select(0, rep_index)
        claim_mask_tiled = claim_out.attention_mask.index_select(0, rep_index)

        # Reliability R_j from raw CLIP patches (avoids projector drift over training).
        if self.entropy is not None:
            R = self.entropy(raw_patches)
        else:
            R = None

        weighting_out = self.weighting(
            patches=projected_patches,
            claim_tokens=claim_tokens_tiled,
            claim_mask=claim_mask_tiled,
            text_tokens=evi_text_out.token_embeddings,
            text_mask=evi_text_out.attention_mask,
            reliability=R,
        )

        evidence_fused = self.fusion(
            weighted_visual=weighting_out.weighted_visual,
            text_tokens=evi_text_out.token_embeddings,
            text_attention_mask=evi_text_out.attention_mask,
        )

        claim_repr = F.normalize(self.claim_head(claim_out.pooled), dim=-1)
        evidence_repr = F.normalize(evidence_fused.pooled, dim=-1)

        claim_comp_repr = None
        if "claim_comp_input_ids" in batch:
            claim_comp_out = self.encode_claim(
                batch["claim_comp_input_ids"], batch["claim_comp_attention_mask"]
            )
            claim_comp_repr = F.normalize(self.claim_head(claim_comp_out.pooled), dim=-1)

        return RetrieverOutput(
            claim_repr=claim_repr,
            evidence_repr=evidence_repr,
            claim_comp_repr=claim_comp_repr,
            weighted_visual=F.normalize(weighting_out.weighted_visual, dim=-1),
            alpha=weighting_out.alpha,
            aux={
                "rep_index": rep_index,
                "S": weighting_out.S,
                "D": weighting_out.D,
                "R": weighting_out.R,
            },
        )

    # ------------------------------------------------------------------
    # Inference encoders for index building
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_claim_for_index(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encode_claim(input_ids, attention_mask)
        return F.normalize(self.claim_head(out.pooled), dim=-1)

    @torch.no_grad()
    def encode_evidence_for_index(self, input_ids, attention_mask, pixel_values, claim_tokens=None, claim_mask=None) -> torch.Tensor:
        """For corpus encoding without a paired claim, S_j is dropped (use neutral).

        Practically we set S to ones by toggling the module behavior.
        """
        evi_text_out = self.encode_evidence_text(input_ids, attention_mask)
        raw_patches, projected_patches = self.encode_visual(pixel_values)
        R = self.entropy(raw_patches) if self.entropy is not None else None

        if claim_tokens is None:
            # Substitute claim_tokens with text_tokens so S_j -> max self-sim -> ~1
            # but practically we just disable S by passing through weighting with
            # use_S handled via a temporary override.
            saved = self.weighting.use_S
            self.weighting.use_S = False
            try:
                w_out = self.weighting(
                    patches=projected_patches,
                    claim_tokens=evi_text_out.token_embeddings,
                    claim_mask=evi_text_out.attention_mask,
                    text_tokens=evi_text_out.token_embeddings,
                    text_mask=evi_text_out.attention_mask,
                    reliability=R,
                )
            finally:
                self.weighting.use_S = saved
        else:
            w_out = self.weighting(
                patches=projected_patches,
                claim_tokens=claim_tokens,
                claim_mask=claim_mask,
                text_tokens=evi_text_out.token_embeddings,
                text_mask=evi_text_out.attention_mask,
                reliability=R,
            )

        fused = self.fusion(
            weighted_visual=w_out.weighted_visual,
            text_tokens=evi_text_out.token_embeddings,
            text_attention_mask=evi_text_out.attention_mask,
        )
        return F.normalize(fused.pooled, dim=-1)
