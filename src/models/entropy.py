import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


IMAGENET_TEMPLATES_FALLBACK = [
    "a photo of a {}.",
    "a picture of a {}.",
]


def _load_concept_words(path: Optional[str]) -> List[str]:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    # Minimal fallback so the system can run without an external concept file.
    # Users should provide an ImageNet-1k list at `path` for full coverage.
    return [
        "person", "man", "woman", "child", "crowd", "car", "truck", "bus", "bike", "train",
        "airplane", "boat", "building", "house", "skyscraper", "bridge", "road", "sign", "tree",
        "flower", "grass", "mountain", "river", "ocean", "sky", "cloud", "sun", "moon", "fire",
        "smoke", "explosion", "weapon", "gun", "knife", "bomb", "soldier", "police", "protest",
        "flag", "logo", "text", "screen", "tv", "phone", "computer", "table", "chair", "food",
        "drink", "cup", "bottle", "plate", "dog", "cat", "horse", "bird", "fish", "cow",
    ]


class EntropyReliability(nn.Module):
    """Computes the entropy-based reliability score R_j for each patch.

    Two schemes:
      A. High-entropy suppression: R_j = 1 - H_j / log(K)
      B. Mid-entropy preference:  R_j = exp(-(H_j - mu)^2 / (2 sigma^2))

    Concept distribution p_j is built by computing CLIP zero-shot similarity
    between projected patch features (mapped back to CLIP space via a linear
    `inv_projector`) OR directly between raw CLIP patches and pre-computed
    concept text embeddings.

    To stay efficient we pre-compute concept embeddings once and reuse them.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        concepts_path: Optional[str] = None,
        K: int = 1000,
        scheme: str = "A",
        mu: float = 4.0,
        sigma: float = 1.5,
        templates: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        assert scheme in {"A", "B"}, f"unknown entropy scheme: {scheme}"
        self.scheme = scheme
        self.mu = mu
        self.sigma = sigma
        self.templates = templates or IMAGENET_TEMPLATES_FALLBACK

        concepts = _load_concept_words(concepts_path)[:K] if K else _load_concept_words(concepts_path)
        self.K = len(concepts)
        self.log_K = math.log(max(self.K, 2))

        concept_embeds, vision_proj = self._build_concept_embeddings(
            clip_model_name, concepts, cache_path=cache_path, device=device
        )
        # CLIP's vision tower outputs in `vision_hidden` dim (e.g. 768 for B/32)
        # while text features live in `projection_dim` (e.g. 512). To compute
        # zero-shot CLIP similarity we must first apply CLIP's visual projection
        # to the patch features.
        self.register_buffer("concept_embeds", concept_embeds, persistent=False)  # (K, D_proj)
        self.register_buffer("vision_proj", vision_proj, persistent=False)        # (D_proj, D_vis)

    @torch.no_grad()
    def _build_concept_embeddings(
        self,
        clip_model_name: str,
        concepts: List[str],
        cache_path: Optional[str],
        device: Optional[str],
    ):
        if cache_path and os.path.isfile(cache_path):
            blob = torch.load(cache_path, map_location="cpu")
            if isinstance(blob, dict) and "concept_embeds" in blob and "vision_proj" in blob:
                return blob["concept_embeds"], blob["vision_proj"]
            # Old cache (concept_embeds only) — fall through to rebuild so we
            # can grab vision_proj too.
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        full_model = CLIPModel.from_pretrained(clip_model_name).to(dev).eval()
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        all_embeds = []
        for concept in concepts:
            prompts = [t.format(concept) for t in self.templates]
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(dev)
            embeds = full_model.get_text_features(**inputs)
            embeds = F.normalize(embeds, dim=-1).mean(dim=0)
            embeds = F.normalize(embeds, dim=-1)
            all_embeds.append(embeds.cpu())
        # CLIP's `visual_projection` maps the vision tower's hidden space
        # (e.g. 768) onto the joint embedding space (e.g. 512) used by text.
        vision_proj = full_model.visual_projection.weight.detach().cpu().clone()   # (D_proj, D_vis)
        # Free CLIP towers; we keep only the K text embeddings + the projection.
        del full_model
        if dev == "cuda":
            torch.cuda.empty_cache()
        concept_embeds = torch.stack(all_embeds, dim=0)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"concept_embeds": concept_embeds, "vision_proj": vision_proj}, cache_path)
        return concept_embeds, vision_proj  # (K, D_proj), (D_proj, D_vis)

    def forward(self, raw_patches: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
        """raw_patches: (B, P, D_vis) — straight from the CLIP vision encoder.

        Returns reliability R: (B, P) in [0, 1].
        """
        # Project from CLIP's vision hidden space (D_vis) into the joint
        # embedding space (D_proj) used by concept_embeds. `vision_proj` is
        # stored as (D_proj, D_vis), matching CLIP's `visual_projection.weight`.
        projected = raw_patches @ self.vision_proj.t()                # (B, P, D_proj)
        patches = F.normalize(projected, dim=-1)
        # (B, P, K)
        sim = torch.einsum("bpd,kd->bpk", patches, self.concept_embeds)
        probs = F.softmax(sim / temperature, dim=-1)
        # Entropy per patch
        H = -(probs * (probs.clamp(min=1e-12)).log()).sum(dim=-1)   # (B, P)

        if self.scheme == "A":
            R = 1.0 - H / self.log_K
        else:
            R = torch.exp(-((H - self.mu) ** 2) / (2.0 * self.sigma ** 2))
        return R.clamp(min=0.0, max=1.0)
