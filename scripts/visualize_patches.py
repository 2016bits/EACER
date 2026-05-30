"""Visualize ECER's per-patch weighting on validation samples.

Picks N claims, runs the trained model on each (claim, positive evidence)
pair, extracts S, D, R, and the final alpha from the weighting module,
and saves a per-sample PNG with the original evidence image alongside
heatmap overlays.

Usage::

    python scripts/visualize_patches.py \\
        --ckpt outputs/mr2_ecer_A/best.pt \\
        --jsonl data/processed/mr2/mr2_val.jsonl \\
        --n_samples 20 \\
        --out_dir outputs/visualizations
"""

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import _evidence_text, _load_image, _load_jsonl
from src.models import ECERRetriever


def _heatmap_to_img(map_1d: np.ndarray, img_size: int) -> np.ndarray:
    """Convert (P,) patch weights to a (img_size, img_size) heatmap.

    CLIP ViT-B/32 with image_size=224 -> 7x7=49 patches.
    """
    P = map_1d.shape[0]
    side = int(np.sqrt(P))
    assert side * side == P, f"patch count {P} is not a perfect square"
    grid = torch.from_numpy(map_1d.reshape(side, side)).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(grid, size=(img_size, img_size), mode="bicubic", align_corners=False)
    return up.squeeze().numpy()


def _overlay(image: Image.Image, heat: np.ndarray, ax, title: str):
    arr = np.array(image.convert("RGB"))
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    ax.imshow(arr)
    ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/mr2_ecer_A/best.pt")
    p.add_argument("--jsonl", default="data/processed/mr2/mr2_val.jsonl")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--out_dir", default="outputs/visualizations")
    p.add_argument("--seed", type=int, default=0,
                   help="Shuffle jsonl items with this RNG before picking n_samples.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["model"]["visual_encoder"])

    model = ECERRetriever(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    print(f"loaded {args.ckpt} (epoch {state.get('epoch')})")

    raw = _load_jsonl(args.jsonl)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(raw)

    img_size = cfg["data"]["image_size"]
    max_claim_len = cfg["data"]["max_claim_len"]
    max_evi_len = cfg["data"]["max_evidence_len"]

    rendered = 0
    for item in raw:
        if rendered >= args.n_samples:
            break
        claim = str(item.get("claim", "")).strip() or "[empty]"
        positives = item.get("positive_evidence") or []
        positives = [pos for pos in positives if pos.get("image_path")]
        if not positives:
            continue
        pos = positives[0]
        evidence_text = _evidence_text(pos) or "[no text]"
        image = _load_image("", pos.get("image_path"), img_size)

        c_tok = tokenizer(claim, padding=True, truncation=True,
                          max_length=max_claim_len, return_tensors="pt")
        e_tok = tokenizer(evidence_text, padding=True, truncation=True,
                          max_length=max_evi_len, return_tensors="pt")
        c_comp = tokenizer(claim, padding=True, truncation=True,
                           max_length=max_claim_len, return_tensors="pt")
        pix = image_processor(images=[image], return_tensors="pt")["pixel_values"]

        batch = {
            "claim_input_ids": c_tok["input_ids"].to(device),
            "claim_attention_mask": c_tok["attention_mask"].to(device),
            "evidence_input_ids": e_tok["input_ids"].to(device),
            "evidence_attention_mask": e_tok["attention_mask"].to(device),
            "pixel_values": pix.to(device),
            "num_negatives_per_sample": torch.tensor([0], dtype=torch.long, device=device),
            "claim_comp_input_ids": c_comp["input_ids"].to(device),
            "claim_comp_attention_mask": c_comp["attention_mask"].to(device),
        }
        with torch.no_grad():
            out = model(batch)

        # M = 1; index 0 is this evidence.
        S = out.aux["S"][0].cpu().numpy() if out.aux["S"] is not None else None
        D = out.aux["D"][0].cpu().numpy() if out.aux["D"] is not None else None
        R = out.aux["R"][0].cpu().numpy() if out.aux["R"] is not None else None
        alpha = out.alpha[0].cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(np.array(image.convert("RGB")))
        axes[0].set_title("evidence image", fontsize=10)
        axes[0].axis("off")
        if S is not None:
            _overlay(image, _heatmap_to_img(S, img_size), axes[1], f"S (claim relevance), max={S.max():.2f}")
        else:
            axes[1].set_title("S off"); axes[1].axis("off")
        if D is not None:
            _overlay(image, _heatmap_to_img(D, img_size), axes[2], f"D (complementarity), max={D.max():.2f}")
        else:
            axes[2].set_title("D off"); axes[2].axis("off")
        if R is not None:
            _overlay(image, _heatmap_to_img(R, img_size), axes[3], f"R (reliability), max={R.max():.2f}")
        else:
            axes[3].set_title("R off"); axes[3].axis("off")
        _overlay(image, _heatmap_to_img(alpha, img_size), axes[4], f"alpha (final), max={alpha.max():.3f}")

        fig.suptitle(f"claim: {claim[:120]}\nevidence: {evidence_text[:120]}", fontsize=9)
        out_path = os.path.join(args.out_dir, f"{item.get('claim_id', f'sample{rendered}')}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        rendered += 1
        print(f"[{rendered}/{args.n_samples}] -> {out_path}")

    print(f"\ndone — {rendered} samples under {args.out_dir}/")


if __name__ == "__main__":
    main()
