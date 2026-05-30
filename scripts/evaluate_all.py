"""Evaluate every trained ECER run on the test split and emit a comparison table.

Discovers ``best.pt`` under each ``outputs/<run>/`` directory, loads the
checkpoint's embedded config, encodes the test corpus + test queries, and
computes Recall@K / Precision@K / NDCG@K / MRR / mAP. Results are written
to:

  outputs/test_results.json       (raw numbers per run)
  outputs/test_results.md         (markdown comparison table)

and printed to stdout.

Usage (from the project root, with the eacer env active)::

    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --runs mr2_base mr2_ecer_A          # subset
    python scripts/evaluate_all.py --split val                          # re-eval val
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

# Silence HF noise the same way training does.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from src.data.collator import EncodeCollator
from src.data.dataset import ClaimQueries, EvidenceCorpus
from src.models import ECERRetriever
from src.utils.metrics import retrieval_metrics


# ---------------------------------------------------------------------------
# Default runs (auto-discovered by scanning outputs/ if --runs not given)
# ---------------------------------------------------------------------------

PRETTY_NAMES = {
    "mr2_base":             "Baseline (no S/D/R/L_comp)",
    "mr2_ablation_ciea":    "CIEA-only (D only)",
    "mr2_ablation_no_S":    "ECER w/o S",
    "mr2_ablation_no_R":    "ECER w/o R",
    "mr2_ablation_no_Lcomp": "ECER w/o L_comp",
    "mr2_ecer_A":           "Full ECER",
}

# Display order (when discovered names overlap with this list).
DISPLAY_ORDER = [
    "mr2_base",
    "mr2_ablation_ciea",
    "mr2_ablation_no_S",
    "mr2_ablation_no_R",
    "mr2_ablation_no_Lcomp",
    "mr2_ecer_A",
]


def _discover_runs(outputs_dir: str) -> List[str]:
    runs = []
    for name in sorted(os.listdir(outputs_dir)):
        ckpt = os.path.join(outputs_dir, name, "best.pt")
        if os.path.isfile(ckpt):
            runs.append(name)
    return runs


def _ordered(runs: List[str]) -> List[str]:
    order = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    return sorted(runs, key=lambda n: (order.get(n, 999), n))


# ---------------------------------------------------------------------------
# Eval one run
# ---------------------------------------------------------------------------

def _move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_one(
    run_dir: str,
    split: str,
    device: torch.device,
    tokenizer_cache: Dict,
    processor_cache: Dict,
    force_test_path: Optional[str] = None,
) -> Dict[str, float]:
    ckpt_path = os.path.join(run_dir, "best.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]

    text_name = cfg["model"]["text_encoder"]
    vis_name = cfg["model"]["visual_encoder"]
    tokenizer = tokenizer_cache.setdefault(text_name, AutoTokenizer.from_pretrained(text_name))
    image_processor = processor_cache.setdefault(vis_name, CLIPImageProcessor.from_pretrained(vis_name))

    model = ECERRetriever(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    data_cfg = cfg["data"]
    path = force_test_path or data_cfg[f"{split}_path"]

    corpus = EvidenceCorpus(path, data_cfg["image_root"], image_size=data_cfg["image_size"])
    queries = ClaimQueries(path)
    print(f"  {split}: {len(queries)} queries, {len(corpus)} evidences")

    evi_coll = EncodeCollator(tokenizer, image_processor,
                              max_len=data_cfg["max_evidence_len"], text_key="text", with_image=True)
    claim_coll = EncodeCollator(tokenizer, image_processor=None,
                                max_len=data_cfg["max_claim_len"], text_key="claim", with_image=False)

    bs = max(cfg["optim"]["batch_size"], 32)
    evi_loader = DataLoader(corpus, batch_size=bs, shuffle=False,
                            num_workers=data_cfg["num_workers"], collate_fn=evi_coll)
    claim_loader = DataLoader(queries, batch_size=bs, shuffle=False,
                              num_workers=data_cfg["num_workers"], collate_fn=claim_coll)

    evi_ids, evi_vecs = [], []
    for batch in tqdm(evi_loader, desc=f"  encode evidence"):
        b = _move(batch, device)
        v = model.encode_evidence_for_index(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"],
            pixel_values=b["pixel_values"],
        )
        evi_ids.extend(batch["ids"])
        evi_vecs.append(v.cpu())
    evi_mat = torch.cat(evi_vecs, dim=0)

    claim_ids, claim_vecs, golds = [], [], []
    for batch in tqdm(claim_loader, desc=f"  encode claims"):
        b = _move(batch, device)
        v = model.encode_claim_for_index(b["input_ids"], b["attention_mask"])
        claim_ids.extend(batch["ids"])
        claim_vecs.append(v.cpu())
        golds.extend(batch["gold_evidence_ids"])
    claim_mat = torch.cat(claim_vecs, dim=0)

    ks = cfg["eval"]["recall_k"]
    top_k = max(ks)
    sims = claim_mat @ evi_mat.t()
    top = sims.topk(min(top_k, sims.size(1)), dim=-1).indices.tolist()
    ranked = [[evi_ids[i] for i in row] for row in top]
    return retrieval_metrics(ranked, golds, ks=ks)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

REPORT_KEYS = ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100",
               "MRR", "mAP", "NDCG@10"]


def _render_markdown(results: Dict[str, Dict[str, float]], split: str) -> str:
    runs = _ordered(list(results.keys()))
    header = ["Run"] + REPORT_KEYS
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for r in runs:
        m = results[r]
        if not m:
            continue
        pretty = PRETTY_NAMES.get(r, r)
        vals = [f"{m.get(k, 0.0):.4f}" for k in REPORT_KEYS]
        md.append("| " + " | ".join([pretty] + vals) + " |")
    md.insert(0, f"# {split.upper()} set retrieval metrics\n")
    return "\n".join(md) + "\n"


def _render_pretty_console(results: Dict[str, Dict[str, float]]) -> str:
    runs = _ordered(list(results.keys()))
    lines = []
    name_w = max(len(PRETTY_NAMES.get(r, r)) for r in runs)
    header = f"{'Run':<{name_w}}  " + "  ".join(f"{k:>10}" for k in REPORT_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for r in runs:
        m = results[r]
        if not m:
            continue
        pretty = PRETTY_NAMES.get(r, r)
        vals = "  ".join(f"{m.get(k, 0.0):>10.4f}" for k in REPORT_KEYS)
        lines.append(f"{pretty:<{name_w}}  {vals}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Specific run names (subfolders of outputs/). Default: auto-discover.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--test_path", default=None,
                        help="Override the test JSONL path (default: read from each ckpt's config)")
    parser.add_argument("--out_json", default=None,
                        help="Default: outputs/<split>_results.json")
    parser.add_argument("--out_md", default=None,
                        help="Default: outputs/<split>_results.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = args.runs or _discover_runs(args.outputs_dir)
    runs = _ordered(runs)
    if not runs:
        raise SystemExit(f"no best.pt found under {args.outputs_dir}/*/")

    print(f"[evaluate_all] split={args.split} device={device} runs={runs}\n")

    results: Dict[str, Dict[str, float]] = {}
    tokenizer_cache: Dict = {}
    processor_cache: Dict = {}

    for run in runs:
        run_dir = os.path.join(args.outputs_dir, run)
        ckpt = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt):
            print(f"[evaluate_all] SKIP {run}: no best.pt")
            continue
        print(f"\n[evaluate_all] === {PRETTY_NAMES.get(run, run)} ({run}) ===")
        t0 = time.time()
        try:
            m = evaluate_one(run_dir, args.split, device, tokenizer_cache, processor_cache,
                             force_test_path=args.test_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[run] = {}
            continue
        results[run] = m
        print(f"  done in {time.time()-t0:.1f}s")
        for k in REPORT_KEYS:
            print(f"    {k:>10s}: {m.get(k, 0.0):.4f}")

    out_json = args.out_json or os.path.join(args.outputs_dir, f"{args.split}_results.json")
    out_md = args.out_md or os.path.join(args.outputs_dir, f"{args.split}_results.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_render_markdown(results, args.split))
    print(f"\n[evaluate_all] wrote {out_json}")
    print(f"[evaluate_all] wrote {out_md}\n")

    print("=" * 80)
    print(f"{args.split.upper()} SET RESULTS")
    print("=" * 80)
    print(_render_pretty_console(results))


if __name__ == "__main__":
    main()
