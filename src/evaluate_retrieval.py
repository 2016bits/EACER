import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from .data.collator import EncodeCollator
from .data.dataset import ClaimQueries, EvidenceCorpus
from .models import ECERRetriever
from .utils import load_config
from .utils.metrics import retrieval_metrics


def _move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def encode_corpus(model, corpus_loader, device):
    ids, vecs = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(corpus_loader, desc="encode evidence"):
            b = _move(batch, device)
            v = model.encode_evidence_for_index(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                pixel_values=b["pixel_values"],
            )
            ids.extend(batch["ids"])
            vecs.append(v.cpu())
    return ids, torch.cat(vecs, dim=0)


def encode_queries(model, query_loader, device):
    ids, vecs, golds = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="encode claims"):
            b = _move(batch, device)
            v = model.encode_claim_for_index(b["input_ids"], b["attention_mask"])
            ids.extend(batch["ids"])
            vecs.append(v.cpu())
            golds.extend(batch["gold_evidence_ids"])
    return ids, torch.cat(vecs, dim=0), golds


def build_index_and_search(claim_mat, evi_mat, top_k: int, use_faiss: bool):
    if use_faiss:
        try:
            import faiss
            d = evi_mat.size(1)
            index = faiss.IndexFlatIP(d)
            index.add(evi_mat.numpy().astype("float32"))
            D, I = index.search(claim_mat.numpy().astype("float32"), top_k)
            return I.tolist()
        except Exception as e:
            print(f"faiss unavailable ({e}); falling back to torch matmul")
    sims = claim_mat @ evi_mat.t()
    return sims.topk(min(top_k, sims.size(1)), dim=-1).indices.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["model"]["visual_encoder"])

    model = ECERRetriever(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)

    path = cfg["data"][f"{args.split}_path"]
    image_root = cfg["data"]["image_root"]

    corpus = EvidenceCorpus(path, image_root, image_size=cfg["data"]["image_size"])
    queries = ClaimQueries(path)

    evi_collator = EncodeCollator(tokenizer, image_processor,
                                  max_len=cfg["data"]["max_evidence_len"],
                                  text_key="text", with_image=True)
    claim_collator = EncodeCollator(tokenizer, image_processor=None,
                                    max_len=cfg["data"]["max_claim_len"],
                                    text_key="claim", with_image=False)

    bs = max(cfg["optim"]["batch_size"], 16)
    corpus_loader = DataLoader(corpus, batch_size=bs, shuffle=False,
                               num_workers=cfg["data"]["num_workers"], collate_fn=evi_collator)
    query_loader = DataLoader(queries, batch_size=bs, shuffle=False,
                              num_workers=cfg["data"]["num_workers"], collate_fn=claim_collator)

    evi_ids, evi_mat = encode_corpus(model, corpus_loader, device)
    claim_ids, claim_mat, golds = encode_queries(model, query_loader, device)

    ks = cfg["eval"]["recall_k"]
    top_k = max(ks)
    indices = build_index_and_search(claim_mat, evi_mat, top_k, cfg["eval"].get("use_faiss", True))
    ranked = [[evi_ids[i] for i in row] for row in indices]

    metrics = retrieval_metrics(ranked, golds, ks=ks)
    print(f"[{args.split}] retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for cid, ranks, gd in zip(claim_ids, ranked, golds):
                f.write(json.dumps({
                    "claim_id": cid,
                    "ranked_evidence_ids": ranks,
                    "gold_evidence_ids": gd,
                }) + "\n")
        print(f"predictions written to {args.out}")


if __name__ == "__main__":
    main()
