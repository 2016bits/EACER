import argparse
import json
import math
import os
import time
from typing import Any, Dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, get_linear_schedule_with_warmup
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from .data import MochegRetrievalDataset, RetrievalCollator
from .data.dataset import ClaimQueries, EvidenceCorpus
from .data.collator import EncodeCollator
from .losses import complementary_alignment_loss, contrastive_retrieval_loss
from .models import ECERRetriever
from .utils import load_config, set_seed
from .utils.config import ensure_dir
from .utils.metrics import retrieval_metrics


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, cfg, tokenizer, image_processor, device, split: str = "val") -> Dict[str, float]:
    data_cfg = cfg["data"]
    path = data_cfg[f"{split}_path"]
    image_root = data_cfg["image_root"]
    use_claim_image = cfg["model"].get("use_claim_image", False)

    corpus = EvidenceCorpus(path, image_root, image_size=data_cfg["image_size"])
    queries = ClaimQueries(
        path,
        with_claim_image=use_claim_image,
        image_root=image_root,
        image_size=data_cfg["image_size"],
    )

    evi_collator = EncodeCollator(
        tokenizer, image_processor,
        max_len=data_cfg["max_evidence_len"],
        text_key="text", with_image=True,
    )
    claim_collator = EncodeCollator(
        tokenizer, image_processor=image_processor if use_claim_image else None,
        max_len=data_cfg["max_claim_len"],
        text_key="claim", with_image=use_claim_image,
    )

    bs = max(cfg["optim"]["batch_size"], 16)
    evi_loader = DataLoader(corpus, batch_size=bs, shuffle=False,
                            num_workers=data_cfg["num_workers"], collate_fn=evi_collator)
    claim_loader = DataLoader(queries, batch_size=bs, shuffle=False,
                              num_workers=data_cfg["num_workers"], collate_fn=claim_collator)

    model.eval()
    evi_ids, evi_vecs = [], []
    for batch in tqdm(evi_loader, desc=f"encode evidence [{split}]"):
        batch = _move(batch, device)
        vec = model.encode_evidence_for_index(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        evi_ids.extend(batch["ids"])
        evi_vecs.append(vec.cpu())
    evi_mat = torch.cat(evi_vecs, dim=0)                  # (N, D)

    claim_ids, claim_vecs, gold_lists = [], [], []
    for batch in tqdm(claim_loader, desc=f"encode claims [{split}]"):
        batch_g = _move(batch, device)
        vec = model.encode_claim_for_index(
            batch_g["input_ids"], batch_g["attention_mask"],
            claim_pixel_values=batch_g.get("pixel_values"),
        )
        claim_ids.extend(batch["ids"])
        claim_vecs.append(vec.cpu())
        gold_lists.extend(batch["gold_evidence_ids"])
    claim_mat = torch.cat(claim_vecs, dim=0)              # (Q, D)

    ks = cfg["eval"]["recall_k"]
    top_k = max(ks)
    sims = claim_mat @ evi_mat.t()                        # (Q, N)
    top = sims.topk(min(top_k, sims.size(1)), dim=-1).indices.tolist()
    ranked = [[evi_ids[i] for i in row] for row in top]

    metrics = retrieval_metrics(ranked, gold_lists, ks=ks)
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, writer, epoch, global_step):
    model.train()
    optim_cfg = cfg["optim"]
    loss_cfg = cfg["loss"]
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for step, batch in enumerate(pbar):
        batch = _move(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(batch)
            l_ret = contrastive_retrieval_loss(
                out.claim_repr,
                out.evidence_repr,
                out.aux["rep_index"],
                batch["num_negatives_per_sample"],
                temperature=loss_cfg["contrastive_temperature"],
            )
            l_comp = torch.tensor(0.0, device=device)
            if loss_cfg.get("use_complementary_loss", False) and out.claim_comp_repr is not None:
                l_comp = complementary_alignment_loss(
                    out.claim_comp_repr,
                    out.weighted_visual,
                    out.aux["rep_index"],
                    batch["num_negatives_per_sample"],
                    temperature=loss_cfg["contrastive_temperature"],
                )
            loss = l_ret + loss_cfg["complementary_lambda"] * l_comp

        loss_to_back = loss / max(optim_cfg["grad_accum_steps"], 1)
        if scaler is not None:
            scaler.scale(loss_to_back).backward()
        else:
            loss_to_back.backward()

        if (step + 1) % max(optim_cfg["grad_accum_steps"], 1) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg["max_grad_norm"])
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if global_step % 20 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_ret", l_ret.item(), global_step)
            writer.add_scalar("train/loss_comp", float(l_comp.item()), global_step)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ret=f"{l_ret.item():.4f}",
                comp=f"{float(l_comp.item()):.4f}",
            )
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["model"]["visual_encoder"])

    use_claim_image = cfg["model"].get("use_claim_image", False)
    train_ds = MochegRetrievalDataset(
        cfg["data"]["train_path"],
        cfg["data"]["image_root"],
        image_size=cfg["data"]["image_size"],
        hard_negatives_per_sample=cfg["data"]["hard_negatives_per_sample"],
        is_train=True,
        with_claim_image=use_claim_image,
    )
    collator = RetrievalCollator(
        tokenizer,
        image_processor,
        max_claim_len=cfg["data"]["max_claim_len"],
        max_evidence_len=cfg["data"]["max_evidence_len"],
        build_complementary_query=cfg["loss"].get("use_complementary_loss", True),
        with_claim_image=use_claim_image,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ECERRetriever(cfg).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"resumed from {args.resume}")

    no_decay = {"bias", "LayerNorm.weight"}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["optim"]["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["optim"]["lr"],
    )

    total_steps = math.ceil(len(train_loader) / max(cfg["optim"]["grad_accum_steps"], 1)) * cfg["optim"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["optim"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb"))
    best_metric, global_step = -1.0, 0
    for epoch in range(1, cfg["optim"]["num_epochs"] + 1):
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, writer, epoch, global_step
        )

        if os.path.isfile(cfg["data"]["val_path"]):
            t0 = time.time()
            metrics = evaluate(model, cfg, tokenizer, image_processor, device, split="val")
            print(f"[epoch {epoch}] val metrics ({time.time() - t0:.1f}s):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
                writer.add_scalar(f"val/{k}", v, epoch)
            main_metric = metrics.get(f"Recall@{cfg['eval']['recall_k'][0]}", 0.0)
            if main_metric > best_metric:
                best_metric = main_metric
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": cfg,
                }
                ckpt_path = os.path.join(cfg["output_dir"], "best.pt")
                torch.save(ckpt, ckpt_path)
                print(f"saved best ckpt -> {ckpt_path}")

        last_path = os.path.join(cfg["output_dir"], "last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, last_path)

    writer.close()


if __name__ == "__main__":
    main()
