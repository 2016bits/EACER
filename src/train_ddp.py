"""DDP training entry point for the ECER retriever.

Key DDP-specific behaviors:

1. **All-gather evidences across ranks** before computing the contrastive
   loss. Each rank still uses its LOCAL claims as anchors but compares them
   against the global pool of evidence representations from every rank.
   This turns the effective in-batch negative pool from
       batch_size - 1
   into
       world_size * batch_size - 1.
   For retrieval, more negatives = stronger contrastive supervision.

2. **DDP wrapper with `find_unused_parameters=True`** because the CLIP
   vision encoder is frozen and contributes no gradient.

3. **Eval runs on rank 0 only.** Other ranks wait at a barrier. Eval is
   a few seconds for the val split, so this is cheap.

4. **Buffers are not broadcast** (`broadcast_buffers=False`). The entropy
   module's `concept_embeds` and `vision_proj` are deterministic per rank
   (built from frozen CLIP weights with a fixed concept list).

Launch via :code:`scripts/train_mr2_ddp.sh` (which wraps :code:`torchrun`).
"""

import argparse
import math
import os
import time
from typing import Any, Dict

# Silence the "tokenizer fork after parallelism" warning from HF.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, get_linear_schedule_with_warmup
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()   # silences the "913 > 512" tokenizer notice

from .data import RetrievalCollator, RetrievalDataset
from .data.collator import EncodeCollator
from .data.dataset import ClaimQueries, EvidenceCorpus
from .losses import complementary_alignment_loss, contrastive_retrieval_loss   # noqa: F401  (kept for parity with single-GPU script)
from .models import ECERRetriever
from .utils import load_config, set_seed
from .utils.config import ensure_dir
from .utils.metrics import retrieval_metrics


# ---------------------------------------------------------------------------
# DDP plumbing
# ---------------------------------------------------------------------------

def _setup_ddp():
    """Initialize the process group from env vars set by `torchrun`."""
    if "WORLD_SIZE" not in os.environ:
        # Fallback: single-process mode (debugging).
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


class _AllGatherWithGrad(torch.autograd.Function):
    """All-gather across ranks; backward returns this rank's grad chunk only.

    Used to broaden the negative pool in contrastive loss while keeping the
    autograd graph well-formed: each rank backprops through its own slice
    of the gathered tensor (matching the slice that lives on this rank).
    DDP's all-reduce on parameter gradients still averages across ranks at
    `optimizer.step()` time.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        # Replace this rank's stub with the grad-bearing tensor so backward
        # can flow through it correctly.
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        chunks = grad_output.chunk(ctx.world_size, dim=0)
        return chunks[ctx.rank].contiguous()


def _all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return _AllGatherWithGrad.apply(tensor)


# ---------------------------------------------------------------------------
# Eval (rank 0 only)
# ---------------------------------------------------------------------------

def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(inner_model, cfg, tokenizer, image_processor, device, split: str = "val") -> Dict[str, float]:
    data_cfg = cfg["data"]
    path = data_cfg[f"{split}_path"]
    image_root = data_cfg["image_root"]
    corpus = EvidenceCorpus(path, image_root, image_size=data_cfg["image_size"])
    queries = ClaimQueries(path)

    evi_coll = EncodeCollator(tokenizer, image_processor,
                              max_len=data_cfg["max_evidence_len"], text_key="text", with_image=True)
    claim_coll = EncodeCollator(tokenizer, image_processor=None,
                                max_len=data_cfg["max_claim_len"], text_key="claim", with_image=False)

    bs = max(cfg["optim"]["batch_size"], 16)
    evi_loader = DataLoader(corpus, batch_size=bs, shuffle=False,
                            num_workers=data_cfg["num_workers"], collate_fn=evi_coll)
    claim_loader = DataLoader(queries, batch_size=bs, shuffle=False,
                              num_workers=data_cfg["num_workers"], collate_fn=claim_coll)

    inner_model.eval()
    evi_ids, evi_vecs = [], []
    for batch in tqdm(evi_loader, desc=f"encode evidence [{split}]"):
        batch = _move(batch, device)
        vec = inner_model.encode_evidence_for_index(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        evi_ids.extend(batch["ids"])
        evi_vecs.append(vec.cpu())
    evi_mat = torch.cat(evi_vecs, dim=0)

    claim_ids, claim_vecs, gold_lists = [], [], []
    for batch in tqdm(claim_loader, desc=f"encode claims [{split}]"):
        batch_g = _move(batch, device)
        vec = inner_model.encode_claim_for_index(batch_g["input_ids"], batch_g["attention_mask"])
        claim_ids.extend(batch["ids"])
        claim_vecs.append(vec.cpu())
        gold_lists.extend(batch["gold_evidence_ids"])
    claim_mat = torch.cat(claim_vecs, dim=0)

    ks = cfg["eval"]["recall_k"]
    top_k = max(ks)
    sims = claim_mat @ evi_mat.t()
    top = sims.topk(min(top_k, sims.size(1)), dim=-1).indices.tolist()
    ranked = [[evi_ids[i] for i in row] for row in top]
    return retrieval_metrics(ranked, gold_lists, ks=ks)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def _contrastive_loss_global(
    claim_repr: torch.Tensor,          # (B_local, D)
    evidence_repr_local: torch.Tensor, # (M_local, D)
    num_negatives_per_sample: torch.Tensor,
    temperature: float,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """InfoNCE against (positives + per-sample hard negs + every other rank's evidences)."""
    evidence_all = _all_gather_with_grad(evidence_repr_local)             # (W*M, D)
    M_local = evidence_repr_local.size(0)
    B = claim_repr.size(0)
    device = claim_repr.device

    # local offset of each sample's positive within this rank's evidence slice
    local_offsets = torch.zeros(B, dtype=torch.long, device=device)
    if B > 1:
        local_offsets[1:] = torch.cumsum(num_negatives_per_sample[:-1] + 1, dim=0)
    targets = rank * M_local + local_offsets                              # global index of positive

    sim = claim_repr @ evidence_all.t() / temperature                     # (B, W*M)
    return F.cross_entropy(sim, targets)


def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, cfg, device,
    rank: int, world_size: int, writer, epoch: int, global_step: int,
):
    model.train()
    optim_cfg = cfg["optim"]
    loss_cfg = cfg["loss"]
    use_amp = device.type == "cuda"
    temp = loss_cfg["contrastive_temperature"]

    is_main = (rank == 0)
    pbar = tqdm(loader, desc=f"epoch {epoch}", disable=not is_main)
    for step, batch in enumerate(pbar):
        batch = _move(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(batch)

            l_ret = _contrastive_loss_global(
                out.claim_repr, out.evidence_repr,
                batch["num_negatives_per_sample"],
                temperature=temp, rank=rank, world_size=world_size,
            )

            l_comp = torch.tensor(0.0, device=device)
            if loss_cfg.get("use_complementary_loss", False) and out.claim_comp_repr is not None:
                l_comp = _contrastive_loss_global(
                    out.claim_comp_repr, out.weighted_visual,
                    batch["num_negatives_per_sample"],
                    temperature=temp, rank=rank, world_size=world_size,
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
        if is_main and global_step % 20 == 0 and writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_ret", l_ret.item(), global_step)
            writer.add_scalar("train/loss_comp", float(l_comp.item()), global_step)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ret=f"{l_ret.item():.4f}",
                comp=f"{float(l_comp.item()):.4f}",
            )
    return global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = _setup_ddp()
    is_main = (rank == 0)

    cfg = load_config(args.config)
    # rank-shifted seed gives every worker a different sampling RNG while
    # keeping model init deterministic across ranks (the model is built
    # AFTER set_seed and SyncedRandom flag, so init is per-rank, which DDP
    # then broadcasts from rank 0).
    set_seed(cfg.get("seed", 42))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        ensure_dir(cfg["output_dir"])
        eff_neg = world_size * cfg["optim"]["batch_size"] - 1
        print(
            f"[ddp] world_size={world_size} | "
            f"batch_per_rank={cfg['optim']['batch_size']} | "
            f"effective in-batch negatives={eff_neg}"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["model"]["visual_encoder"])

    train_ds = RetrievalDataset(
        cfg["data"]["train_path"],
        cfg["data"]["image_root"],
        image_size=cfg["data"]["image_size"],
        hard_negatives_per_sample=cfg["data"]["hard_negatives_per_sample"],
        is_train=True,
    )
    collator = RetrievalCollator(
        tokenizer, image_processor,
        max_claim_len=cfg["data"]["max_claim_len"],
        max_evidence_len=cfg["data"]["max_evidence_len"],
        build_complementary_query=cfg["loss"].get("use_complementary_loss", True),
    )
    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ECERRetriever(cfg).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        if is_main:
            print(f"resumed from {args.resume}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,    # CLIP vision is frozen -> some params get no grad
            broadcast_buffers=False,        # entropy buffers are deterministic per rank
        )
    inner_model = model.module if world_size > 1 else model

    # ---- optimizer / scheduler ------------------------------------------------
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

    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb")) if is_main else None
    best_metric, global_step = -1.0, 0

    for epoch in range(1, cfg["optim"]["num_epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device,
            rank=rank, world_size=world_size, writer=writer, epoch=epoch, global_step=global_step,
        )

        # Evaluate on rank 0 only; everyone else waits.
        if is_main and os.path.isfile(cfg["data"]["val_path"]):
            t0 = time.time()
            metrics = evaluate(inner_model, cfg, tokenizer, image_processor, device, split="val")
            print(f"[epoch {epoch}] val metrics ({time.time() - t0:.1f}s):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
                if writer is not None:
                    writer.add_scalar(f"val/{k}", v, epoch)
            main_metric = metrics.get(f"Recall@{cfg['eval']['recall_k'][0]}", 0.0)
            if main_metric > best_metric:
                best_metric = main_metric
                ckpt_path = os.path.join(cfg["output_dir"], "best.pt")
                torch.save({
                    "model": inner_model.state_dict(),
                    "epoch": epoch, "metrics": metrics, "config": cfg,
                }, ckpt_path)
                print(f"saved best ckpt -> {ckpt_path}")
            last_path = os.path.join(cfg["output_dir"], "last.pt")
            torch.save({"model": inner_model.state_dict(), "epoch": epoch, "config": cfg}, last_path)

        if world_size > 1:
            dist.barrier()

    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
