import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _evidence_text(ev: Dict[str, Any]) -> str:
    parts = []
    for key in ("title", "caption", "text", "ocr"):
        v = ev.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip() or "[no text]"


def _load_image(image_root: str, rel_path: Optional[str], image_size: int) -> Image.Image:
    if rel_path:
        full = rel_path if os.path.isabs(rel_path) else os.path.join(image_root, rel_path)
        if os.path.isfile(full):
            try:
                return Image.open(full).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (image_size, image_size), (0, 0, 0))


@dataclass
class RetrievalSample:
    claim_id: str
    claim: str
    positive: Dict[str, Any]
    negatives: List[Dict[str, Any]]
    claim_image_path: Optional[str] = None


class RetrievalDataset(Dataset):
    """Reads JSONL items conforming to README §13.

    Each line:
      {claim_id, claim, label, positive_evidence: [...], negative_evidence: [...]}

    For each (claim, positive_evidence) pair we yield one sample. Hard negatives
    are sampled from `negative_evidence` if available; otherwise the collator
    relies on in-batch negatives.

    Works with MOCHEG, MR2, or any other dataset converted to this schema.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        image_size: int = 224,
        hard_negatives_per_sample: int = 0,
        is_train: bool = True,
        with_claim_image: bool = False,
    ):
        super().__init__()
        self.image_root = image_root
        self.image_size = image_size
        self.hard_negatives_per_sample = hard_negatives_per_sample
        self.is_train = is_train
        self.with_claim_image = with_claim_image

        raw = _load_jsonl(jsonl_path)
        self.samples: List[RetrievalSample] = []
        for item in raw:
            claim_id = str(item.get("claim_id"))
            claim = str(item.get("claim", "")).strip()
            positives = item.get("positive_evidence", []) or []
            negatives = item.get("negative_evidence", []) or []
            if not claim or not positives:
                continue
            claim_img = item.get("claim_image_path") if with_claim_image else None
            for pos in positives:
                self.samples.append(
                    RetrievalSample(
                        claim_id=claim_id,
                        claim=claim,
                        positive=pos,
                        negatives=list(negatives),
                        claim_image_path=claim_img,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _materialise(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "evidence_id": str(ev.get("evidence_id", "")),
            "text": _evidence_text(ev),
            "image": _load_image(self.image_root, ev.get("image_path"), self.image_size),
            "has_text": bool(ev.get("text") or ev.get("caption") or ev.get("ocr")),
            "has_image": bool(ev.get("image_path")),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        pos = self._materialise(s.positive)
        sampled_neg: List[Dict[str, Any]] = []
        if self.hard_negatives_per_sample > 0 and s.negatives:
            pool = list(s.negatives)
            if self.is_train:
                random.shuffle(pool)
            for neg in pool[: self.hard_negatives_per_sample]:
                sampled_neg.append(self._materialise(neg))
        out = {
            "claim_id": s.claim_id,
            "claim": s.claim,
            "positive": pos,
            "negatives": sampled_neg,
        }
        if self.with_claim_image:
            out["claim_image"] = _load_image(
                self.image_root, s.claim_image_path, self.image_size
            )
        return out


class EvidenceCorpus(Dataset):
    """Flat corpus of all unique evidences across positives + negatives.

    Used at evaluation time to encode the whole pool once and build an index.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_root = image_root
        self.image_size = image_size

        raw = _load_jsonl(jsonl_path)
        seen = set()
        self.items: List[Dict[str, Any]] = []
        for item in raw:
            for key in ("positive_evidence", "negative_evidence"):
                for ev in item.get(key, []) or []:
                    eid = str(ev.get("evidence_id", ""))
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    self.items.append(ev)

        self.id_to_index = {str(ev.get("evidence_id")): i for i, ev in enumerate(self.items)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ev = self.items[idx]
        return {
            "evidence_id": str(ev.get("evidence_id", "")),
            "text": _evidence_text(ev),
            "image": _load_image(self.image_root, ev.get("image_path"), self.image_size),
        }


class ClaimQueries(Dataset):
    """Flat list of (claim_id, claim, gold_evidence_ids) for evaluation.

    When ``with_claim_image=True`` also materialises ``claim_image`` (PIL) so the
    eval-time collator can build claim pixel values for image-aware models.
    """

    def __init__(
        self,
        jsonl_path: str,
        with_claim_image: bool = False,
        image_root: str = "",
        image_size: int = 224,
    ):
        super().__init__()
        self.with_claim_image = with_claim_image
        self.image_root = image_root
        self.image_size = image_size

        raw = _load_jsonl(jsonl_path)
        self.items: List[Dict[str, Any]] = []
        for item in raw:
            cid = str(item.get("claim_id"))
            claim = str(item.get("claim", "")).strip()
            golds = [str(e.get("evidence_id")) for e in (item.get("positive_evidence") or []) if e.get("evidence_id")]
            if not claim or not golds:
                continue
            entry = {"claim_id": cid, "claim": claim, "gold_evidence_ids": golds}
            if with_claim_image:
                entry["claim_image_path"] = item.get("claim_image_path")
            self.items.append(entry)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        if not self.with_claim_image:
            return item
        # Lazy-load to keep RAM bounded
        return {
            **item,
            "claim_image": _load_image(self.image_root, item.get("claim_image_path"), self.image_size),
        }


# Backwards-compatible alias kept so older configs/scripts keep working.
MochegRetrievalDataset = RetrievalDataset
