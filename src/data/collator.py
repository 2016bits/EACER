from typing import Any, Dict, List

import torch
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase


class RetrievalCollator:
    """Pads claim and evidence sides; preprocesses images via CLIP processor.

    Output batch keys:
      - claim_input_ids, claim_attention_mask
      - claim_comp_input_ids, claim_comp_attention_mask  (q_comp: text overlap masked)
      - claim_pixel_values                               (B, 3, H, W) if with_claim_image
      - evidence_input_ids, evidence_attention_mask
      - pixel_values                                     (B+B*neg, 3, H, W)
      - num_negatives_per_sample
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: CLIPImageProcessor,
        max_claim_len: int = 64,
        max_evidence_len: int = 128,
        build_complementary_query: bool = True,
        with_claim_image: bool = False,
        comp_target: str = "full_text",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_claim_len = max_claim_len
        self.max_evidence_len = max_evidence_len
        self.build_complementary_query = build_complementary_query
        self.with_claim_image = with_claim_image
        # ``comp_target`` chooses which evidence string the q_comp masking is
        # built against:
        #   "full_text" — historical EACER behaviour (full title+caption+ocr)
        #   "caption"   — CIEA-faithful: mask only against the short caption
        #                 (falls back to title when caption empty; see
        #                  dataset.py:_materialise's caption_for_comp)
        if comp_target not in ("full_text", "caption"):
            raise ValueError(f"comp_target must be 'full_text' or 'caption', got {comp_target}")
        self.comp_target = comp_target

    def _mask_overlap(self, claim: str, evidence_text: str) -> str:
        """Build q_comp by masking claim tokens that overlap with evidence text.

        Uses tokenizer ID overlap, matching CIEA's recipe.
        """
        claim_ids = self.tokenizer(claim, add_special_tokens=False)["input_ids"]
        evi_ids = set(self.tokenizer(evidence_text, add_special_tokens=False)["input_ids"])
        mask_token = self.tokenizer.mask_token or "[MASK]"
        kept_tokens = []
        for tid in claim_ids:
            if tid in evi_ids:
                kept_tokens.append(mask_token)
            else:
                kept_tokens.append(self.tokenizer.decode([tid], skip_special_tokens=False))
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.tokenize(" ".join(kept_tokens))
        ) or claim

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims: List[str] = []
        claims_comp: List[str] = []
        claim_images = []
        evidence_texts: List[str] = []
        evidence_images = []
        has_image_evidence: List[bool] = []
        num_negatives_list: List[int] = []
        claim_ids: List[str] = []
        positive_evidence_ids: List[str] = []

        for sample in batch:
            claims.append(sample["claim"])
            claim_ids.append(sample["claim_id"])
            pos = sample["positive"]
            positive_evidence_ids.append(pos["evidence_id"])
            evidence_texts.append(pos["text"])
            evidence_images.append(pos["image"])
            has_image_evidence.append(bool(pos.get("has_image", True)))
            if self.with_claim_image:
                claim_images.append(sample["claim_image"])

            if self.build_complementary_query:
                target_text = (
                    pos.get("caption_for_comp") or pos["text"]
                    if self.comp_target == "caption"
                    else pos["text"]
                )
                claims_comp.append(self._mask_overlap(sample["claim"], target_text))

            negs = sample.get("negatives", [])
            num_negatives_list.append(len(negs))
            for neg in negs:
                evidence_texts.append(neg["text"])
                evidence_images.append(neg["image"])
                has_image_evidence.append(bool(neg.get("has_image", True)))

        claim_tok = self.tokenizer(
            claims,
            padding=True,
            truncation=True,
            max_length=self.max_claim_len,
            return_tensors="pt",
        )

        # Map claim_ids to compact 0..U-1 ints for multi-positive InfoNCE masking.
        seen = {}
        claim_int_id = []
        for cid in claim_ids:
            if cid not in seen:
                seen[cid] = len(seen)
            claim_int_id.append(seen[cid])

        out = {
            "claim_ids": claim_ids,
            "claim_int_id": torch.tensor(claim_int_id, dtype=torch.long),
            "positive_evidence_ids": positive_evidence_ids,
            "claim_input_ids": claim_tok["input_ids"],
            "claim_attention_mask": claim_tok["attention_mask"],
            "num_negatives_per_sample": torch.tensor(num_negatives_list, dtype=torch.long),
        }

        if self.build_complementary_query:
            claim_comp_tok = self.tokenizer(
                claims_comp,
                padding=True,
                truncation=True,
                max_length=self.max_claim_len,
                return_tensors="pt",
            )
            out["claim_comp_input_ids"] = claim_comp_tok["input_ids"]
            out["claim_comp_attention_mask"] = claim_comp_tok["attention_mask"]

        evi_tok = self.tokenizer(
            evidence_texts,
            padding=True,
            truncation=True,
            max_length=self.max_evidence_len,
            return_tensors="pt",
        )
        out["evidence_input_ids"] = evi_tok["input_ids"]
        out["evidence_attention_mask"] = evi_tok["attention_mask"]

        pixel = self.image_processor(images=evidence_images, return_tensors="pt")["pixel_values"]
        out["pixel_values"] = pixel
        out["has_image_evidence"] = torch.tensor(has_image_evidence, dtype=torch.bool)

        if self.with_claim_image:
            cpix = self.image_processor(images=claim_images, return_tensors="pt")["pixel_values"]
            out["claim_pixel_values"] = cpix
        return out


class EncodeCollator:
    """Light collator for encoding-only datasets (claim corpus or evidence corpus)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: CLIPImageProcessor = None,
        max_len: int = 128,
        text_key: str = "text",
        with_image: bool = False,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.text_key = text_key
        self.with_image = with_image

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b[self.text_key] for b in batch]
        ids_field = "evidence_id" if "evidence_id" in batch[0] else "claim_id"
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        out = {
            "ids": [str(b[ids_field]) for b in batch],
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }
        if self.with_image:
            assert self.image_processor is not None
            # Evidence corpus uses "image"; claim corpus (image-aware) uses "claim_image".
            image_field = "claim_image" if "claim_image" in batch[0] else "image"
            images = [b[image_field] for b in batch]
            out["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
            # Evidence corpus also wants has_image so the model can zero out the
            # visual contribution for text-only documents.
            if image_field == "image":
                # EvidenceCorpus stores no has_image flag; infer from path presence.
                # If the path exists we treat the image as valid.
                # (EvidenceCorpus already silently returns a black image when the
                # path is missing, so we record has_image=False for those.)
                # Cheap proxy: non-zero mean of the image tensor.
                has_img = (out["pixel_values"].abs().mean(dim=(1, 2, 3)) > 1e-3)
                out["has_image"] = has_img
        if "gold_evidence_ids" in batch[0]:
            out["gold_evidence_ids"] = [b["gold_evidence_ids"] for b in batch]
        return out
