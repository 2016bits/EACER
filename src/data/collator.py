from typing import Any, Dict, List

import torch
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase


class RetrievalCollator:
    """Pads claim and evidence sides; preprocesses images via CLIP processor.

    Output batch keys:
      - claim_input_ids, claim_attention_mask
      - claim_comp_input_ids, claim_comp_attention_mask  (q_comp: text overlap masked)
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
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_claim_len = max_claim_len
        self.max_evidence_len = max_evidence_len
        self.build_complementary_query = build_complementary_query

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
        evidence_texts: List[str] = []
        evidence_images = []
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

            if self.build_complementary_query:
                claims_comp.append(self._mask_overlap(sample["claim"], pos["text"]))

            negs = sample.get("negatives", [])
            num_negatives_list.append(len(negs))
            for neg in negs:
                evidence_texts.append(neg["text"])
                evidence_images.append(neg["image"])

        claim_tok = self.tokenizer(
            claims,
            padding=True,
            truncation=True,
            max_length=self.max_claim_len,
            return_tensors="pt",
        )

        out = {
            "claim_ids": claim_ids,
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
            images = [b["image"] for b in batch]
            out["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        if "gold_evidence_ids" in batch[0]:
            out["gold_evidence_ids"] = [b["gold_evidence_ids"] for b in batch]
        return out
