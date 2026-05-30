# Data layout for ECER retrieval

```
data/
├── raw/
│   └── mr2 -> /mnt/data/yangjun/data/mr2/queries_dataset_merge   (symlink)
└── processed/
    └── mr2/
        ├── mr2_train.jsonl
        ├── mr2_val.jsonl
        └── mr2_test.jsonl
```

`raw/` is a symlink to the on-disk dataset cache. `processed/` holds the JSONL
files consumed by `src/data/dataset.py`.

## MR2

[MR2](https://github.com/THU-BPM/MR2) is a multimodal misinformation dataset
collected from Twitter and Weibo. Each query (claim) consists of an image and a
short caption; the dataset additionally provides per-claim retrieved evidence
from a forward image search and an inverse image search.

We treat MR2 as an **evidence retrieval** corpus:
- The claim is the caption text (the image is currently unused on the claim
  side, in line with the text-only claim encoder in README §5.1).
- Each retrieved web page (from both `direct_annotation.json` and
  `inverse_annotation.json`) becomes an evidence with optional image + text.
- All evidences attached to a claim's annotation files are treated as
  positives. Negatives are drawn in-batch during contrastive training.

### Conversion

```bash
bash scripts/preprocess_mr2.sh                     # uses defaults below
# or
python scripts/preprocess_mr2.py \
    --mr2_root /mnt/data/yangjun/data/mr2/queries_dataset_merge \
    --out_dir  data/processed/mr2 \
    --include_query_image
```

After conversion you should see ~11k / 1.3k / 1.1k claims for train/val/test.

### Evidence schema

Each evidence is one of two kinds:

| Source                | Image       | Text fields used                                             |
|-----------------------|-------------|--------------------------------------------------------------|
| `mr2_direct/*`        | `.jpg` file | `page_title` + `snippet` + flattened `caption`               |
| `mr2_inverse/*`       | (none)      | `title` + flattened `caption`                                |
| `mr2_inverse/summary` | (none)      | Google `best_guess_lbl` + extracted `entities` (one per claim) |

`caption` may originally be a dict (`{"alt_node": "..."}` or `{"title_node": "..."}`);
the preprocessor flattens it into a string and concatenates everything into
`evidence.text` (consumed by `_evidence_text` in `src/data/dataset.py`).

### Notes / known simplifications

1. The claim's own query image is exported as `claim_image_path` for future
   multimodal-claim extensions but is **not** currently used by the model.
2. MR2 inverse-search pages have only HTML files (no paired image), so they
   appear as text-only evidence. A black placeholder image is used at load time
   so the batch stays a regular tensor.
3. MR2 has no per-evidence relevance labels. All evidences from a claim's two
   annotation files are treated as positives; the contrastive loss relies on
   in-batch negatives (other claims' evidences).
