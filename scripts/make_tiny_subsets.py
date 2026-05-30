"""Slice the MR2 JSONL files into tiny subsets for smoke training.

Used to verify the real encoder pipeline (xlm-roberta + CLIP) end-to-end
before committing to a full training run. The subsets are deterministic
(first-N claims) so successive runs are reproducible.
"""

import argparse
import os
import shutil


def slice_jsonl(src: str, dst: str, n: int) -> int:
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if written >= n:
                break
            fout.write(line)
            written += 1
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="data/processed/mr2",
                   help="dir containing mr2_{train,val,test}.jsonl")
    p.add_argument("--dst_dir", default="data/processed/mr2_tiny",
                   help="dir to write the tiny subsets to")
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_val",   type=int, default=100)
    p.add_argument("--n_test",  type=int, default=100)
    p.add_argument("--force",   action="store_true",
                   help="overwrite existing tiny files")
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        src = os.path.join(args.src_dir, f"mr2_{split}.jsonl")
        dst = os.path.join(args.dst_dir, f"mr2_{split}.jsonl")
        if os.path.isfile(dst) and not args.force:
            print(f"skip {dst} (exists; pass --force to overwrite)")
            continue
        if not os.path.isfile(src):
            raise SystemExit(f"missing source: {src}; run scripts/preprocess_mr2.sh first")
        wrote = slice_jsonl(src, dst, n)
        print(f"{dst}: {wrote} claims")


if __name__ == "__main__":
    main()
