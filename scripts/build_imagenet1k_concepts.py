"""Download the ImageNet-1k class names and write them as a concept file.

The entropy module wants K=1000 concept strings, one per line, that the CLIP
text tower can embed. ImageNet-1k labels are CLIP's de-facto canonical concept
vocabulary (CLIP zero-shot accuracy is reported on these).

We try in order:
  1. HuggingFace's `clip-vit-base-patch32` repo (has imagenet_classes.txt)
  2. A pinned GitHub source (OpenAI CLIP)
  3. Fallback: bail with a clear error message

Output: data/imagenet1k_concepts.txt (one concept per line, no commas).
"""

import os
import sys
import urllib.request

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "imagenet1k_concepts.txt")

SOURCES = [
    # OpenAI CLIP's own list, used for their reported zero-shot ImageNet number.
    "https://raw.githubusercontent.com/openai/CLIP/main/notebooks/data/imagenet_classes.txt",
    # Common alternative.
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
]


def _try_download(url: str, timeout: int = 30) -> bytes:
    print(f"[concepts] trying {url}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _normalize(raw: bytes) -> list:
    text = raw.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # OpenAI's file is "tench, Tinca tinca" style — keep only the first comma-split term.
    cleaned = []
    for line in lines:
        first = line.split(",")[0].strip()
        if first:
            cleaned.append(first)
    return cleaned


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f if _.strip())
        if n >= 1000:
            print(f"[concepts] {OUT_PATH} already has {n} concepts; nothing to do")
            return

    for url in SOURCES:
        try:
            raw = _try_download(url)
            concepts = _normalize(raw)
            if len(concepts) >= 1000:
                with open(OUT_PATH, "w", encoding="utf-8") as f:
                    for c in concepts[:1000]:
                        f.write(c + "\n")
                print(f"[concepts] wrote {OUT_PATH} ({len(concepts[:1000])} concepts) from {url}")
                return
            print(f"[concepts] {url} returned only {len(concepts)} concepts; trying next")
        except Exception as e:
            print(f"[concepts] {url} failed: {e}")

    print(
        "[concepts] ERROR — could not download ImageNet-1k labels.\n"
        "If your network is blocked, enable the proxy first:\n"
        "    proxy\n"
        "    python scripts/build_imagenet1k_concepts.py\n"
        "Or download manually and save 1000 lines (one concept per line) at:\n"
        f"    {OUT_PATH}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
