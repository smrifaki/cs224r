"""
Download a small smoke subset from ImageNet-1k validation split.
Saves 200 images to data/smoke/ for local CPU debugging.

Run once (requires HF authentication):
    python scripts/make_smoke_data.py

Output:
    data/smoke/images/          JPEG files named {index:04d}_cls{label}.jpg
    data/smoke/metadata.json    [{filename, label, label_name}, ...]
"""
import json
from pathlib import Path

from datasets import load_dataset

N = 200
OUT_DIR = Path("data/smoke")
IMG_DIR = OUT_DIR / "images"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {N} images from ILSVRC/imagenet-1k (validation, streaming=True)…")
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
    )

    metadata = []
    for i, ex in enumerate(ds):
        if i >= N:
            break

        label: int = ex["label"]
        img = ex["image"].convert("RGB")  # PIL Image

        fname = f"{i:04d}_cls{label:04d}.jpg"
        img.save(IMG_DIR / fname, format="JPEG", quality=95)

        metadata.append({"filename": fname, "label": label})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N}")

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {N} images saved to {IMG_DIR}/")
    print(f"Metadata: {OUT_DIR / 'metadata.json'}")
    print(f"Labels present: {len({m['label'] for m in metadata})} unique classes")


if __name__ == "__main__":
    main()
