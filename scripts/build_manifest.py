"""Build a JSON manifest of {path, label} entries from an ImageNet val tree.

Expects:
    root/
        n01440764/  *.JPEG
        n01443537/  *.JPEG
        ...

Class folder name maps to label index via root/class_index_map.txt if
present, else alphabetical order over the folder names.

Usage:
    python scripts/build_manifest.py /path/to/imagenet/val > manifest.json
    python scripts/build_manifest.py /path/to/imagenet/val --limit 1000 > smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build(root: Path, class_index: dict[str, int]) -> list[dict]:
    items: list[dict] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_index[cls_dir.name]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                items.append({"path": str(img_path), "label": label})
    return items


def load_index(root: Path) -> dict[str, int]:
    idx_file = root / "class_index_map.txt"
    if idx_file.exists():
        out: dict[str, int] = {}
        for line in idx_file.read_text().splitlines():
            name, idx = line.strip().split()
            out[name] = int(idx)
        return out
    return {p.name: i for i, p in enumerate(sorted(p for p in root.iterdir() if p.is_dir()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--limit", type=int, default=None, help="cap manifest size for quick runs")
    args = p.parse_args()

    idx = load_index(args.root)
    items = build(args.root, idx)
    if args.limit:
        items = items[: args.limit]
    json.dump(items, sys.stdout)


if __name__ == "__main__":
    main()
