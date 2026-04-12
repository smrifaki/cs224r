"""ImageNet manifest loader + backbone helper.

The manifest is a JSON list of {"path": str, "label": int} entries built by
`scripts/build_manifest.py`. Keeping the file list out-of-band lets us pin
seeds across runs without committing the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from foveated.envs.foveated_env import BackboneAdapter


def load_manifest(manifest: Path) -> tuple[list[Path], list[int]]:
    items = json.loads(manifest.read_text())
    paths = [Path(it["path"]) for it in items]
    labels = [int(it["label"]) for it in items]
    return paths, labels


def load_backbone(name: str, device: str) -> BackboneAdapter:
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=1000)
    is_vit = "vit" in name.lower()
    return BackboneAdapter(model, is_vit=is_vit, device=device)
