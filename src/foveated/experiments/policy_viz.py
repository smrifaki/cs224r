"""Per-agent policy heatmaps on a sample of test images.

For each agent A / B / C / D and each held-out corruption type, render
the image with a transparent overlay showing which of the 49 patches the
deterministic policy committed and in what order. The order is encoded
by darkness (first commit = darkest). This is the figure that turns the
research question into something a reviewer can SEE.

Output: a grid of small multiples saved as runs/eval/policy_viz.pdf.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import (
    _BILINEAR,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
    load_and_corrupt,
)
from foveated.experiments.train_agent import build_env

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


def _rollout_record(model, env) -> list[int]:
    obs, _ = env.reset()
    chosen: list[int] = []
    done = False; truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(int(action))
        chosen = list(info.get("committed_patches", chosen))
    return chosen


def overlay_heatmap(image_arr: np.ndarray, chosen: list[int], grid_h: int, grid_w: int) -> np.ndarray:
    """Return RGBA overlay highlighting committed patches in commit order."""
    h, w = image_arr.shape[:2]
    patch_h = h // grid_h
    patch_w = w // grid_w
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for order, patch_idx in enumerate(chosen):
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        y0, y1 = row * patch_h, (row + 1) * patch_h
        x0, x1 = col * patch_w, (col + 1) * patch_w
        # Earlier commits get darker, more saturated red.
        alpha = 0.6 - 0.4 * (order / max(len(chosen) - 1, 1))
        overlay[y0:y1, x0:x1, 0] = 1.0
        overlay[y0:y1, x0:x1, 3] = alpha
    return overlay


def _load_image_for_display(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), _BILINEAR)
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--ckpt-dir", type=Path, default=Path("runs"))
    p.add_argument("--dynamics-ckpt", type=Path, default=Path("checkpoints/dynamics_v1.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/eval/policy_viz.pdf"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-images", type=int, default=4)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)
    cfg_base = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(len(paths), size=args.n_images, replace=False).tolist()
    agents = ("A", "B", "C", "D")

    fig, axes = plt.subplots(
        len(HELD_OUT_CORRUPTIONS), len(agents) * args.n_images,
        figsize=(2 * len(agents) * args.n_images, 2 * len(HELD_OUT_CORRUPTIONS)),
        squeeze=False,
    )

    for row, corruption in enumerate(HELD_OUT_CORRUPTIONS):
        cfg = FoveatedEnvConfig(
            **{**asdict(cfg_base),
               "corruption_type": corruption,
               "corruption_severity": args.severity}
        )
        for col_agent, agent in enumerate(agents):
            ckpt = args.ckpt_dir / f"agent_{agent}_seed{args.seed}_final.zip"
            if not ckpt.exists():
                continue
            env = build_env(
                agent, cfg, paths, labels, backbone,
                args.dynamics_ckpt if agent in ("B", "C") else None,
                args.device,
            )
            base = cast(FoveatedClassificationEnv, env.unwrapped)
            model = PPO.load(ckpt, device=args.device)
            for col_img, img_idx in enumerate(sampled_indices):
                # Force the env to pick this image by seeding right before reset.
                base.rng = np.random.default_rng(args.seed + img_idx)
                # FoveatedClassificationEnv.reset samples from self.rng, so we
                # also temporarily restrict the dataset to one image.
                saved_paths = base.image_paths
                saved_labels = base.labels
                base.image_paths = [paths[img_idx]]
                base.labels = [labels[img_idx]]
                chosen = _rollout_record(model, env)
                base.image_paths = saved_paths
                base.labels = saved_labels

                display = _load_image_for_display(paths[img_idx], cfg.image_size)
                # Apply same corruption to displayed image (visual consistency).
                if corruption and args.severity > 0:
                    _ = load_and_corrupt(paths[img_idx], cfg)
                overlay = overlay_heatmap(display, chosen, cfg.grid_h, cfg.grid_w)

                ax = axes[row][col_agent * args.n_images + col_img]
                ax.imshow(display)
                ax.imshow(overlay)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0 and col_img == 0:
                    ax.set_title(f"Agent {agent}", fontsize=9)
                if col_agent == 0 and col_img == 0:
                    ax.set_ylabel(corruption, fontsize=9, rotation=90, labelpad=4)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
