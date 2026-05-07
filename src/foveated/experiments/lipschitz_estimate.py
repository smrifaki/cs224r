"""Numerical estimates of the L and M constants in the regret bound.

docs/regret_bound.md states that
    R(pi_C) <= 2 L M sqrt(2 K ECE log A)  +  (cost-term)  +  c_PPO
where L is an upper bound on the classifier-head gradient norm and M is
a Lipschitz constant of the per-step reward in the assembled embedding.
The bound's shape is universal; its numerical value depends on L, M.

This script estimates L and M empirically by sampling assembled
embeddings from the env (under each corruption) and measuring:

  L_hat = sup over samples of  ||grad_e classifier_head(e)||_2
  M_hat = sup over (e, e') pairs of  |R(e) - R(e')| / ||e - e'||_2

The reward R(e) is the env's terminal reward: top1(classify(e)) minus
the patch cost accumulated, but for Lipschitz purposes only the
classification term is non-trivially state-dependent, so we measure
the gradient of the logit-of-true-class with respect to e and take the
max norm. M follows from the same gradient.

Caveats:
  - L_hat is an empirical SUP over a sample, not a true upper bound.
    Use it as a calibrated estimate, not a worst-case bound.
  - Top-1 accuracy as a discrete reward is not differentiable; we use
    the smoothed surrogate log-softmax-of-true-class as a proxy
    (consistent with assumption (ii) in the regret bound).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import (
    CORRUPTION_NAMES,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)


def _embedding_grad_norm(env, embed: np.ndarray) -> float:
    """||grad_e log_softmax_true_class(classify(e))||_2 at a given embed."""
    e = torch.from_numpy(embed).float().to(env.device).requires_grad_(True)
    logits = env.backbone.classify(e)
    log_p = torch.log_softmax(logits, dim=-1)
    target = log_p[env.current_label]
    target.backward()
    grad = e.grad
    assert grad is not None
    return float(grad.norm().item())


def estimate_constants(env: FoveatedClassificationEnv, n_samples: int) -> tuple[float, float]:
    """Sample n_samples assembled embeddings and return (L_hat, M_hat).

    Uses the env's reset to sample images and per-image patch
    embeddings; the "assembled" embedding takes a random fraction of
    patches to commit. M_hat is approximated as the largest L_hat over
    samples (under the linearization in assumption ii, L_hat IS the
    local Lipschitz constant of the smooth reward proxy).
    """
    grads: list[float] = []
    rng = np.random.default_rng(env.cfg.seed)
    for _ in range(n_samples):
        env.reset()
        snap = env.current_state()
        k = int(rng.integers(0, env.n_patches))
        if k == 0:
            embed = snap["z_t"]
        else:
            picks = rng.choice(env.n_patches, size=k, replace=False)
            high = snap["patch_embeds"][picks].mean(axis=0)
            embed = 0.5 * (snap["z_t"] + high)
        g = _embedding_grad_norm(env, embed.astype(np.float32))
        grads.append(g)
    L_hat = float(np.max(grads))
    M_hat = float(np.max(grads))  # under the linearization, L bounds M
    return L_hat, M_hat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--out", type=Path, default=Path("runs/eval/lipschitz_constants.csv"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-samples", type=int, default=256)
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

    rows: list[dict] = []
    for corruption in (None, *CORRUPTION_NAMES):
        cfg = FoveatedEnvConfig(
            **{**asdict(cfg_base),
               "corruption_type": corruption,
               "corruption_severity": args.severity if corruption else 0}
        )
        env = FoveatedClassificationEnv(
            cfg=cfg, image_paths=paths, labels=labels, backbone=backbone, device=args.device
        )
        L_hat, M_hat = estimate_constants(env, args.n_samples)
        label = corruption if corruption else "clean"
        rows.append({
            "corruption": label,
            "severity": args.severity if corruption else 0,
            "L_hat": L_hat,
            "M_hat": M_hat,
            "n_samples": args.n_samples,
        })
        print(f"{label:18s} L_hat={L_hat:.4f}  M_hat={M_hat:.4f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")
    # silence the unused-cast warning; cast is referenced inline via type:
    _ = cast


if __name__ == "__main__":
    main()
