"""Fit a last-layer Laplace posterior on top of a trained dynamics model.

One-time post-training step. Reads the dynamics checkpoint produced by
algos/dynamics_train.py, replays the random-policy rollouts to gather
features, computes the Gauss-Newton posterior covariance for the
mean_head weights, and saves it alongside.

This is what Agent C should consume to get BALD-aligned epistemic
uncertainty instead of total predictive uncertainty. See
docs/regret_bound.md and models/dynamics_bayesian.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.collect_rollouts import collect_triples
from foveated.models.dynamics import ForwardDynamics
from foveated.models.dynamics_bayesian import fit_last_layer_laplace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/laplace_v1.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=2000)
    p.add_argument("--prior-precision", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)

    ckpt = torch.load(args.dynamics_ckpt, map_location=args.device)
    dyn = ForwardDynamics(**ckpt["config"]).to(args.device).eval()
    dyn.load_state_dict(ckpt["state_dict"])

    cfg = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )
    env = FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=labels, backbone=backbone, device=args.device
    )
    rng = np.random.default_rng(args.seed)
    z_t_np, a_t_np, _ = collect_triples(env, args.n_episodes, rng)
    z_t = torch.from_numpy(z_t_np).to(args.device)
    a_t = torch.from_numpy(a_t_np).to(args.device)
    print(f"collected {len(z_t_np)} triples; fitting Laplace posterior")

    posterior = fit_last_layer_laplace(dyn, z_t, a_t, prior_precision=args.prior_precision)
    torch.save(
        {
            "posterior_cov": posterior.posterior_cov.cpu(),
            "prior_precision": posterior.prior_precision,
            "config": ckpt["config"],
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
