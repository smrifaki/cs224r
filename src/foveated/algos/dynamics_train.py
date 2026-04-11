"""Pretrain the forward dynamics model on clean ImageNet random rollouts.

Run as a script:

    python -m foveated.algos.dynamics_train --manifest manifest.json \
        --n-episodes 2000 --out checkpoints/dynamics_v1.pt

The output checkpoint is consumed by `experiments.train_agent` when training
Agents B and C.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.collect_rollouts import collect_triples
from foveated.models.dynamics import ForwardDynamics


def train_dynamics(
    z_t: np.ndarray,
    a_t: np.ndarray,
    z_next: np.ndarray,
    n_actions: int,
    embed_dim: int,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    action_dropout_p: float = 0.5,
) -> ForwardDynamics:
    model = ForwardDynamics(embed_dim=embed_dim, n_actions=n_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(z_t),
        torch.from_numpy(a_t),
        torch.from_numpy(z_next),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for ep in range(epochs):
        losses = []
        for z, a, zn in loader:
            z = z.to(device); a = a.to(device); zn = zn.to(device)
            loss = model.nll_loss(z, a, zn, action_dropout_p=action_dropout_p)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {ep:3d}  nll {np.mean(losses):.4f}")

    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--n-episodes", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/dynamics_v1.pt"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)

    cfg = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )
    env = FoveatedClassificationEnv(cfg=cfg, image_paths=paths, labels=labels, backbone=backbone, device=args.device)
    rng = np.random.default_rng(args.seed)
    z_t, a_t, z_next = collect_triples(env, args.n_episodes, rng)
    print(f"collected {len(z_t)} triples")

    model = train_dynamics(
        z_t, a_t, z_next,
        n_actions=env.n_patches,
        embed_dim=cfg.embed_dim,
        epochs=args.epochs,
        device=args.device,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"embed_dim": cfg.embed_dim, "n_actions": env.n_patches},
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
