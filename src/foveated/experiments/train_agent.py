"""Goal-conditioned PPO trainer for Agents A / B / C / D.

Single entry point: `python -m foveated.experiments.train_agent --agent {A,B,C,D}`.
All agents share backbone, env, and hyperparameters; they differ only in
the observation / reward wrapper. The cardinal rule of Phase 4 (hold
everything constant except the channel under study) is enforced by routing
all four agents through `build_env` below.

    Agent A: baseline, no extra features.
    Agent B: intrinsic reward = beta * ||precision-weighted residual||^2.
    Agent C: per-patch prospective uncertainty u(a) appended to obs.
    Agent D: classifier entropy of current assembly appended to obs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.feature_wrappers import (
    ClassifierEntropyObsWrapper,
    ProspectiveUncertaintyObsWrapper,
)
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.envs.intrinsic_wrapper import IntrinsicRewardWrapper
from foveated.models.dynamics import ForwardDynamics
from foveated.models.dynamics_bayesian import LaplacePosterior


def build_env(
    agent: str,
    cfg: FoveatedEnvConfig,
    image_paths: list[Path],
    labels: list[int],
    backbone,
    dynamics_ckpt: Path | None,
    device: str = "cuda",
    laplace_ckpt: Path | None = None,
) -> gym.Env:
    env = FoveatedClassificationEnv(
        cfg=cfg,
        image_paths=image_paths,
        labels=labels,
        backbone=backbone,
        device=device,
    )
    if agent == "A":
        return env
    if agent == "D":
        return ClassifierEntropyObsWrapper(env, device=device)

    assert dynamics_ckpt is not None and dynamics_ckpt.exists(), (
        f"agent {agent} requires a trained dynamics checkpoint at {dynamics_ckpt}"
    )
    ckpt = torch.load(dynamics_ckpt, map_location=device)
    dyn = ForwardDynamics(**ckpt["config"]).to(device)
    dyn.load_state_dict(ckpt["state_dict"])

    posterior: LaplacePosterior | None = None
    if laplace_ckpt is not None and laplace_ckpt.exists():
        lap = torch.load(laplace_ckpt, map_location=device)
        posterior = LaplacePosterior(
            posterior_cov=lap["posterior_cov"].to(device),
            prior_precision=float(lap["prior_precision"]),
        )

    if agent == "B":
        return IntrinsicRewardWrapper(env, dyn, device=device)
    if agent == "C":
        return ProspectiveUncertaintyObsWrapper(env, dyn, posterior=posterior, device=device)
    raise ValueError(f"unknown agent {agent}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--dynamics-ckpt", type=Path, default=Path("checkpoints/dynamics_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--corruption", default=None)
    p.add_argument("--severity", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)
    cfg = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
        corruption_type=args.corruption,
        corruption_severity=args.severity,
    )

    def make():
        return build_env(args.agent, cfg, paths, labels, backbone, args.dynamics_ckpt, args.device)

    vec_env = DummyVecEnv([make])
    model = PPO(
        "MlpPolicy", vec_env, seed=args.seed, verbose=1,
        tensorboard_log=str(args.out_dir / "tb"),
        n_steps=2048, batch_size=256, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )
    cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(args.out_dir / "ckpt"),
        name_prefix=f"agent_{args.agent}_seed{args.seed}",
    )
    model.learn(total_timesteps=args.total_steps, progress_bar=True, callback=cb)
    model.save(args.out_dir / f"agent_{args.agent}_seed{args.seed}_final")


if __name__ == "__main__":
    main()
