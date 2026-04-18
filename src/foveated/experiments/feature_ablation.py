"""Mechanism evidence: does Agent C actually use the prospective feature?

At inference time, zero out the prospective uncertainty slot in Agent C's
observation and re-evaluate. If Agent C's Pareto / regret degrades
substantially relative to the baseline-with-zeroed-feature comparison,
the policy is genuinely using the feature; if not, the policy ignored it
and any Agent-C-vs-A gap is from a confound (different obs dim, different
PPO trajectory).

This is the cleanest single-experiment proof that the feature is the
mechanism, not just a passive concomitant.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.train_agent import build_env

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


class _ZeroProspectiveWrapper(gym.ObservationWrapper):
    """Zero out the prospective uncertainty slot in Agent C's observation.

    The original wrapper appends n_actions dims after the base observation.
    This wrapper writes zeros in those slots, leaving everything else
    untouched. Used only at evaluation time.
    """

    def __init__(self, env: gym.Env, base_obs_dim: int, n_actions: int):
        super().__init__(env)
        self.base_obs_dim = base_obs_dim
        self.n_actions = n_actions
        self.observation_space = env.observation_space

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = observation.copy()
        observation[self.base_obs_dim : self.base_obs_dim + self.n_actions] = 0.0
        return observation


def _rollout_episode(model, env) -> tuple[float, int]:
    obs, _ = env.reset()
    done = False; truncated = False; total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(int(action))
        total_reward += float(reward)
    base = cast(FoveatedClassificationEnv, env.unwrapped)
    acc = total_reward + base.cfg.patch_cost * len(base.committed_patches)
    return max(0.0, min(1.0, acc)), len(base.committed_patches)


def evaluate_with_and_without_feature(
    ckpt: Path,
    cfg_base: FoveatedEnvConfig,
    paths,
    labels,
    backbone,
    dynamics_ckpt: Path,
    corruption: str,
    severity: int,
    n_episodes: int,
    device: str,
) -> dict:
    cfg = FoveatedEnvConfig(**{**asdict(cfg_base), "corruption_type": corruption, "corruption_severity": severity})
    n_patches = cfg.grid_h * cfg.grid_w
    n_actions = n_patches + (1 if cfg.allow_stop_action else 0)
    goal_dim = len(cfg.goal_corruption_slots) + 1
    base_obs_dim = cfg.embed_dim + n_patches + goal_dim

    env_full = build_env("C", cfg, paths, labels, backbone, dynamics_ckpt, device)
    env_zero = _ZeroProspectiveWrapper(env_full, base_obs_dim=base_obs_dim, n_actions=n_actions)
    model = PPO.load(ckpt, device=device)

    full = []; zero = []
    rng = np.random.default_rng(cfg.seed)
    base_full = cast(FoveatedClassificationEnv, env_full.unwrapped)
    base_zero = cast(FoveatedClassificationEnv, env_zero.unwrapped)
    for _ in range(n_episodes):
        seed = int(rng.integers(2**31))
        base_full.rng = np.random.default_rng(seed)
        base_zero.rng = np.random.default_rng(seed)
        full.append(_rollout_episode(model, env_full)[0])
        zero.append(_rollout_episode(model, env_zero)[0])
    return {
        "corruption": corruption,
        "severity": severity,
        "n_episodes": n_episodes,
        "acc_with_feature": float(np.mean(full)),
        "acc_with_feature_std": float(np.std(full)),
        "acc_zeroed_feature": float(np.mean(zero)),
        "acc_zeroed_feature_std": float(np.std(zero)),
        "delta": float(np.mean(full) - np.mean(zero)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--agent-c-ckpt", type=Path, required=True)
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/eval/feature_ablation.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=128)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)
    cfg_base = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )

    out: list[dict] = []
    for corruption in HELD_OUT_CORRUPTIONS:
        r = evaluate_with_and_without_feature(
            ckpt=args.agent_c_ckpt,
            cfg_base=cfg_base,
            paths=paths,
            labels=labels,
            backbone=backbone,
            dynamics_ckpt=args.dynamics_ckpt,
            corruption=corruption,
            severity=args.severity,
            n_episodes=args.n_episodes,
            device=args.device,
        )
        out.append(r)
        print(
            f"corruption={corruption} acc_full={r['acc_with_feature']:.3f} "
            f"acc_zero={r['acc_zeroed_feature']:.3f} delta={r['delta']:+.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
