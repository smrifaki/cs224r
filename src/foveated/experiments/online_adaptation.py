"""Online test-time updates of the dynamics-model variance head.

The adaptation curve from experiments/evaluate.py measures how the
running accuracy of a FROZEN policy evolves with more test-time
episodes. With the dynamics model also frozen, "adaptation" reduces to
whatever the policy's own state (PPO MLP feed-forward) can pick up,
which is limited.

This script adds a single axis: at test time, the variance head
of the dynamics model takes a small gradient step per episode using
the observed retrospective residuals on the patches the policy
committed. Everything else stays frozen: the PPO policy, the dynamics
model's mean head, the trunk, the backbone. Only logvar_head moves.
This is the cheapest sensible form of test-time training and is the
one that directly improves Agent C's decision-time feature under
distribution shift.

The comparison reported by this script is:
    pi_C with FROZEN dynamics  vs  pi_C with online-updated log_s.
On held-out corruptions where the dynamics model was poorly calibrated
(see calibration.py), the online update should narrow the calibration
gap and improve the adaptation curve.

This experiment is the empirical complement of the regret bound's
ECE-dependent prefactor: lowering ECE at test time should lower regret.
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
import torch
from stable_baselines3 import PPO

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.evaluate import _rollout_episode
from foveated.experiments.train_agent import build_env
from foveated.models.dynamics import ForwardDynamics

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


@torch.no_grad()
def _commitments_with_residuals(env, model) -> tuple[float, list[tuple[np.ndarray, int, np.ndarray]]]:
    """Roll out one episode under the deterministic policy and return
    (final_acc, [(z_t, a_t, z_next), ...]) for each committed patch.
    z_t is the pre-commit low-res embedding; z_next is the post-commit
    embedding. These triples are what the online updater consumes.
    """
    obs, _ = env.reset()
    base = cast(FoveatedClassificationEnv, env.unwrapped)
    triples: list[tuple[np.ndarray, int, np.ndarray]] = []
    prev_z = obs[: base.cfg.embed_dim].copy()
    done = False; truncated = False; total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        obs, reward, done, truncated, _ = env.step(a)
        cur_z = obs[: base.cfg.embed_dim].copy()
        if a < base.n_patches:
            triples.append((prev_z, a, cur_z))
        total_reward += float(reward)
        prev_z = cur_z
    acc = total_reward + base.cfg.patch_cost * len(base.committed_patches)
    return max(0.0, min(1.0, acc)), triples


def _online_update_logvar(
    dynamics: ForwardDynamics,
    triples: list[tuple[np.ndarray, int, np.ndarray]],
    lr: float,
    device: str,
) -> None:
    """One AdamW-like step on the logvar_head only.

    Freezes every other parameter. The gradient comes from the NLL of
    the observed (z_t, a_t, z_next) triples, computed using the model's
    current mean prediction. This is exactly the variance-only piece
    of the dynamics NLL, executed at test time on whatever the policy
    committed.
    """
    if not triples:
        return
    z = torch.from_numpy(np.stack([t[0] for t in triples])).float().to(device)
    a = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
    zn = torch.from_numpy(np.stack([t[2] for t in triples])).float().to(device)

    dynamics.train()
    for p in dynamics.parameters():
        p.requires_grad_(False)
    for p in dynamics.logvar_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(dynamics.logvar_head.parameters(), lr=lr)

    z_hat, log_s = dynamics(z, a)
    nll = ((zn - z_hat.detach()) ** 2 / log_s.exp() + log_s).mean()
    opt.zero_grad()
    nll.backward()
    opt.step()

    # Restore eval mode and re-freeze everything; subsequent callers
    # expect the dynamics module to behave like a frozen model.
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad_(False)


def run(args) -> dict[str, list[float]]:
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)

    out: dict[str, list[float]] = {"frozen": [], "online": []}
    cfg_base = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )

    for corruption in HELD_OUT_CORRUPTIONS:
        cfg = FoveatedEnvConfig(
            **{**asdict(cfg_base),
               "corruption_type": corruption,
               "corruption_severity": args.severity}
        )

        # Frozen-dynamics baseline.
        env_frozen = build_env("C", cfg, paths, labels, backbone, args.dynamics_ckpt, args.device)
        model = PPO.load(args.agent_c_ckpt, device=args.device)
        accs_frozen = []
        for _ in range(args.n_episodes):
            acc, _ = _rollout_episode(model, env_frozen)
            accs_frozen.append(acc)
        out["frozen"].append(float(np.mean(accs_frozen)))

        # Online-dynamics variant. Reload dynamics fresh so the online
        # updates do not leak across corruption types.
        ckpt = torch.load(args.dynamics_ckpt, map_location=args.device)
        dyn = ForwardDynamics(**ckpt["config"]).to(args.device).eval()
        dyn.load_state_dict(ckpt["state_dict"])
        for p in dyn.parameters():
            p.requires_grad_(False)

        env_online = build_env("C", cfg, paths, labels, backbone, args.dynamics_ckpt, args.device)
        # Hot-swap the wrapper's dynamics reference to our online-updated one.
        env_online.dyn = dyn  # type: ignore[attr-defined]
        accs_online = []
        for _ in range(args.n_episodes):
            acc, triples = _commitments_with_residuals(env_online, model)
            accs_online.append(acc)
            _online_update_logvar(dyn, triples, lr=args.online_lr, device=args.device)
        out["online"].append(float(np.mean(accs_online)))

        print(
            f"corruption={corruption}  frozen={out['frozen'][-1]:.3f}  "
            f"online={out['online'][-1]:.3f}  delta={out['online'][-1] - out['frozen'][-1]:+.3f}"
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--agent-c-ckpt", type=Path, required=True)
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=128)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--online-lr", type=float, default=1e-4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = run(args)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(HELD_OUT_CORRUPTIONS))
    w = 0.4
    ax.bar(x - w / 2, out["frozen"], w, label="frozen dynamics", color="#888888")
    ax.bar(x + w / 2, out["online"], w, label="online log_s update", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(HELD_OUT_CORRUPTIONS, rotation=30, ha="right")
    ax.set_ylabel("top-1 (held-out)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "online_adaptation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir}/online_adaptation.pdf")


if __name__ == "__main__":
    main()
