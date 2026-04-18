"""Find images where prospective uncertainty fires but committing fails.

This is the failure-mode probe for Agent C. The hypothesis is that
high prospective uncertainty agrees with high information value
(BALD-style reasoning). The failure mode is that the dynamics model
attributes high variance to high-frequency texture or to large
aleatoric noise, in which case the policy commits to patches that
look surprising but do not actually help the classifier.

The probe is: roll out the deterministic Agent C policy on each
held-out corruption. For each episode, record per-step
(prospective_uncertainty_of_chosen_patch, classifier_marginal_gain).
If high uncertainty correlates positively with marginal gain, the
feature is doing its job. If the correlation is near zero or
negative, the feature is firing on uninformative patches.

Output:
  runs/eval/adversarial_test.csv     per-episode rank correlation
  runs/eval/adversarial_test.pdf     scatter of uncertainty vs gain
"""
from __future__ import annotations

import argparse
import csv
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
from foveated.experiments.train_agent import build_env

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. No scipy dependency."""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _classifier_log_p_true(env, embed: np.ndarray, label: int) -> float:
    with torch.no_grad():
        logits = env.backbone.classify(
            torch.from_numpy(embed).float().to(env.device)
        )
        log_p = torch.log_softmax(logits, dim=-1)
        return float(log_p[label].item())


def _episode_probe(env, model) -> list[tuple[float, float]]:
    """Return per-step (uncertainty_of_chosen_patch, log_p_true_gain)."""
    obs, _ = env.reset()
    base = cast(FoveatedClassificationEnv, env.unwrapped)
    embed_dim = base.cfg.embed_dim
    n_patches = base.n_patches

    pairs: list[tuple[float, float]] = []
    prev_logp = _classifier_log_p_true(base, base.low_res_embed, base.current_label)
    done = False; truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        # The appended uncertainty slot for the chosen action.
        # Layout: [low_res_embed, mask, goal, prospective_u(0..n_actions-1)].
        u_slot_start = embed_dim + n_patches + base.goal_dim
        u_a = float(obs[u_slot_start + a]) if u_slot_start + a < len(obs) else 0.0
        obs, _, done, truncated, _ = env.step(a)
        if a < n_patches:
            new_logp = _classifier_log_p_true(base, _current_assembly(base), base.current_label)
            gain = new_logp - prev_logp
            pairs.append((u_a, gain))
            prev_logp = new_logp
    return pairs


def _current_assembly(base: FoveatedClassificationEnv) -> np.ndarray:
    if base.committed_patches:
        high = base.patch_embeds[base.committed_patches].mean(axis=0)
        return 0.5 * (base.low_res_embed + high)
    return base.low_res_embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--agent-c-ckpt", type=Path, required=True)
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--laplace-ckpt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=128)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)
    cfg_base = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )

    rows: list[dict] = []
    all_pairs: list[tuple[str, float, float]] = []
    for corruption in HELD_OUT_CORRUPTIONS:
        cfg = FoveatedEnvConfig(
            **{**asdict(cfg_base),
               "corruption_type": corruption,
               "corruption_severity": args.severity}
        )
        env = build_env(
            "C", cfg, paths, labels, backbone, args.dynamics_ckpt, args.device,
            laplace_ckpt=args.laplace_ckpt,
        )
        model = PPO.load(args.agent_c_ckpt, device=args.device)
        pairs: list[tuple[float, float]] = []
        for _ in range(args.n_episodes):
            pairs.extend(_episode_probe(env, model))
        if not pairs:
            continue
        us = np.array([p[0] for p in pairs])
        gs = np.array([p[1] for p in pairs])
        rho = _spearman(us, gs)
        rows.append({
            "corruption": corruption,
            "n_pairs": len(pairs),
            "spearman_rho": rho,
            "u_mean": float(us.mean()),
            "u_std": float(us.std()),
            "gain_mean": float(gs.mean()),
            "gain_std": float(gs.std()),
        })
        for u_, g_ in pairs:
            all_pairs.append((corruption, u_, g_))
        print(f"corruption={corruption}  rho={rho:+.3f}  n={len(pairs)}")

    if not rows:
        print("no probe pairs collected")
        return

    with (args.out_dir / "adversarial_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, len(HELD_OUT_CORRUPTIONS), figsize=(3 * len(HELD_OUT_CORRUPTIONS), 3), sharey=True)
    for ax, corruption in zip(axes, HELD_OUT_CORRUPTIONS, strict=False):
        sub = [(u, g) for c, u, g in all_pairs if c == corruption]
        if not sub:
            continue
        us = np.array([u for u, _ in sub])
        gs = np.array([g for _, g in sub])
        ax.scatter(us, gs, s=4, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        rho = next((r["spearman_rho"] for r in rows if r["corruption"] == corruption), float("nan"))
        ax.set_title(f"{corruption}  rho={rho:+.2f}")
        ax.set_xlabel("prospective u of chosen patch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("log p(true) gain at commit")
    fig.tight_layout()
    fig.savefig(args.out_dir / "adversarial_test.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir}/adversarial_test.pdf")


if __name__ == "__main__":
    main()
