"""Patch-budget sweep K in {4, 8, 16}; direct empirical test of regret-bound P2.

docs/regret_bound.md predicts R(pi_C) ~ sqrt(K) for fixed ECE. The
proposal locks K = 8; this is a robustness ablation around that point.
Reported in the appendix as a P2 test.

For each K, retrain Agent A and Agent C with --env-max-patches set to
K and evaluate on the four held-out corruptions. The C-A regret gap is
computed per K; a sqrt-fit verifies P2 visually and quantitatively.

NOTE: this script does NOT retrain inside itself; training over three
Ks at three seeds is too long for a single experiment file. The script
expects checkpoints already produced by running

    for K in 4 8 16; do
      for agent in A C; do
        python -m foveated.experiments.train_agent --agent $agent \
          --seed 42 --manifest manifest.json \
          --env-max-patches $K --out-dir runs/seed_42_K$K
      done
    done

then this script aggregates and fits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import FoveatedEnvConfig
from foveated.experiments.evaluate import _rollout_episode
from foveated.experiments.train_agent import build_env

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


def _accuracy(env, model, n_episodes: int) -> tuple[float, float]:
    accs: list[float] = []
    for _ in range(n_episodes):
        acc, _ = _rollout_episode(model, env)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--laplace-ckpt", type=Path, default=None)
    p.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
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
    for K in args.ks:
        run_dir = args.runs_root / f"seed_{args.seed}_K{K}"
        for agent in ("A", "C"):
            ckpt = run_dir / f"agent_{agent}_seed{args.seed}_final.zip"
            if not ckpt.exists():
                print(f"[skip] missing {ckpt}")
                continue
            for corruption in HELD_OUT_CORRUPTIONS:
                cfg = FoveatedEnvConfig(
                    **{**asdict(cfg_base),
                       "corruption_type": corruption,
                       "corruption_severity": args.severity,
                       "max_patches": K}
                )
                env = build_env(
                    agent, cfg, paths, labels, backbone,
                    args.dynamics_ckpt if agent == "C" else None,
                    args.device,
                    laplace_ckpt=args.laplace_ckpt if agent == "C" else None,
                )
                model = PPO.load(ckpt, device=args.device)
                mean, std = _accuracy(env, model, args.n_episodes)
                rows.append({
                    "agent": agent, "K": K, "corruption": corruption,
                    "acc_mean": mean, "acc_std": std,
                })
                print(f"agent={agent} K={K} corr={corruption} acc={mean:.3f}")

    if not rows:
        print("no checkpoints found; nothing to sweep")
        return

    with (args.out_dir / "budget_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    by_K: dict[int, dict[str, float]] = {}
    for r in rows:
        K = int(r["K"])
        by_K.setdefault(K, {})[r["agent"]] = by_K[K].get(r["agent"], 0.0) + r["acc_mean"]
    for K in by_K:
        n = sum(1 for r in rows if r["K"] == K and r["agent"] == "A")
        if n:
            by_K[K]["A"] /= n
            by_K[K]["C"] /= n
    ks = sorted(by_K.keys())
    gaps = [by_K[k].get("C", 0.0) - by_K[k].get("A", 0.0) for k in ks]
    ax.plot(ks, gaps, marker="o", label="empirical C - A gap")
    if len(ks) >= 2 and abs(gaps[0]) > 1e-9:
        ref = [gaps[0] * np.sqrt(k / ks[0]) for k in ks]
        ax.plot(ks, ref, linestyle="--", label="sqrt(K) reference (anchored to first K)")
    ax.set_xlabel("patch budget K")
    ax.set_ylabel("top-1 gap (C - A), held-out")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "budget_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir}/budget_sweep.pdf")


if __name__ == "__main__":
    main()
