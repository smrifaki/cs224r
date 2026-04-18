"""Held-out accuracy as a function of ImageNet-C severity, all four agents.

Prediction P1 of the regret bound: regret scales as sqrt(ECE), and ECE
grows roughly monotonically with corruption severity. So the held-out
regret of Agent C should grow sublinearly with severity, with the Agent
C - Agent A gap potentially widening (more uncertainty signal to extract)
or narrowing (calibration breakdown) as severity rises. Both directions
are interesting; the experiment is designed to be agnostic and report.

For each held-out corruption type and each severity in {1, 2, 3, 4, 5},
roll out N episodes per agent and record (accuracy, n_patches). Plot
accuracy vs severity, one line per agent, four facets per corruption.

This is the test that distinguishes "C helps under any shift" from "C
helps only under mild shift" from "C breaks under high shift". The
proposal predicts the first; the BALD calibration argument predicts the
last; the truth is empirical.
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
SEVERITIES = (1, 2, 3, 4, 5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--ckpt-dir", type=Path, default=Path("runs"))
    p.add_argument("--dynamics-ckpt", type=Path, default=Path("checkpoints/dynamics_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
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
    for agent in ("A", "B", "C", "D"):
        ckpt = args.ckpt_dir / f"agent_{agent}_seed{args.seed}_final.zip"
        if not ckpt.exists():
            print(f"[skip] missing checkpoint {ckpt}")
            continue
        for corruption in HELD_OUT_CORRUPTIONS:
            for sev in SEVERITIES:
                cfg = FoveatedEnvConfig(
                    **{**asdict(cfg_base),
                       "corruption_type": corruption,
                       "corruption_severity": sev}
                )
                env = build_env(
                    agent, cfg, paths, labels, backbone,
                    args.dynamics_ckpt if agent in ("B", "C") else None,
                    args.device,
                )
                model = PPO.load(ckpt, device=args.device)
                accs = []
                for _ in range(args.n_episodes):
                    acc, _ = _rollout_episode(model, env)
                    accs.append(acc)
                rows.append({
                    "agent": agent,
                    "corruption": corruption,
                    "severity": sev,
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                })
                print(f"agent={agent} corr={corruption} sev={sev} acc={np.mean(accs):.3f}")

    with (args.out_dir / "severity_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, len(HELD_OUT_CORRUPTIONS), figsize=(3 * len(HELD_OUT_CORRUPTIONS), 3), sharey=True)
    for ax, corruption in zip(axes, HELD_OUT_CORRUPTIONS):
        for agent in ("A", "B", "C", "D"):
            sub = [r for r in rows if r["agent"] == agent and r["corruption"] == corruption]
            if not sub:
                continue
            xs = [r["severity"] for r in sub]
            ys = [r["acc_mean"] for r in sub]
            errs = [r["acc_std"] / np.sqrt(args.n_episodes) for r in sub]
            ax.errorbar(xs, ys, yerr=errs, label=f"Agent {agent}", marker="o", capsize=2)
        ax.set_title(corruption)
        ax.set_xlabel("Severity")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Top-1 accuracy")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "severity_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir}/severity_sweep.pdf")


if __name__ == "__main__":
    main()
