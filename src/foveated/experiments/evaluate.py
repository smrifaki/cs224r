"""Evaluation sweep producing the three deliverables for Phase 5.

For each (agent, corruption, severity), roll out N episodes and record per-
episode (top-1, n_patches_committed, regret). Output:

    1. fig1_pareto.pdf , top-1 accuracy vs average committed-patch budget
       on held-out corruption types. One line per agent.
    2. fig2_adaptation.pdf , top-1 accuracy as a function of corruption-
       samples seen at evaluation time, policy frozen. Measures in-context
       adaptation only via the prediction-error feature / entropy feature
       / intrinsic reward.
    3. regret.csv , per (agent, corruption) regret = oracle - agent.

Saves runs/eval.npz with everything, runs/regret.csv, and the two pdfs.
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

from foveated.algos.stats import report_pairwise
from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import (
    CORRUPTION_NAMES,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)
from foveated.experiments.train_agent import build_env

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")
TRAIN_CORRUPTIONS = tuple(c for c in CORRUPTION_NAMES if c not in HELD_OUT_CORRUPTIONS)


def _rollout_episode(model, env) -> tuple[float, int]:
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(int(action))
        total_reward += float(reward)
    base = cast(FoveatedClassificationEnv, env.unwrapped)
    acc = total_reward + base.cfg.patch_cost * len(base.committed_patches)
    return max(0.0, min(1.0, acc)), len(base.committed_patches)


def _oracle_accuracy(env, n_episodes: int) -> float:
    accs = []
    base = cast(FoveatedClassificationEnv, env.unwrapped)
    for _ in range(n_episodes):
        env.reset()
        snap = base.current_state()
        full = 0.5 * (snap["z_t"] + snap["patch_embeds"].mean(axis=0))
        logits = base.backbone.classify(
            torch.from_numpy(full).float().to(base.device)
        )
        accs.append(float(int(logits.argmax().item()) == snap["label"]))
    return float(np.mean(accs))


def evaluate_one(
    agent: str,
    ckpt: Path,
    cfg_base: FoveatedEnvConfig,
    paths,
    labels,
    backbone,
    dynamics_ckpt: Path | None,
    corruption: str,
    severity: int,
    n_episodes: int,
    device: str,
) -> dict:
    cfg = FoveatedEnvConfig(**{**asdict(cfg_base), "corruption_type": corruption, "corruption_severity": severity})
    env = build_env(agent, cfg, paths, labels, backbone, dynamics_ckpt, device)
    model = PPO.load(ckpt, env=env, device=device)

    accs, counts = [], []
    running_mean = []
    for _ep in range(n_episodes):
        acc, n = _rollout_episode(model, env)
        accs.append(acc)
        counts.append(n)
        running_mean.append(float(np.mean(accs)))

    oracle = _oracle_accuracy(env, n_episodes=min(64, n_episodes))
    return {
        "agent": agent,
        "corruption": corruption,
        "severity": severity,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "n_patches_mean": float(np.mean(counts)),
        "n_patches_std": float(np.std(counts)),
        "oracle": oracle,
        "regret": oracle - float(np.mean(accs)),
        "adaptation_curve": running_mean,
        "per_episode_acc": [float(a) for a in accs],
    }


def make_pareto_figure(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    agents = sorted({r["agent"] for r in results})
    for a in agents:
        rs = [r for r in results if r["agent"] == a and r["corruption"] in HELD_OUT_CORRUPTIONS]
        x = [r["n_patches_mean"] for r in rs]
        y = [r["acc_mean"] for r in rs]
        ax.scatter(x, y, label=f"Agent {a}", s=40)
    ax.set_xlabel("Average committed patches per episode")
    ax.set_ylabel("Top-1 accuracy (held-out corruptions)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def make_adaptation_figure(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    agents = sorted({r["agent"] for r in results})
    for a in agents:
        rs = [r for r in results if r["agent"] == a and r["corruption"] in HELD_OUT_CORRUPTIONS]
        if not rs:
            continue
        max_n = max(len(r["adaptation_curve"]) for r in rs)
        curves = np.full((len(rs), max_n), np.nan)
        for i, r in enumerate(rs):
            curves[i, : len(r["adaptation_curve"])] = r["adaptation_curve"]
        m = np.nanmean(curves, axis=0)
        ax.plot(m, label=f"Agent {a}")
    ax.set_xlabel("Episodes seen at test time")
    ax.set_ylabel("Running top-1 accuracy (held-out)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_regret_csv(results: list[dict], out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "corruption", "severity", "acc_mean", "oracle", "regret"])
        for r in results:
            w.writerow([
                r["agent"], r["corruption"], r["severity"],
                f"{r['acc_mean']:.4f}", f"{r['oracle']:.4f}", f"{r['regret']:.4f}",
            ])


def write_pairwise_stats(results: list[dict], out: Path, seed: int = 0) -> None:
    """Bootstrap CI + paired permutation test + Cliff's delta for each
    pairwise comparison, pooled over held-out corruptions.

    Pairs compared:
        C vs A: prospective uncertainty as a feature beats no feature
        C vs D: dynamics-model variance beats classifier entropy
        B vs A: pathak intrinsic reward beats no intrinsic
        C vs B: feature framing beats reward framing of the same signal
    """
    by_agent: dict[str, np.ndarray] = {}
    for a in ("A", "B", "C", "D"):
        accs = np.concatenate([
            np.asarray(r["per_episode_acc"], dtype=np.float64)
            for r in results
            if r["agent"] == a and r["corruption"] in HELD_OUT_CORRUPTIONS
        ]) if any(r["agent"] == a for r in results) else np.array([], dtype=np.float64)
        if len(accs):
            by_agent[a] = accs

    pairs = [("C", "A"), ("C", "D"), ("B", "A"), ("C", "B")]
    rows = []
    for a, b in pairs:
        if a in by_agent and b in by_agent and len(by_agent[a]) == len(by_agent[b]):
            rows.append(report_pairwise(a, b, by_agent[a], by_agent[b], seed=seed))

    with out.open("w", newline="") as f:
        if not rows:
            f.write("# no pairwise stats available; insufficient overlap of agent results\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--ckpt-dir", type=Path, default=Path("runs"))
    p.add_argument("--dynamics-ckpt", type=Path, default=Path("checkpoints/dynamics_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=256)
    p.add_argument("--severity", type=int, default=3)
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

    results: list[dict] = []
    for agent in ("A", "B", "C", "D"):
        ckpt = args.ckpt_dir / f"agent_{agent}_seed{args.seed}_final.zip"
        if not ckpt.exists():
            print(f"[skip] missing checkpoint {ckpt}")
            continue
        for corruption in TRAIN_CORRUPTIONS + HELD_OUT_CORRUPTIONS:
            r = evaluate_one(
                agent=agent,
                ckpt=ckpt,
                cfg_base=cfg_base,
                paths=paths,
                labels=labels,
                backbone=backbone,
                dynamics_ckpt=args.dynamics_ckpt if agent in ("B", "C") else None,
                corruption=corruption,
                severity=args.severity,
                n_episodes=args.n_episodes,
                device=args.device,
            )
            results.append(r)
            print(
                f"agent={agent} corruption={corruption} "
                f"acc={r['acc_mean']:.3f}+-{r['acc_std']:.3f} "
                f"patches={r['n_patches_mean']:.1f} "
                f"regret={r['regret']:.3f}"
            )

    np.savez(args.out_dir / "eval.npz", results=np.asarray(results, dtype=object))
    write_regret_csv(results, args.out_dir / "regret.csv")
    write_pairwise_stats(results, args.out_dir / "pairwise_stats.csv", seed=args.seed)
    make_pareto_figure(results, args.out_dir / "fig1_pareto.pdf")
    make_adaptation_figure(results, args.out_dir / "fig2_adaptation.pdf")
    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
