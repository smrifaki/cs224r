"""Composite results figure for the report.

Single figure with five panels in a 2x3 grid (one slot left blank):
    (a) Pareto: accuracy vs committed-patch budget, held-out corruptions.
    (b) Adaptation: running top-1 vs episode count, held-out corruptions.
    (c) Reliability: predicted variance vs realized squared residual,
        with the clean curve overlaid for reference.
    (d) Feature-ablation bars: Agent C with and without prospective
        uncertainty feature zeroed at inference.
    (e) Severity: top-1 vs corruption severity for all four agents,
        pooled over held-out corruptions.

The figure is built from the artifacts already produced by the
existing experiments (eval.npz, calibration_ece.csv,
feature_ablation.json, severity_sweep.csv) so this script does not
re-run training. It just stitches.

This is the figure the report front-loads. Reviewers should be able to
read off the four headline comparisons from this one figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")
AGENT_COLOR = {"A": "#888888", "B": "#1f77b4", "C": "#d62728", "D": "#2ca02c"}


def _load_eval(npz_path: Path) -> list[dict]:
    if not npz_path.exists():
        return []
    data = np.load(npz_path, allow_pickle=True)
    return list(data["results"])


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def panel_pareto(ax, results: list[dict]) -> None:
    for a in sorted({r["agent"] for r in results}):
        rs = [r for r in results if r["agent"] == a and r["corruption"] in HELD_OUT_CORRUPTIONS]
        if not rs:
            continue
        xs = [r["n_patches_mean"] for r in rs]
        ys = [r["acc_mean"] for r in rs]
        ax.scatter(xs, ys, label=f"{a}", s=30, color=AGENT_COLOR.get(a, "k"))
    ax.set_xlabel("avg committed patches")
    ax.set_ylabel("top-1 (held-out)")
    ax.set_title("(a) Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def panel_adaptation(ax, results: list[dict]) -> None:
    for a in sorted({r["agent"] for r in results}):
        rs = [r for r in results if r["agent"] == a and r["corruption"] in HELD_OUT_CORRUPTIONS]
        if not rs:
            continue
        max_n = max(len(r["adaptation_curve"]) for r in rs)
        curves = np.full((len(rs), max_n), np.nan)
        for i, r in enumerate(rs):
            curves[i, : len(r["adaptation_curve"])] = r["adaptation_curve"]
        m = np.nanmean(curves, axis=0)
        ax.plot(m, label=f"{a}", color=AGENT_COLOR.get(a, "k"))
    ax.set_xlabel("episodes at test time")
    ax.set_ylabel("running top-1")
    ax.set_title("(b) Adaptation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def panel_reliability(ax, ece_rows: list[dict]) -> None:
    if not ece_rows:
        ax.text(0.5, 0.5, "calibration_ece.csv not found", ha="center", va="center")
        ax.set_title("(c) Reliability")
        return
    corrs = [r["corruption"] for r in ece_rows]
    eces = [float(r["ece"]) for r in ece_rows]
    colors = ["#666666" if c == "clean" else "#d62728" for c in corrs]
    ax.bar(range(len(corrs)), eces, color=colors)
    ax.set_xticks(range(len(corrs)))
    ax.set_xticklabels(corrs, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("ECE")
    ax.set_title("(c) Dynamics-model calibration")
    ax.grid(True, axis="y", alpha=0.3)


def panel_feature_ablation(ax, fa_rows: list[dict]) -> None:
    if not fa_rows:
        ax.text(0.5, 0.5, "feature_ablation.json not found", ha="center", va="center")
        ax.set_title("(d) Feature ablation")
        return
    corrs = [r["corruption"] for r in fa_rows]
    full = [r["acc_with_feature"] for r in fa_rows]
    zero = [r["acc_zeroed_feature"] for r in fa_rows]
    x = np.arange(len(corrs))
    w = 0.4
    ax.bar(x - w / 2, full, w, label="C, feature on", color="#d62728")
    ax.bar(x + w / 2, zero, w, label="C, feature zeroed", color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(corrs, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("top-1")
    ax.set_title("(d) Mechanism")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def panel_severity(ax, sev_rows: list[dict]) -> None:
    if not sev_rows:
        ax.text(0.5, 0.5, "severity_sweep.csv not found", ha="center", va="center")
        ax.set_title("(e) Severity")
        return
    by_agent: dict[str, list[tuple[int, float]]] = {}
    for r in sev_rows:
        if r["corruption"] not in HELD_OUT_CORRUPTIONS:
            continue
        by_agent.setdefault(r["agent"], []).append((int(r["severity"]), float(r["acc_mean"])))
    for a, pts in sorted(by_agent.items()):
        bs = sorted(set(s for s, _ in pts))
        ys = [np.mean([acc for sev, acc in pts if sev == b]) for b in bs]
        ax.plot(bs, ys, marker="o", label=a, color=AGENT_COLOR.get(a, "k"))
    ax.set_xlabel("severity")
    ax.set_ylabel("top-1")
    ax.set_title("(e) Severity sweep (held-out)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--out", type=Path, default=Path("runs/eval/composite_figure.pdf"))
    args = p.parse_args()

    results = _load_eval(args.eval_dir / "eval.npz")
    ece_rows = _load_csv(args.eval_dir / "calibration_ece.csv")
    fa_rows = _load_json(args.eval_dir / "feature_ablation.json")
    sev_rows = _load_csv(args.eval_dir / "severity_sweep.csv")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panel_pareto(axes[0, 0], results)
    panel_adaptation(axes[0, 1], results)
    panel_reliability(axes[0, 2], ece_rows)
    panel_feature_ablation(axes[1, 0], fa_rows)
    panel_severity(axes[1, 1], sev_rows)
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
