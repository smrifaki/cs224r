"""Small-multiples breakout: top-1 by agent for each corruption type.

The composite figure pools over the four held-out corruptions and
hides the per-corruption pattern. The per-corruption breakout shows
the same numbers but one panel per corruption, so a reviewer can see
where Agent C is winning and where it is losing.

Read from `aggregated_per_cell.csv` produced by `aggregate_seeds.py`.
Output: `per_corruption_breakout.pdf`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HELD_OUT = ("snow", "frost", "fog", "pixelate")
AGENT_ORDER = ("A", "B", "C", "D")
AGENT_COLOR = {"A": "#888888", "B": "#1f77b4", "C": "#d62728", "D": "#2ca02c"}


def _load(per_cell: Path) -> list[dict]:
    if not per_cell.exists():
        return []
    with per_cell.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=Path, default=Path("runs/eval/aggregated_per_cell.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/eval/per_corruption_breakout.pdf"))
    p.add_argument("--corruptions", nargs="+", default=list(HELD_OUT))
    args = p.parse_args()

    rows = _load(args.per_cell)
    if not rows:
        print(f"missing {args.per_cell}; run aggregate_seeds first")
        return

    n_panels = len(args.corruptions)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, corruption in zip(axes, args.corruptions, strict=False):
        means = []
        los = []
        his = []
        for agent in AGENT_ORDER:
            match = [r for r in rows if r["agent"] == agent and r["corruption"] == corruption]
            if match:
                m = float(match[0]["acc_mean"])
                lo = float(match[0]["acc_ci_lo"])
                hi = float(match[0]["acc_ci_hi"])
            else:
                m = lo = hi = float("nan")
            means.append(m); los.append(lo); his.append(hi)

        xs = np.arange(len(AGENT_ORDER))
        ax.bar(
            xs, means,
            yerr=[
                [m - lo for m, lo in zip(means, los, strict=False)],
                [hi - m for hi, m in zip(his, means, strict=False)],
            ],
            color=[AGENT_COLOR[a] for a in AGENT_ORDER],
            capsize=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(AGENT_ORDER)
        ax.set_title(corruption)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("top-1, pooled across seeds")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
