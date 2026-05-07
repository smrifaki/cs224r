"""Pool per-seed evaluation results and report mean + 95% CI per cell.

evaluate.py writes one runs/eval/eval.npz per seed; this script reads all
seeded eval files, pools per-episode top-1 across the locked seeds (42,
1337, 2024), and emits the aggregated table the report consumes.

Confirmatory comparisons pool first across the
four held-out corruption types, then across seeds. This script implements
that exact pooling order so the report's headline number is the locked
one.

Outputs:
    runs/eval/aggregated_per_cell.csv      one row per (agent, corruption)
    runs/eval/aggregated_held_out.csv      pooled over held-out corruptions
    runs/eval/aggregated_pairwise.csv      H1..H4 comparisons
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from foveated.algos.stats import bootstrap_ci, cliffs_delta, paired_permutation_test

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")
LOCKED_SEEDS = (42, 1337, 2024)
PAIRS = [("C", "A"), ("C", "D"), ("B", "A"), ("C", "B")]


def _load_seed(eval_dir: Path, seed: int) -> list[dict] | None:
    path = eval_dir / f"seed_{seed}" / "eval.npz"
    if not path.exists():
        path = eval_dir / "eval.npz" if seed == LOCKED_SEEDS[0] else None  # type: ignore[assignment]
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return list(data["results"])


def _per_episode(rows: list[dict], agent: str, corruption: str) -> np.ndarray:
    for r in rows:
        if r["agent"] == agent and r["corruption"] == corruption:
            return np.asarray(r["per_episode_acc"], dtype=np.float64)
    return np.array([], dtype=np.float64)


def write_per_cell(
    by_seed: dict[int, list[dict]],
    out: Path,
    seed_for_boot: int = 0,
) -> None:
    rng = np.random.default_rng(seed_for_boot)
    rows: list[dict] = []
    agents = sorted({r["agent"] for s in by_seed.values() for r in s})
    corruptions = sorted({r["corruption"] for s in by_seed.values() for r in s})
    for agent in agents:
        for corruption in corruptions:
            pooled = np.concatenate(
                [_per_episode(s, agent, corruption) for s in by_seed.values()]
            ) if by_seed else np.array([], dtype=np.float64)
            if len(pooled) == 0:
                continue
            mean, lo, hi = bootstrap_ci(pooled, rng=rng)
            rows.append({
                "agent": agent,
                "corruption": corruption,
                "n_episodes_total": len(pooled),
                "n_seeds": int(sum(1 for s in by_seed.values() if any(r["agent"] == agent and r["corruption"] == corruption for r in s))),
                "acc_mean": mean,
                "acc_ci_lo": lo,
                "acc_ci_hi": hi,
            })
    with out.open("w", newline="") as f:
        if not rows:
            f.write("# no aggregated cells; check that seed eval dirs exist\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_held_out(
    by_seed: dict[int, list[dict]],
    out: Path,
    seed_for_boot: int = 0,
) -> None:
    rng = np.random.default_rng(seed_for_boot)
    agents = sorted({r["agent"] for s in by_seed.values() for r in s})
    rows: list[dict] = []
    for agent in agents:
        pooled = np.concatenate([
            _per_episode(s, agent, c)
            for s in by_seed.values()
            for c in HELD_OUT_CORRUPTIONS
        ]) if by_seed else np.array([], dtype=np.float64)
        if len(pooled) == 0:
            continue
        mean, lo, hi = bootstrap_ci(pooled, rng=rng)
        rows.append({
            "agent": agent,
            "n_episodes_total": len(pooled),
            "n_seeds": len(by_seed),
            "acc_mean": mean,
            "acc_ci_lo": lo,
            "acc_ci_hi": hi,
        })
    with out.open("w", newline="") as f:
        if not rows:
            f.write("# no held-out aggregate; check seed eval dirs\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_pairwise(
    by_seed: dict[int, list[dict]],
    out: Path,
    seed_for_test: int = 0,
) -> None:
    rng = np.random.default_rng(seed_for_test)

    def pool(agent: str) -> np.ndarray:
        return np.concatenate([
            _per_episode(s, agent, c)
            for s in by_seed.values()
            for c in HELD_OUT_CORRUPTIONS
        ]) if by_seed else np.array([], dtype=np.float64)

    rows: list[dict] = []
    for a, b in PAIRS:
        x = pool(a)
        y = pool(b)
        if len(x) == 0 or len(y) == 0:
            continue
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        mean_a, lo_a, hi_a = bootstrap_ci(x, rng=rng)
        mean_b, lo_b, hi_b = bootstrap_ci(y, rng=rng)
        diff, p = paired_permutation_test(x, y, rng=rng)
        delta = cliffs_delta(x, y)
        # Confirmation rule (p < 0.05 AND |delta| > 0.147 with predicted direction).
        confirmed = (p < 0.05) and (abs(delta) > 0.147) and (diff > 0)
        rows.append({
            "pair": f"{a} vs {b}",
            "n_episodes_paired": int(n),
            "mean_a": mean_a, "ci_a_lo": lo_a, "ci_a_hi": hi_a,
            "mean_b": mean_b, "ci_b_lo": lo_b, "ci_b_hi": hi_b,
            "diff": diff,
            "p_value": p,
            "cliffs_delta": delta,
            "verdict": (
                "confirmed" if confirmed else
                ("suggestive" if (p < 0.05) and (diff > 0) else "null")
            ),
        })
    with out.open("w", newline="") as f:
        if not rows:
            f.write("# no pairwise aggregate; check seed eval dirs\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=Path("runs/eval"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--seed-for-tests", type=int, default=0,
                   help="seed for bootstrap and permutation RNGs; not a training seed")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_seed: dict[int, list[dict]] = {}
    for s in LOCKED_SEEDS:
        rows = _load_seed(args.eval_root, s)
        if rows is not None:
            by_seed[s] = rows
    if not by_seed:
        print("no seeded eval results found; nothing to aggregate")
        return

    print(f"aggregating over seeds: {sorted(by_seed.keys())}")
    write_per_cell(by_seed, args.out_dir / "aggregated_per_cell.csv",
                   seed_for_boot=args.seed_for_tests)
    write_held_out(by_seed, args.out_dir / "aggregated_held_out.csv",
                   seed_for_boot=args.seed_for_tests)
    write_pairwise(by_seed, args.out_dir / "aggregated_pairwise.csv",
                   seed_for_test=args.seed_for_tests)
    print(f"wrote {args.out_dir}/aggregated_*.csv")


if __name__ == "__main__":
    main()
