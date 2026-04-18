"""Fit the regret-bound predictions against the empirical results.

docs/regret_bound.md states two testable scaling predictions:
  P1. R(agent C) ~ sqrt(ECE)   (severity-driven calibration)
  P2. R(agent C) ~ sqrt(K)     (patch-budget-driven)

This script loads the relevant CSVs and fits a sqrt-scaling
relationship to each, reports R^2, slope, and intercept, and writes a
short markdown summary the report can drop in as-is.

Inputs (paths defaulted; missing inputs are tolerated, the corresponding
section of the report is omitted):
  runs/eval/aggregated_per_cell.csv  (acc per agent x corruption)
  runs/eval/calibration_ece.csv      (ECE per corruption)
  runs/eval/budget_sweep.csv         (acc per K per corruption)
  runs/eval/oracle_topk.csv          (oracle for regret denominator)

Output:
  runs/eval/bound_validation.md
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


HELD_OUT = ("snow", "frost", "fog", "pixelate")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _oracle_for(corruption: str, oracle_rows: list[dict]) -> float | None:
    for r in oracle_rows:
        if r["corruption"] == corruption:
            return float(r["oracle_topk_acc"])
    return None


def _ece_for(corruption: str, ece_rows: list[dict]) -> float | None:
    for r in ece_rows:
        if r["corruption"] == corruption:
            return float(r["ece"])
    return None


def fit_sqrt(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a * sqrt(x) + b by linear regression in sqrt(x). Return (a, b, R^2)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    sx = np.sqrt(np.maximum(x, 0.0))
    A = np.vstack([sx, np.ones_like(sx)]).T
    coef, _res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a * sx + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def validate_p1(per_cell: list[dict], ece_rows: list[dict], oracle_rows: list[dict]) -> dict:
    """P1: agent C regret should scale as sqrt(ECE) across corruption types."""
    xs: list[float] = []
    ys: list[float] = []
    for r in per_cell:
        if r["agent"] != "C" or r["corruption"] not in HELD_OUT:
            continue
        ece = _ece_for(r["corruption"], ece_rows)
        oracle = _oracle_for(r["corruption"], oracle_rows)
        acc = float(r["acc_mean"])
        if ece is None or oracle is None:
            continue
        regret = oracle - acc
        xs.append(ece); ys.append(regret)
    if not xs:
        return {"available": False}
    a, b, r2 = fit_sqrt(np.array(xs), np.array(ys))
    return {
        "available": True,
        "n_points": len(xs),
        "slope_sqrt_ece": a,
        "intercept": b,
        "r2": r2,
        "ece_range": (min(xs), max(xs)),
        "regret_range": (min(ys), max(ys)),
    }


def validate_p2(budget: list[dict], oracle_rows: list[dict]) -> dict:
    """P2: regret should scale as sqrt(K) for fixed corruption."""
    by_K_corr: dict[tuple[int, str], list[float]] = {}
    for r in budget:
        if r["agent"] != "C" or r["corruption"] not in HELD_OUT:
            continue
        K = int(r["K"])
        by_K_corr.setdefault((K, r["corruption"]), []).append(float(r["acc_mean"]))

    xs: list[float] = []
    ys: list[float] = []
    for (K, corr), accs in by_K_corr.items():
        oracle = _oracle_for(corr, oracle_rows)
        if oracle is None:
            continue
        regret = oracle - float(np.mean(accs))
        xs.append(K); ys.append(regret)
    if not xs:
        return {"available": False}
    a, b, r2 = fit_sqrt(np.array(xs), np.array(ys))
    return {
        "available": True,
        "n_points": len(xs),
        "slope_sqrt_K": a,
        "intercept": b,
        "r2": r2,
        "K_range": (min(xs), max(xs)),
        "regret_range": (min(ys), max(ys)),
    }


def write_report(p1: dict, p2: dict, out: Path) -> None:
    lines = ["# Regret-bound validation against empirical results", ""]

    lines.append("## P1: regret ~ sqrt(ECE)")
    if p1.get("available"):
        verdict = "supported" if p1["r2"] > 0.5 else "weakly supported" if p1["r2"] > 0.2 else "not supported"
        lines += [
            f"- n points: {p1['n_points']}",
            f"- slope (sqrt(ECE) -> regret): {p1['slope_sqrt_ece']:.4f}",
            f"- intercept: {p1['intercept']:.4f}",
            f"- R^2: {p1['r2']:.3f}",
            f"- verdict: {verdict}",
            "",
        ]
    else:
        lines += ["- insufficient data; calibration_ece.csv or oracle_topk.csv missing", ""]

    lines.append("## P2: regret ~ sqrt(K)")
    if p2.get("available"):
        verdict = "supported" if p2["r2"] > 0.5 else "weakly supported" if p2["r2"] > 0.2 else "not supported"
        lines += [
            f"- n points: {p2['n_points']}",
            f"- slope (sqrt(K) -> regret): {p2['slope_sqrt_K']:.4f}",
            f"- intercept: {p2['intercept']:.4f}",
            f"- R^2: {p2['r2']:.3f}",
            f"- verdict: {verdict}",
            "",
        ]
    else:
        lines += ["- insufficient data; run budget_sweep.py first", ""]

    out.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--out", type=Path, default=Path("runs/eval/bound_validation.md"))
    args = p.parse_args()

    per_cell = _load_csv(args.eval_dir / "aggregated_per_cell.csv")
    ece_rows = _load_csv(args.eval_dir / "calibration_ece.csv")
    budget = _load_csv(args.eval_dir / "budget_sweep.csv")
    oracle_rows = _load_csv(args.eval_dir / "oracle_topk.csv")

    p1 = validate_p1(per_cell, ece_rows, oracle_rows)
    p2 = validate_p2(budget, oracle_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(p1, p2, args.out)
    print(f"wrote {args.out}")
    print(f"P1: {p1}")
    print(f"P2: {p2}")
    _ = math  # reserved for future analytic-bound numeric prefactor


if __name__ == "__main__":
    main()
