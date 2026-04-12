"""Statistical rigor for the four-agent ablation.

Bootstrap confidence intervals, paired permutation tests, and Cliff's
delta (non-parametric effect size). The proposal claims Agent C beats
Agent A in adaptation under distribution shift; we owe a falsifiable
statistical procedure that turns a point-estimate gap into a p-value
and an effect size that a reviewer can read off.

All routines are deterministic given a numpy seed. They operate on
per-episode arrays of metric values (top-1, regret, n_patches) and are
agnostic to which metric is in.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of x.

    Returns (mean, lo, hi) where (lo, hi) is the central (1-alpha) interval.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(x)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Two-sided paired permutation test on mean(x) - mean(y).

    x and y are paired observations of the same length (e.g., per-episode
    metric of two agents evaluated on the same images, same seed). Returns
    (observed_diff, two_sided_p_value).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    assert x.shape == y.shape, "paired test requires equal-length arrays"
    d = x - y
    obs = float(d.mean())
    n = len(d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return obs, p


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size on x vs y.

    delta in [-1, +1]. Conventions: |delta| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, else large (Romano et al. 2006).
    """
    nx, ny = len(x), len(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (nx * ny))


def report_pairwise(
    name_a: str,
    name_b: str,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    seed: int = 0,
) -> dict[str, float | str]:
    """Single-row comparison summary for one agent pair on one metric.

    Returns a dict suitable for csv or pandas DataFrame.
    """
    rng = np.random.default_rng(seed)
    mean_a, lo_a, hi_a = bootstrap_ci(a_vals, rng=rng)
    mean_b, lo_b, hi_b = bootstrap_ci(b_vals, rng=rng)
    diff, p = paired_permutation_test(a_vals, b_vals, rng=rng)
    delta = cliffs_delta(a_vals, b_vals)
    return {
        "pair": f"{name_a} vs {name_b}",
        "mean_a": mean_a, "lo_a": lo_a, "hi_a": hi_a,
        "mean_b": mean_b, "lo_b": lo_b, "hi_b": hi_b,
        "diff": diff,
        "p_value": p,
        "cliffs_delta": delta,
    }
