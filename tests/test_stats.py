"""Sanity tests for the stats module.

This module is exercised by
these three primitives. If any of them is buggy the entire confirmatory
analysis is invalid. Tests pin behavior under known distributions where
the right answer is analytically known.
"""
from __future__ import annotations

import numpy as np
import pytest

from foveated.algos.stats import (
    bootstrap_ci,
    cliffs_delta,
    paired_permutation_test,
    report_pairwise,
)


def test_bootstrap_ci_covers_true_mean_under_normal():
    """Across 200 simulated samples from N(mu, 1), the 95 percent CI should
    contain the true mean in at least 90 percent of cases. Loose threshold
    to keep the test fast and stable; the actual coverage should be ~95%.
    """
    rng = np.random.default_rng(0)
    mu_true = 0.7
    covered = 0
    trials = 200
    for _ in range(trials):
        x = rng.normal(mu_true, 1.0, size=200)
        boot_rng = np.random.default_rng(int(rng.integers(2**31)))
        _, lo, hi = bootstrap_ci(x, n_boot=2000, rng=boot_rng)
        if lo <= mu_true <= hi:
            covered += 1
    assert covered / trials >= 0.90, (
        f"bootstrap CI coverage too low: {covered}/{trials}"
    )


def test_bootstrap_ci_shrinks_with_n():
    """CI half-width should shrink roughly as 1/sqrt(n)."""
    rng = np.random.default_rng(1)
    widths = []
    for n in (50, 500):
        x = rng.normal(0, 1, size=n)
        _, lo, hi = bootstrap_ci(x, n_boot=2000, rng=np.random.default_rng(2))
        widths.append(hi - lo)
    ratio = widths[0] / widths[1]
    assert 2.0 < ratio < 5.0, (
        f"CI width should shrink ~sqrt(10) ~ 3.16x with 10x more data; got {ratio:.2f}x"
    )


def test_paired_permutation_test_calibrated_under_null():
    """When x and y are i.i.d. from the same distribution, the permutation
    p-value should be approximately uniform on [0, 1]. We test the weaker
    property that the rejection rate at alpha=0.05 is approximately 0.05.
    """
    rng = np.random.default_rng(2)
    rejections = 0
    trials = 200
    for _ in range(trials):
        n = 50
        x = rng.normal(0, 1, size=n)
        y = rng.normal(0, 1, size=n)
        _, p = paired_permutation_test(
            x, y, n_perm=1000, rng=np.random.default_rng(int(rng.integers(2**31)))
        )
        if p < 0.05:
            rejections += 1
    rate = rejections / trials
    # Wide tolerance because n_perm is moderate and trials are 200.
    assert 0.01 < rate < 0.12, (
        f"Type I error rate {rate:.3f} is too far from nominal 0.05"
    )


def test_paired_permutation_test_detects_real_effect():
    """With a clean +0.5 shift in 100 paired samples, the p-value should be
    well below 0.05 essentially every time."""
    rng = np.random.default_rng(3)
    p_vals = []
    for _ in range(10):
        n = 100
        z = rng.normal(0, 1, size=n)
        x = z + 0.5
        y = z
        _, p = paired_permutation_test(
            x, y, n_perm=2000, rng=np.random.default_rng(int(rng.integers(2**31)))
        )
        p_vals.append(p)
    assert max(p_vals) < 0.05, f"failed to detect 0.5 sigma effect: p={p_vals}"


def test_cliffs_delta_known_distributions():
    """Cliff's delta on known distributions: delta = 0 for identical,
    delta = 1 for fully separated.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, size=200)
    y = rng.normal(0, 1, size=200)
    assert abs(cliffs_delta(x, y)) < 0.15, "delta should be ~0 for same-distribution"

    x2 = np.full(50, 1.0)
    y2 = np.full(50, 0.0)
    assert cliffs_delta(x2, y2) == pytest.approx(1.0), "delta should be 1 for fully separated"
    assert cliffs_delta(y2, x2) == pytest.approx(-1.0), "delta should flip sign"


def test_report_pairwise_returns_expected_keys():
    rng = np.random.default_rng(5)
    a = rng.normal(0.6, 1, size=100)
    b = rng.normal(0.0, 1, size=100)
    out = report_pairwise("C", "A", a, b, seed=0)
    expected = {
        "pair", "mean_a", "lo_a", "hi_a", "mean_b", "lo_b", "hi_b",
        "diff", "p_value", "cliffs_delta",
    }
    assert expected.issubset(out.keys())
    assert out["pair"] == "C vs A"
    assert isinstance(out["p_value"], float)
