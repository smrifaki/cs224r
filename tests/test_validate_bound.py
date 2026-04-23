"""Tests for the regret-bound prediction validator.

validate_bound.py fits sqrt-scaling to the empirical regret and
reports R^2 plus a verdict per prediction. The verdicts feed into
the report directly, so the procedure has to be right.

Three properties tested:

  1. fit_sqrt recovers the slope of a clean y = a*sqrt(x) curve to
     within numerical tolerance.
  2. fit_sqrt returns R^2 close to 1.0 on a clean curve and lower
     R^2 on a noisy curve.
  3. validate_p1 and validate_p2 return the right "not available"
     shape when input CSVs are missing or empty.
"""
from __future__ import annotations

import numpy as np

from foveated.experiments.validate_bound import (
    fit_sqrt,
    validate_p1,
    validate_p2,
)


def test_fit_sqrt_recovers_clean_slope():
    rng = np.random.default_rng(0)
    a_true = 0.7
    x = rng.uniform(0.05, 1.0, size=64)
    y = a_true * np.sqrt(x)
    a_hat, b_hat, r2 = fit_sqrt(x, y)
    assert abs(a_hat - a_true) < 1e-3
    assert abs(b_hat) < 1e-3
    assert r2 > 0.99


def test_fit_sqrt_lower_r2_under_noise():
    rng = np.random.default_rng(1)
    a_true = 0.5
    x = rng.uniform(0.1, 1.0, size=64)
    y = a_true * np.sqrt(x) + rng.normal(0, 0.2, size=64)
    _, _, r2 = fit_sqrt(x, y)
    assert r2 < 0.99
    assert r2 > 0.0


def test_fit_sqrt_handles_too_few_points():
    a, b, r2 = fit_sqrt(np.array([1.0]), np.array([1.0]))
    assert np.isnan(a) and np.isnan(b) and np.isnan(r2)


def test_validate_p1_missing_inputs():
    result = validate_p1([], [], [])
    assert result == {"available": False}


def test_validate_p2_missing_inputs():
    result = validate_p2([], [])
    assert result == {"available": False}


def test_validate_p1_happy_path():
    """Synthetic clean sqrt(ECE) relationship; should fit well."""
    rng = np.random.default_rng(2)
    held_out = ("snow", "frost", "fog", "pixelate")
    eces = rng.uniform(0.05, 0.5, size=4)
    oracle_acc = 0.8
    a_true = 0.6

    ece_rows = [{"corruption": c, "ece": str(eces[i]), "severity": "3", "n_samples": "256"}
                for i, c in enumerate(held_out)]
    oracle_rows = [{"corruption": c, "oracle_topk_acc": str(oracle_acc),
                    "severity": "3", "k": "8", "is_held_out": "True"} for c in held_out]
    per_cell = [{"agent": "C", "corruption": c, "n_episodes_total": "256",
                 "n_seeds": "3", "acc_mean": str(oracle_acc - a_true * np.sqrt(eces[i])),
                 "acc_ci_lo": "0", "acc_ci_hi": "1"} for i, c in enumerate(held_out)]
    result = validate_p1(per_cell, ece_rows, oracle_rows)
    assert result["available"]
    assert result["n_points"] == 4
    assert abs(result["slope_sqrt_ece"] - a_true) < 0.05
    assert result["r2"] > 0.99


def test_validate_p2_happy_path():
    """Synthetic clean sqrt(K) relationship; should fit well."""
    held_out = ("snow", "frost", "fog", "pixelate")
    Ks = (4, 8, 16)
    oracle_acc = 0.8
    a_true = 0.05

    oracle_rows = [{"corruption": c, "oracle_topk_acc": str(oracle_acc),
                    "severity": "3", "k": "8", "is_held_out": "True"} for c in held_out]
    budget = []
    for K in Ks:
        for c in held_out:
            budget.append({
                "agent": "C", "K": str(K), "corruption": c,
                "acc_mean": str(oracle_acc - a_true * np.sqrt(K)),
                "acc_std": "0.01",
            })
    result = validate_p2(budget, oracle_rows)
    assert result["available"]
    assert result["n_points"] == len(Ks) * len(held_out)
    assert abs(result["slope_sqrt_K"] - a_true) < 0.01
    assert result["r2"] > 0.99
