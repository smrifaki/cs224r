"""Tests for the calibration helper functions.

Pin the reliability-curve binning and the ECE computation. The
calibration diagnostic is load-bearing for the regret-bound argument,
so we owe these tests.
"""
from __future__ import annotations

import numpy as np

from foveated.experiments.calibration import (
    expected_calibration_error,
    reliability_curve,
)


def test_reliability_curve_shape():
    rng = np.random.default_rng(0)
    sq_res = rng.uniform(0, 1, size=500)
    var = rng.uniform(0, 1, size=500)
    bin_var, bin_sq = reliability_curve(sq_res, var, n_bins=10)
    assert bin_var.shape == (10,)
    assert bin_sq.shape == (10,)


def test_ece_zero_on_perfect_calibration():
    """If predicted variance equals realized squared residual exactly,
    the reliability curve is the identity and ECE is zero.
    """
    rng = np.random.default_rng(1)
    var = rng.uniform(0.01, 1, size=2000)
    sq_res = var.copy()
    ece = expected_calibration_error(sq_res, var, n_bins=10)
    assert ece < 1e-6, f"ECE on perfect calibration should be ~0, got {ece}"


def test_ece_positive_on_systematic_miscalibration():
    """If predicted variance is shrunk by half relative to realized,
    ECE should be substantially positive (the model is overconfident).
    """
    rng = np.random.default_rng(2)
    var = rng.uniform(0.01, 1, size=2000)
    sq_res = 2 * var
    ece = expected_calibration_error(sq_res, var, n_bins=10)
    assert ece > 0.1, f"ECE on 2x overconfident should be substantial, got {ece}"


def test_reliability_curve_monotone_under_calibrated_data():
    """When variance and squared residual are strongly correlated, the
    binned reliability curve should be monotone non-decreasing.
    """
    rng = np.random.default_rng(3)
    n = 4000
    base = rng.uniform(0.01, 1, size=n)
    noise = rng.normal(0, 0.05, size=n)
    var = base
    sq_res = base + noise
    _, bin_sq = reliability_curve(sq_res, var, n_bins=10)
    mask = np.isfinite(bin_sq)
    increases = (np.diff(bin_sq[mask]) >= -1e-3).sum()
    assert increases >= mask.sum() - 2, "binned curve should be roughly monotone"


def test_ece_handles_all_nan_bins():
    """All identical predictions should produce a degenerate reliability
    curve and still return a finite ECE (possibly NaN; check finite or NaN).
    """
    var = np.ones(100) * 0.5
    sq_res = np.ones(100) * 0.5
    ece = expected_calibration_error(sq_res, var, n_bins=10)
    assert np.isnan(ece) or ece >= 0
