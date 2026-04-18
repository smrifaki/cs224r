"""Tests for the multi-seed aggregation helpers.

The aggregation step pools per-episode accuracies across seeds and
held-out corruption types and runs the confirmatory
tests. We pin three properties:

  1. write_per_cell sorts uniquely by (agent, corruption) and includes
     bootstrap CIs.
  2. write_held_out pools across the four held-out corruption types.
  3. write_pairwise emits the four comparisons with
     a verdict in {confirmed, suggestive, null}.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from foveated.experiments.aggregate_seeds import (
    HELD_OUT_CORRUPTIONS,
    PAIRS,
    write_held_out,
    write_pairwise,
    write_per_cell,
)


def _fake_seed_rows(agent_means: dict[str, float], n_episodes: int, rng) -> list[dict]:
    rows = []
    for agent, mu in agent_means.items():
        for corruption in HELD_OUT_CORRUPTIONS + ("clean",):
            accs = rng.binomial(1, mu, size=n_episodes).astype(float)
            rows.append({
                "agent": agent,
                "corruption": corruption,
                "per_episode_acc": accs.tolist(),
                "acc_mean": float(accs.mean()),
                "n_patches_mean": 4.0,
                "adaptation_curve": list(accs.cumsum() / np.arange(1, n_episodes + 1)),
            })
    return rows


def test_write_per_cell_one_row_per_pair(tmp_path):
    rng = np.random.default_rng(0)
    by_seed = {
        42: _fake_seed_rows({"A": 0.40, "C": 0.55}, 200, rng),
        1337: _fake_seed_rows({"A": 0.42, "C": 0.58}, 200, rng),
    }
    out = tmp_path / "per_cell.csv"
    write_per_cell(by_seed, out, seed_for_boot=0)
    text = out.read_text().strip()
    assert text, "per_cell.csv should not be empty"
    with out.open() as f:
        rows = list(csv.DictReader(f))
    pairs = {(r["agent"], r["corruption"]) for r in rows}
    expected_pairs = {(a, c) for a in ("A", "C") for c in HELD_OUT_CORRUPTIONS + ("clean",)}
    assert pairs == expected_pairs
    for r in rows:
        assert float(r["acc_ci_lo"]) <= float(r["acc_mean"]) <= float(r["acc_ci_hi"])


def test_write_held_out_pools_four_corruptions(tmp_path):
    rng = np.random.default_rng(1)
    by_seed = {
        42: _fake_seed_rows({"A": 0.30, "C": 0.60}, 100, rng),
        1337: _fake_seed_rows({"A": 0.32, "C": 0.62}, 100, rng),
        2024: _fake_seed_rows({"A": 0.31, "C": 0.61}, 100, rng),
    }
    out = tmp_path / "held_out.csv"
    write_held_out(by_seed, out, seed_for_boot=0)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    by_agent = {r["agent"]: r for r in rows}
    assert set(by_agent) == {"A", "C"}
    # Should pool over 3 seeds x 4 corruptions x 100 = 1200 episodes each.
    assert int(by_agent["A"]["n_episodes_total"]) == 3 * 4 * 100
    assert int(by_agent["C"]["n_episodes_total"]) == 3 * 4 * 100
    assert float(by_agent["C"]["acc_mean"]) > float(by_agent["A"]["acc_mean"])


def test_write_pairwise_emits_comparison_pairs(tmp_path):
    rng = np.random.default_rng(2)
    by_seed = {
        42: _fake_seed_rows({"A": 0.35, "B": 0.40, "C": 0.62, "D": 0.45}, 200, rng),
        1337: _fake_seed_rows({"A": 0.37, "B": 0.41, "C": 0.63, "D": 0.46}, 200, rng),
    }
    out = tmp_path / "pairwise.csv"
    write_pairwise(by_seed, out, seed_for_test=0)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    pair_names = {r["pair"] for r in rows}
    expected = {f"{a} vs {b}" for a, b in PAIRS}
    assert pair_names == expected
    for r in rows:
        assert r["verdict"] in {"confirmed", "suggestive", "null"}


def test_empty_input_writes_safe_marker(tmp_path):
    out_a = tmp_path / "per_cell.csv"
    out_b = tmp_path / "held_out.csv"
    out_c = tmp_path / "pairwise.csv"
    write_per_cell({}, out_a, seed_for_boot=0)
    write_held_out({}, out_b, seed_for_boot=0)
    write_pairwise({}, out_c, seed_for_test=0)
    assert "no" in out_a.read_text().lower()
    assert "no" in out_b.read_text().lower()
    assert "no" in out_c.read_text().lower()
