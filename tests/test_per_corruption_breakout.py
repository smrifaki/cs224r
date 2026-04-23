"""Tests for the per-corruption breakout figure script.

The script reads aggregated_per_cell.csv and emits one PDF. Tests
verify that it tolerates missing input gracefully and produces a
non-empty PDF when given a small in-memory CSV.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _write_per_cell(path: Path) -> None:
    rows = []
    for agent in ("A", "B", "C", "D"):
        for corruption in ("snow", "frost", "fog", "pixelate"):
            rows.append({
                "agent": agent,
                "corruption": corruption,
                "n_episodes_total": "300",
                "n_seeds": "3",
                "acc_mean": "0.5",
                "acc_ci_lo": "0.45",
                "acc_ci_hi": "0.55",
            })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_breakout_runs_on_minimal_csv(tmp_path):
    csv_path = tmp_path / "per_cell.csv"
    out_path = tmp_path / "breakout.pdf"
    _write_per_cell(csv_path)

    result = subprocess.run(
        [
            sys.executable, "-m", "foveated.experiments.per_corruption_breakout",
            "--per-cell", str(csv_path),
            "--out", str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"breakout failed: {result.stderr}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_breakout_tolerates_missing_input(tmp_path):
    missing = tmp_path / "missing.csv"
    out_path = tmp_path / "breakout.pdf"
    result = subprocess.run(
        [
            sys.executable, "-m", "foveated.experiments.per_corruption_breakout",
            "--per-cell", str(missing),
            "--out", str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # Script returns 0 even when input is missing; it just prints a message.
    assert result.returncode == 0
    assert not out_path.exists()
