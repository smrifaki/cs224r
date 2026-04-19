"""Subprocess test for composite_figure.

The script reads up to four input files and emits one PDF. We
verify it runs with all four inputs present and degrades gracefully
when any subset is missing.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_eval_npz(path: Path) -> None:
    results = []
    for agent in ("A", "B", "C", "D"):
        for corruption in ("snow", "frost", "fog", "pixelate", "gaussian_noise"):
            results.append({
                "agent": agent,
                "corruption": corruption,
                "severity": 3,
                "acc_mean": 0.45 + 0.05 * (agent == "C"),
                "acc_std": 0.05,
                "n_patches_mean": 5.0,
                "n_patches_std": 1.0,
                "oracle": 0.7,
                "regret": 0.25 - 0.05 * (agent == "C"),
                "adaptation_curve": list(np.linspace(0.4, 0.6, 64)),
                "per_episode_acc": [0.5] * 32,
            })
    np.savez(path, results=np.asarray(results, dtype=object))


def _write_ece_csv(path: Path) -> None:
    rows = [
        {"corruption": "clean", "severity": "0", "ece": "0.02", "n_samples": "1000"},
        {"corruption": "snow", "severity": "3", "ece": "0.18", "n_samples": "1000"},
        {"corruption": "frost", "severity": "3", "ece": "0.22", "n_samples": "1000"},
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_feature_ablation_json(path: Path) -> None:
    payload = [
        {"corruption": "snow", "severity": 3, "n_episodes": 64,
         "acc_with_feature": 0.50, "acc_with_feature_std": 0.05,
         "acc_zeroed_feature": 0.45, "acc_zeroed_feature_std": 0.05, "delta": 0.05},
    ]
    path.write_text(json.dumps(payload))


def _write_severity_csv(path: Path) -> None:
    rows = []
    for agent in ("A", "C"):
        for corruption in ("snow", "frost"):
            for sev in (1, 3, 5):
                rows.append({
                    "agent": agent, "corruption": corruption, "severity": sev,
                    "acc_mean": 0.5 - 0.05 * (sev - 1),
                    "acc_std": 0.04,
                })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_composite_figure_runs_with_all_inputs(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_eval_npz(eval_dir / "eval.npz")
    _write_ece_csv(eval_dir / "calibration_ece.csv")
    _write_feature_ablation_json(eval_dir / "feature_ablation.json")
    _write_severity_csv(eval_dir / "severity_sweep.csv")

    out = eval_dir / "composite.pdf"
    res = subprocess.run(
        [sys.executable, "-m", "foveated.experiments.composite_figure",
         "--eval-dir", str(eval_dir), "--out", str(out)],
        check=False, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"composite failed: {res.stderr}"
    assert out.exists() and out.stat().st_size > 0


def test_composite_figure_handles_missing_inputs(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = eval_dir / "composite.pdf"
    res = subprocess.run(
        [sys.executable, "-m", "foveated.experiments.composite_figure",
         "--eval-dir", str(eval_dir), "--out", str(out)],
        check=False, capture_output=True, text=True,
    )
    # Script should still exit cleanly and produce a (mostly empty) PDF.
    assert res.returncode == 0, f"composite failed: {res.stderr}"
    assert out.exists() and out.stat().st_size > 0
