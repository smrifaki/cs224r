"""Dynamics-model uncertainty calibration under clean vs ImageNet-C inputs.

The sufficient-statistic claim (see docs/sufficient_statistic_claim.md)
holds under the assumption that the dynamics model is well-calibrated.
ImageNet-C is exactly the regime where calibration breaks. This script
quantifies how badly.

Procedure: for each corruption type, collect (residual^2, predicted
variance) pairs from random-policy rollouts. The reliability curve plots
binned predicted variance vs realized squared residual. Expected
Calibration Error (ECE) is the L1 distance between the curve and the
diagonal.

Outputs:
    runs/eval/calibration_reliability.pdf , reliability curves by corruption
    runs/eval/calibration_ece.csv         , ECE per corruption type
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import (
    CORRUPTION_NAMES,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)
from foveated.experiments.collect_rollouts import collect_triples
from foveated.models.dynamics import ForwardDynamics

N_BINS = 10
EVAL_CORRUPTIONS: tuple[str | None, ...] = (None,) + CORRUPTION_NAMES


@torch.no_grad()
def _residuals_and_variances(
    dyn: ForwardDynamics,
    z_t: np.ndarray,
    a_t: np.ndarray,
    z_next: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of (squared_residual_per_sample, predicted_variance_per_sample).

    Squared residual is averaged over the embedding dim. Predicted variance is
    exp(log_s) averaged over the embedding dim.
    """
    sq_res_all: list[np.ndarray] = []
    var_all: list[np.ndarray] = []
    for i in range(0, len(z_t), batch_size):
        zt = torch.from_numpy(z_t[i : i + batch_size]).float().to(device)
        at = torch.from_numpy(a_t[i : i + batch_size]).long().to(device)
        zn = torch.from_numpy(z_next[i : i + batch_size]).float().to(device)
        z_hat, log_s = dyn(zt, at)
        sq_res = (zn - z_hat).pow(2).mean(dim=-1).cpu().numpy()
        var = log_s.exp().mean(dim=-1).cpu().numpy()
        sq_res_all.append(sq_res)
        var_all.append(var)
    return np.concatenate(sq_res_all), np.concatenate(var_all)


def reliability_curve(
    sq_res: np.ndarray, var: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Bin by predicted variance, return (bin_mean_var, bin_mean_sq_res)."""
    edges = np.quantile(var, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.digitize(var, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    bin_var = np.array([var[idx == k].mean() if (idx == k).any() else np.nan for k in range(n_bins)])
    bin_sq = np.array([sq_res[idx == k].mean() if (idx == k).any() else np.nan for k in range(n_bins)])
    return bin_var, bin_sq


def expected_calibration_error(sq_res: np.ndarray, var: np.ndarray, n_bins: int = N_BINS) -> float:
    bin_var, bin_sq = reliability_curve(sq_res, var, n_bins=n_bins)
    mask = ~(np.isnan(bin_var) | np.isnan(bin_sq))
    if not mask.any():
        return float("nan")
    return float(np.abs(bin_var[mask] - bin_sq[mask]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--dynamics-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=256)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)

    ckpt = torch.load(args.dynamics_ckpt, map_location=args.device)
    dyn = ForwardDynamics(**ckpt["config"]).to(args.device).eval()
    dyn.load_state_dict(ckpt["state_dict"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")

    ece_rows: list[dict] = []
    for corruption in EVAL_CORRUPTIONS:
        cfg = FoveatedEnvConfig(
            seed=args.seed,
            backbone_name=args.backbone,
            embed_dim=384 if "vit" in args.backbone.lower() else 2048,
            corruption_type=corruption,
            corruption_severity=args.severity if corruption else 0,
        )
        env = FoveatedClassificationEnv(cfg=cfg, image_paths=paths, labels=labels, backbone=backbone, device=args.device)
        rng = np.random.default_rng(args.seed)
        z_t, a_t, z_next = collect_triples(env, args.n_episodes, rng)
        sq_res, var = _residuals_and_variances(dyn, z_t, a_t, z_next, device=args.device)

        bin_var, bin_sq = reliability_curve(sq_res, var)
        ece = expected_calibration_error(sq_res, var)
        label = corruption if corruption else "clean"
        ax.plot(bin_var, bin_sq, marker="o", label=f"{label} (ECE={ece:.3f})", alpha=0.7)
        ece_rows.append({
            "corruption": label,
            "severity": args.severity if corruption else 0,
            "ece": ece,
            "n_samples": int(len(sq_res)),
        })
        print(f"{label:18s}  ECE={ece:.4f}  n={len(sq_res)}")

    ax.set_xlabel("Predicted variance (bin mean)")
    ax.set_ylabel("Realized squared residual (bin mean)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "calibration_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    with (args.out_dir / "calibration_ece.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "severity", "ece", "n_samples"])
        w.writeheader()
        w.writerows(ece_rows)
    print(f"saved {args.out_dir}/calibration_reliability.pdf")
    print(f"saved {args.out_dir}/calibration_ece.csv")


if __name__ == "__main__":
    main()
