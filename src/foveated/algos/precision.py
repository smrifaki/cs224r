"""Precision-weighted residual.

Used by the dynamics training loop, by Agent B's intrinsic reward, and (in
the retrospective form) by post-hoc analysis. The prospective uncertainty
signal used by Agent C at decision time lives on the dynamics model itself
(`ForwardDynamics.prospective_uncertainty`).
"""
from __future__ import annotations

import torch


def precision_weighted_residual(
    z_obs: torch.Tensor, z_hat: torch.Tensor, log_s: torch.Tensor
) -> torch.Tensor:
    """Per-coordinate residual scaled by predicted precision exp(-log_s/2)."""
    return (z_obs - z_hat) * (-log_s / 2).exp()
