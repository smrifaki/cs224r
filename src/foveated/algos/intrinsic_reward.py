"""Pathak-style intrinsic reward for Agent B.

reward_intrinsic_t = beta * || precision-weighted residual ||_2^2

beta anneals linearly from beta_start to beta_end over `anneal_steps` env
steps. Default annealing keeps intrinsic from dominating extrinsic late in
training, when the policy has already learned to sample informative patches
and the extrinsic gradient should be the strongest signal.
"""
from __future__ import annotations

import torch

from foveated.algos.precision import precision_weighted_residual


def intrinsic_reward(
    z_obs: torch.Tensor,
    z_hat: torch.Tensor,
    log_s: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    e = precision_weighted_residual(z_obs, z_hat, log_s)
    return beta * e.pow(2).sum(dim=-1)
