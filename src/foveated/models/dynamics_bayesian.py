"""Last-layer Laplace approximation on the dynamics model.

The plain ForwardDynamics outputs a Gaussian (mu, log_s) at each input. By
construction the log_s head learns total predictive variance, which mixes
ALEATORIC (irreducible noise in the next embedding) and EPISTEMIC (the
model's uncertainty about its own parameters). BALD-style information gain
selects on epistemic uncertainty only , aleatoric uncertainty does not
reduce after observing the patch. Using total variance as Agent C's
feature is the textbook BALD failure mode.

This module wraps a trained ForwardDynamics with a last-layer Laplace
approximation (Kristiadi et al. 2020), giving a closed-form epistemic-
variance estimate that is cheap at inference and easy to plug into
prospective_uncertainty without retraining. The architecture is:

    z_t, a_t -> trunk -> phi (penultimate features, dim H)
                          -> mean_head    (linear, output dim D)
                          -> logvar_head  (linear, output dim D, frozen MAP)

The MAP-trained mean_head weights W_mu live in R^{D x H}. We treat the
posterior over W_mu as Gaussian with mean W_mu^MAP and covariance equal to
the inverse of the Gauss-Newton approximation of the Hessian of the NLL
w.r.t. W_mu evaluated at convergence. Predictive epistemic variance per
embedding-dim and per (z, a):

    Var_epistemic[d](z, a) = phi(z, a)^T  H^{-1}_d  phi(z, a)

where H_d is the Hessian block for output dim d. The Gauss-Newton
approximation collapses H_d to the precision-weighted feature-covariance
sum_i phi_i phi_i^T / sigma^2_i + prior_precision * I. We store this
matrix once after training; inference is a single bilinear form per dim.

The total variance Var_predictive = Var_aleatoric + Var_epistemic where
Var_aleatoric = exp(log_s). Agent C should select on Var_epistemic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from foveated.models.dynamics import ForwardDynamics


@dataclass
class LaplacePosterior:
    """Posterior precision and feature-extraction for one (output_dim, hidden_dim) head."""

    posterior_cov: torch.Tensor  # (D, H, H) per-output-dim
    prior_precision: float


def _trunk_features(model: ForwardDynamics, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
    a_onehot = F.one_hot(a_t, num_classes=model.n_actions).float()
    x = torch.cat([z_t, a_onehot], dim=-1)
    return model.trunk(x)


@torch.no_grad()
def fit_last_layer_laplace(
    model: ForwardDynamics,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    prior_precision: float = 1.0,
    chunk: int = 4096,
) -> LaplacePosterior:
    """Compute the Gauss-Newton posterior covariance for mean_head weights.

    Inputs are the same triples used to train the dynamics model. Outputs a
    LaplacePosterior carrying a (D, H, H) tensor of per-output-dim posterior
    covariance matrices. Cost is O(N * D * H^2) one-time; storage is D * H^2.

    For ForwardDynamics defaults (D=embed_dim=384, H=hidden_dim=512) this is
    384 * 512^2 ~= 100M floats. Use bfloat16 if memory is tight. For the
    canonical CS224R scale this is fine.
    """
    device = z_t.device
    model.eval()
    H = cast(nn.Linear, model.mean_head).in_features
    D = cast(nn.Linear, model.mean_head).out_features

    info = torch.zeros(D, H, H, device=device, dtype=torch.float32)
    prior = prior_precision * torch.eye(H, device=device, dtype=torch.float32).unsqueeze(0)

    for i in range(0, len(z_t), chunk):
        z = z_t[i : i + chunk]
        a = a_t[i : i + chunk]
        phi = _trunk_features(model, z, a)
        log_s = model.logvar_head(phi).clamp(model.LOG_VAR_MIN, model.LOG_VAR_MAX)
        precision = (-log_s).exp()  # (B, D)
        # Per-output-dim Fisher contribution: sum_b precision_b[d] * phi_b phi_b^T
        # Implemented as einsum to keep the D axis explicit.
        info += torch.einsum("bd,bh,bk->dhk", precision, phi, phi)

    info += prior  # broadcasts prior over D output dims
    cov = torch.linalg.inv(info)  # (D, H, H)
    return LaplacePosterior(posterior_cov=cov, prior_precision=prior_precision)


@torch.no_grad()
def epistemic_log_variance(
    model: ForwardDynamics,
    posterior: LaplacePosterior,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
) -> torch.Tensor:
    """Return per-(sample, output_dim) epistemic log-variance.

    Var_epistemic[b, d] = phi(z_b, a_b)^T cov[d] phi(z_b, a_b).
    """
    phi = _trunk_features(model, z_t, a_t)
    var = torch.einsum("bh,dhk,bk->bd", phi, posterior.posterior_cov, phi)
    var = var.clamp_min(1e-12)
    return var.log()


@torch.no_grad()
def prospective_epistemic_uncertainty(
    model: ForwardDynamics,
    posterior: LaplacePosterior,
    z_t: torch.Tensor,
    committed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-action epistemic log-variance summary, BALD-aligned.

    Mirrors model.prospective_uncertainty but returns the EPISTEMIC component
    via the Laplace posterior rather than the total log_s. Already-committed
    actions get masked to a large negative value.
    """
    b = z_t.shape[0]
    n = model.n_actions
    z_rep = z_t.unsqueeze(1).expand(b, n, model.embed_dim).reshape(b * n, -1)
    a_rep = torch.arange(n, device=z_t.device).repeat(b)
    log_eps = epistemic_log_variance(model, posterior, z_rep, a_rep)  # (b*n, D)
    score = log_eps.mean(dim=-1).view(b, n)
    if committed_mask is not None:
        score = score.masked_fill(committed_mask.bool(), -1e4)
    return score
