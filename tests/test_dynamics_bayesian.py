"""Sanity tests for the last-layer Laplace approximation.

Three properties pin the implementation:
  1. The posterior covariance must be symmetric positive-definite per
     output dim (else inversion was wrong or numerics blew up).
  2. The epistemic predictive variance must shrink as more training
     data is used to compute the Fisher information.
  3. The prospective_epistemic_uncertainty helper must return shape
     (B, n_actions) and respect the committed-mask argument.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from foveated.models.dynamics import ForwardDynamics
from foveated.models.dynamics_bayesian import (
    epistemic_log_variance,
    fit_last_layer_laplace,
    prospective_epistemic_uncertainty,
)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return ForwardDynamics(embed_dim=4, n_actions=3, hidden_dim=8, depth=2)


def _tiny_training_data(n: int, embed_dim: int, n_actions: int, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(n, embed_dim, generator=rng)
    a = torch.randint(0, n_actions, (n,), generator=rng)
    a_onehot = F.one_hot(a, num_classes=n_actions).float()
    z_next = z + 0.1 * a_onehot[:, :1] + 0.2 * torch.randn(n, embed_dim, generator=rng)
    return z, a, z_next


def test_posterior_covariance_is_symmetric_psd(tiny_model):
    z, a, _ = _tiny_training_data(200, 4, 3)
    post = fit_last_layer_laplace(tiny_model, z, a, prior_precision=1.0)
    assert post.posterior_cov.shape == (4, 8, 8)
    for d in range(4):
        cov_d = post.posterior_cov[d]
        symm_err = (cov_d - cov_d.T).abs().max().item()
        assert symm_err < 1e-4, f"posterior cov[{d}] is not symmetric: err={symm_err}"
        eig = torch.linalg.eigvalsh(cov_d)
        min_eig = eig.min().item()
        assert min_eig > -1e-5, f"posterior cov[{d}] is not PSD: min eig={min_eig}"


def test_epistemic_log_variance_shapes(tiny_model):
    z, a, _ = _tiny_training_data(100, 4, 3)
    post = fit_last_layer_laplace(tiny_model, z, a)
    log_var = epistemic_log_variance(tiny_model, post, z[:5], a[:5])
    assert log_var.shape == (5, 4)
    assert torch.isfinite(log_var).all()


def test_epistemic_variance_shrinks_with_more_data(tiny_model):
    """More Fisher-information data means a tighter posterior and lower
    predictive epistemic variance at a fixed query point.
    """
    z_small, a_small, _ = _tiny_training_data(50, 4, 3, seed=1)
    z_big, a_big, _ = _tiny_training_data(2000, 4, 3, seed=1)
    post_small = fit_last_layer_laplace(tiny_model, z_small, a_small, prior_precision=1.0)
    post_big = fit_last_layer_laplace(tiny_model, z_big, a_big, prior_precision=1.0)

    z_query = torch.randn(32, 4, generator=torch.Generator().manual_seed(7))
    a_query = torch.randint(0, 3, (32,), generator=torch.Generator().manual_seed(7))
    log_var_small = epistemic_log_variance(tiny_model, post_small, z_query, a_query).mean().item()
    log_var_big = epistemic_log_variance(tiny_model, post_big, z_query, a_query).mean().item()
    assert log_var_big < log_var_small, (
        f"epistemic variance did not shrink with N: small={log_var_small:.3f} big={log_var_big:.3f}"
    )


def test_prospective_epistemic_masks_committed(tiny_model):
    z, a, _ = _tiny_training_data(200, 4, 3)
    post = fit_last_layer_laplace(tiny_model, z, a)
    z_query = z[:2]
    committed = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.float32)
    u = prospective_epistemic_uncertainty(tiny_model, post, z_query, committed_mask=committed)
    assert u.shape == (2, 3)
    assert u[0, 0].item() < -1e3
    assert u[0, 2].item() < -1e3
    assert u[1, 2].item() < -1e3
    assert u[0, 1].item() > -1e3
    assert u[1, 0].item() > -1e3
    assert u[1, 1].item() > -1e3


def test_prior_precision_increases_with_smaller_data():
    """At fixed prior_precision, infinitesimal training data should leave
    the posterior dominated by the prior, hence near (1 / prior_precision) * I.
    Sanity check that prior plumbing works.
    """
    model = ForwardDynamics(embed_dim=3, n_actions=2, hidden_dim=4, depth=1)
    z = torch.zeros(1, 3)
    a = torch.zeros(1, dtype=torch.long)
    post_strong_prior = fit_last_layer_laplace(model, z, a, prior_precision=100.0)
    post_weak_prior = fit_last_layer_laplace(model, z, a, prior_precision=0.01)
    diag_strong = post_strong_prior.posterior_cov[0].diag().mean().item()
    diag_weak = post_weak_prior.posterior_cov[0].diag().mean().item()
    assert diag_weak > diag_strong, "stronger prior should yield tighter (smaller) posterior"
