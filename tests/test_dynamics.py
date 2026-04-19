"""Sanity tests for the forward dynamics model.

These pin three invariants that an NLL-trained Gaussian dynamics model
should satisfy and one that the project's anti-collapse term should
guarantee. If any of these tests fail in CI, the dynamics model is
miscalibrated or mis-conditioned and downstream Agent C results are
suspect.

Cost is < 30 seconds on CPU; runs without a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foveated.models.dynamics import ForwardDynamics


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return ForwardDynamics(embed_dim=16, n_actions=4, hidden_dim=32, depth=2)


def test_forward_shapes(small_model):
    z = torch.randn(8, 16)
    a = torch.randint(0, 4, (8,))
    mu, log_s = small_model(z, a)
    assert mu.shape == (8, 16)
    assert log_s.shape == (8, 16)
    assert log_s.min().item() >= small_model.LOG_VAR_MIN - 1e-6
    assert log_s.max().item() <= small_model.LOG_VAR_MAX + 1e-6


def test_action_conditioning_gradient_flow(small_model):
    """Gradient w.r.t. the action one-hot must be non-zero somewhere.

    If the model ignores the action input the dynamics-conditional
    decision rule degenerates and Agent C reduces to Agent D.
    """
    z = torch.randn(8, 16, requires_grad=False)
    a = torch.randint(0, 4, (8,))
    a_onehot = F.one_hot(a, num_classes=4).float().requires_grad_(True)
    x = torch.cat([z, a_onehot], dim=-1)
    h = small_model.trunk(x)
    out = small_model.mean_head(h).sum()
    out.backward()
    assert a_onehot.grad is not None
    assert a_onehot.grad.abs().sum().item() > 1e-6, "model ignores action input"


def test_query_all_actions_matches_per_action_forward(small_model):
    """query_all_actions and forward must agree to floating-point error."""
    z = torch.randn(3, 16)
    mu_all, log_s_all = small_model.query_all_actions(z)
    assert mu_all.shape == (3, 4, 16)
    assert log_s_all.shape == (3, 4, 16)
    for action in range(4):
        a = torch.full((3,), action, dtype=torch.long)
        mu, log_s = small_model(z, a)
        torch.testing.assert_close(mu_all[:, action], mu)
        torch.testing.assert_close(log_s_all[:, action], log_s)


def test_prospective_uncertainty_masks_committed(small_model):
    z = torch.randn(2, 16)
    committed = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.float32)
    u = small_model.prospective_uncertainty(z, committed_mask=committed)
    assert u.shape == (2, 4)
    assert u[0, 0].item() < -1e3
    assert u[0, 2].item() < -1e3
    assert u[1, 1].item() < -1e3
    assert u[0, 1].item() > -1e3
    assert u[0, 3].item() > -1e3
    assert u[1, 3].item() > -1e3


def test_anti_collapse_under_stationary_targets():
    """When z_next == z_t for every sample, the anti-collapse term should keep
    the mean head from learning the identity. Even after many gradient steps
    the predicted mean should retain a residual gap to z_t.

    This is the project-specific design choice that prevents Agent C's
    feature from going to zero on stationary inputs.
    """
    torch.manual_seed(1)
    model = ForwardDynamics(embed_dim=8, n_actions=2, hidden_dim=16, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    z = torch.randn(64, 8)
    a = torch.randint(0, 2, (64,))
    z_next = z.clone()

    model.train()
    for _ in range(400):
        loss = model.nll_loss(z, a, z_next, action_dropout_p=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model(z, a)
        residual = (mu - z).pow(2).mean().item()
    assert residual > 1e-3, (
        "anti-collapse failed: model converged to identity on stationary targets"
    )


def test_calibration_on_synthetic_gaussian():
    """Train on (z_t, a_t, z_next) where z_next = mean(z_t, a_t) + noise of
    known variance sigma^2, and verify that the predicted log_s converges to
    log(sigma^2) within tolerance.

    The mean function is intentionally simple so the model has enough
    capacity to fit it; the variance head is the thing being calibrated.
    """
    torch.manual_seed(2)
    D, A = 8, 3
    model = ForwardDynamics(embed_dim=D, n_actions=A, hidden_dim=32, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    sigma2 = 0.25  # ground-truth variance
    log_sigma2 = float(np.log(sigma2))

    rng = torch.Generator().manual_seed(3)
    n = 2000

    def make_batch(bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn(bs, D, generator=rng)
        a = torch.randint(0, A, (bs,), generator=rng)
        a_onehot = F.one_hot(a, num_classes=A).float()
        # ground-truth mean function: linear in (z, a_onehot)
        true_W = torch.eye(D).repeat(1, 1)
        mean = z @ true_W + a_onehot[:, :1] * 0.1
        z_next = mean + (sigma2 ** 0.5) * torch.randn(bs, D, generator=rng)
        return z, a, z_next

    model.train()
    for _ in range(2000):
        z, a, z_next = make_batch(64)
        loss = model.nll_loss(z, a, z_next, action_dropout_p=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z, a, _ = make_batch(n)
        _, log_s = model(z, a)
        mean_log_s = float(log_s.mean().item())
    assert abs(mean_log_s - log_sigma2) < 0.4, (
        f"variance head miscalibrated: mean log_s={mean_log_s:.3f} vs "
        f"ground truth log(sigma^2)={log_sigma2:.3f}"
    )


def test_dynamics_train_step_decreases_loss():
    """Smoke test: a single AdamW step on random data should decrease NLL."""
    torch.manual_seed(4)
    model = ForwardDynamics(embed_dim=8, n_actions=2, hidden_dim=16, depth=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    z = torch.randn(32, 8)
    a = torch.randint(0, 2, (32,))
    z_next = z + 0.1 * torch.randn(32, 8)

    model.train()
    losses = []
    for _ in range(20):
        loss = model.nll_loss(z, a, z_next, action_dropout_p=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], "loss did not decrease over 20 steps"
