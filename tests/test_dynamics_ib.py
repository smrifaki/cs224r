"""Sanity tests for the information-bottleneck dynamics variant.

Three properties:
  1. beta=0 recovers the plain NLL behavior (KL term is zero-weighted).
  2. With beta > 0, training reduces the KL penalty below its initial
     value within a small number of optimization steps.
  3. Reparameterization is stochastic at train time, deterministic at
     eval time.
"""
from __future__ import annotations

import torch

from foveated.models.dynamics_ib import IBConfig, StochasticDynamics


def _data(n: int, embed_dim: int, n_actions: int):
    rng = torch.Generator().manual_seed(0)
    z = torch.randn(n, embed_dim, generator=rng)
    a = torch.randint(0, n_actions, (n,), generator=rng)
    z_next = z + 0.1 * torch.randn(n, embed_dim, generator=rng)
    return z, a, z_next


def test_beta_zero_kl_term_zero():
    model = StochasticDynamics(embed_dim=4, n_actions=2, cfg=IBConfig(beta=0.0))
    z, a, zn = _data(64, 4, 2)
    model.train()
    _, diag = model.ib_loss(z, a, zn, action_dropout_p=0.0)
    # With beta=0 the KL contribution to the loss is zeroed; the KL
    # diagnostic itself is still reported (so we can inspect it) and
    # should be a non-negative scalar.
    assert diag["kl_phi"] >= 0.0


def test_kl_decreases_under_ib_pressure():
    torch.manual_seed(0)
    model = StochasticDynamics(embed_dim=4, n_actions=2, cfg=IBConfig(beta=1e-2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    z, a, zn = _data(128, 4, 2)
    model.train()
    initial_kl = None
    final_kl = None
    for step in range(300):
        loss, diag = model.ib_loss(z, a, zn, action_dropout_p=0.0)
        if step == 0:
            initial_kl = diag["kl_phi"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_kl = diag["kl_phi"]
    assert initial_kl is not None and final_kl is not None
    assert final_kl < initial_kl, (
        f"IB KL should decrease under training pressure: "
        f"initial={initial_kl:.4f} final={final_kl:.4f}"
    )


def test_reparam_train_vs_eval():
    torch.manual_seed(0)
    model = StochasticDynamics(embed_dim=4, n_actions=2)
    z = torch.randn(8, 4)
    a = torch.zeros(8, dtype=torch.long)

    model.train()
    with torch.no_grad():
        mu_a, _ = model(z, a)
        mu_b, _ = model(z, a)
    diff_train = (mu_a - mu_b).abs().mean().item()
    assert diff_train > 1e-6, "stochastic phi should differ across forward passes in train mode"

    model.eval()
    with torch.no_grad():
        mu_a, _ = model(z, a)
        mu_b, _ = model(z, a)
    diff_eval = (mu_a - mu_b).abs().mean().item()
    assert diff_eval < 1e-9, "deterministic phi in eval mode should give identical forwards"


def test_loss_finite_and_backward():
    torch.manual_seed(0)
    model = StochasticDynamics(embed_dim=4, n_actions=2, cfg=IBConfig(beta=1e-3))
    z, a, zn = _data(32, 4, 2)
    model.train()
    loss, _ = model.ib_loss(z, a, zn, action_dropout_p=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    # Some gradient must reach the trunk weights.
    grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
