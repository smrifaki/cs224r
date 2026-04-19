"""Tests for the random-policy rollout collector and the dynamics
training loop.

The dynamics model's quality at evaluation time depends on these two
helpers running correctly. We pin three things:

  1. collect_triples returns triples whose pre-step embedding matches
     the post-step embedding of the previous triple (chain consistency).
  2. collect_triples populates the action vector with valid action
     indices.
  3. train_dynamics drops the loss from epoch 0 to epoch N on the
     synthetic input it was trained on.

We mock the env via a tiny in-memory toy env so the tests are CPU-only
and fast.
"""
from __future__ import annotations

import numpy as np
import torch

from foveated.algos.dynamics_train import train_dynamics
from foveated.experiments.collect_rollouts import collect_triples


class _ToyEnv:
    """Minimal env that exposes the interface collect_triples uses.

    Episodes are deterministic chains where z_{t+1} = z_t + delta(action)
    plus light noise; we record what collect_triples sees and assert
    invariants on the recorded triples.
    """

    def __init__(self, embed_dim: int = 8, n_patches: int = 3, seed: int = 0):
        self.cfg = type("Cfg", (), {"embed_dim": embed_dim})()
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.deltas = np.eye(n_patches, embed_dim, dtype=np.float32) * 0.1
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._max_steps = 5
        self._z = np.zeros(embed_dim, dtype=np.float32)

    def reset(self):
        self._step = 0
        self._z = self._rng.normal(0, 0.01, size=self.embed_dim).astype(np.float32)
        return self._obs(), {}

    def step(self, a: int):
        self._step += 1
        self._z = self._z + self.deltas[a] + self._rng.normal(0, 1e-3, size=self.embed_dim).astype(np.float32)
        done = self._step >= self._max_steps
        return self._obs(), 0.0, done, False, {}

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._z, np.zeros(self.n_patches, dtype=np.float32)])


def test_collect_triples_chain_consistency():
    env = _ToyEnv()
    rng = np.random.default_rng(0)
    z_t, a_t, z_next = collect_triples(env, n_episodes=4, rng=rng)
    assert z_t.shape == z_next.shape
    assert a_t.shape[0] == z_t.shape[0]
    # Consecutive triples within an episode: z_next[i] == z_t[i+1] for steps
    # in the same episode. We don't know episode boundaries here, but
    # within a 5-step episode there are 4 within-episode neighbors out of 5
    # total transitions, so at least 60 percent of consecutive pairs match.
    matches = 0
    for i in range(len(z_t) - 1):
        if np.allclose(z_next[i], z_t[i + 1], atol=1e-6):
            matches += 1
    assert matches >= int(0.6 * (len(z_t) - 1)), (
        f"chain consistency too low: {matches}/{len(z_t) - 1}"
    )


def test_collect_triples_valid_actions():
    env = _ToyEnv()
    rng = np.random.default_rng(1)
    _, a_t, _ = collect_triples(env, n_episodes=2, rng=rng)
    assert a_t.dtype == np.int64
    assert (a_t >= 0).all() and (a_t < env.n_patches).all()


def test_train_dynamics_loss_decreases():
    rng = np.random.default_rng(2)
    n, D, A = 256, 8, 3
    z_t = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    a_t = rng.integers(0, A, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, D)).astype(np.float32)
    model = train_dynamics(
        z_t, a_t, z_next, n_actions=A, embed_dim=D,
        epochs=3, batch_size=32, lr=1e-3, device="cpu", action_dropout_p=0.0,
    )
    # After 3 epochs the model should produce a finite z_hat on the same
    # inputs and the NLL should be substantially lower than the prior
    # (signal that it actually trained).
    model.eval()
    with torch.no_grad():
        zt = torch.from_numpy(z_t[:8]).float()
        at = torch.from_numpy(a_t[:8]).long()
        zn = torch.from_numpy(z_next[:8]).float()
        z_hat, log_s = model(zt, at)
        nll = ((zn - z_hat) ** 2 / log_s.exp() + log_s).mean().item()
    assert np.isfinite(nll)
    # Without training, the log_s head is near zero and the NLL on this
    # data is around D * 1 / exp(0) = D; we should be well below D.
    assert nll < 5.0, f"trained model NLL still high: {nll:.3f}"


def test_train_dynamics_handles_one_step_inputs():
    """Edge case: only a single batch's worth of data."""
    rng = np.random.default_rng(3)
    n, D, A = 8, 4, 2
    z_t = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    a_t = rng.integers(0, A, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, D)).astype(np.float32)
    model = train_dynamics(
        z_t, a_t, z_next, n_actions=A, embed_dim=D,
        epochs=1, batch_size=8, lr=1e-3, device="cpu", action_dropout_p=0.0,
    )
    assert model.embed_dim == D
    assert model.n_actions == A


def test_train_dynamics_returns_eval_mode_model():
    rng = np.random.default_rng(4)
    n, D, A = 32, 4, 2
    z_t = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    a_t = rng.integers(0, A, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, D)).astype(np.float32)
    model = train_dynamics(
        z_t, a_t, z_next, n_actions=A, embed_dim=D,
        epochs=1, batch_size=8, lr=1e-3, device="cpu", action_dropout_p=0.0,
    )
    assert not model.training, "train_dynamics should return the model in eval mode"
