"""Integration test for the dynamics-train, Laplace-fit, wrapper chain.

The three pieces have unit tests individually. This test exercises
the chain end-to-end on synthetic data to catch interface mismatches
that the unit tests can miss.

Sequence:
  1. Train ForwardDynamics on synthetic (z_t, a_t, z_next) triples.
  2. Fit a last-layer Laplace posterior on the trained model.
  3. Construct a ProspectiveUncertaintyObsWrapper with the posterior.
  4. Reset and step the wrapper; verify the appended slot is the
     BALD-aligned epistemic log-variance (differs from the total
     log-variance path, finite, respects committed-action masking).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foveated.algos.dynamics_train import train_dynamics
from foveated.envs.feature_wrappers import ProspectiveUncertaintyObsWrapper
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.models.dynamics_bayesian import fit_last_layer_laplace


@dataclass
class _MockBackbone:
    embed_dim: int = 16
    device: str = "cpu"

    def low_res(self, image, low_size):
        return torch.zeros(self.embed_dim, 4, 4), torch.zeros(self.embed_dim)

    def patch_features(self, image, grid_h, grid_w):
        torch.manual_seed(0)
        return torch.randn(self.embed_dim, grid_h, grid_w) * 0.1

    def classify(self, pooled):
        logits = torch.zeros(1000)
        logits[0] = float(pooled.mean().item()) * 10.0
        return logits.squeeze()


def _build_env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=16, grid_h=4, grid_w=4, max_patches=4, backbone_name="mock",
    )
    paths = [tmp_path / f"img_{i}.jpeg" for i in range(4)]
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=[0] * 4,
        backbone=_MockBackbone(), device="cpu",
    )


def test_dynamics_train_then_laplace_then_wrapper_chain(tmp_path):
    env = _build_env(tmp_path)

    rng = np.random.default_rng(0)
    n, D, A = 128, 16, env.n_patches
    z_t = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    a_t = rng.integers(0, A, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, D)).astype(np.float32)

    model = train_dynamics(
        z_t, a_t, z_next,
        n_actions=A, embed_dim=D,
        epochs=2, batch_size=16, lr=1e-3, device="cpu", action_dropout_p=0.0,
    )

    z_t_t = torch.from_numpy(z_t).float()
    a_t_t = torch.from_numpy(a_t).long()
    posterior = fit_last_layer_laplace(model, z_t_t, a_t_t, prior_precision=1.0)

    base_dim = env.observation_space.shape[0]  # type: ignore[index]

    # Total-variance path.
    wrapped_total = ProspectiveUncertaintyObsWrapper(
        env, dynamics=model, posterior=None, device="cpu"
    )
    obs_total, _ = wrapped_total.reset()
    appended_total = obs_total[base_dim:]

    # Laplace path on a fresh env.
    env2 = _build_env(tmp_path)
    wrapped_lap = ProspectiveUncertaintyObsWrapper(
        env2, dynamics=model, posterior=posterior, device="cpu"
    )
    obs_lap, _ = wrapped_lap.reset()
    appended_lap = obs_lap[base_dim:]

    assert np.isfinite(appended_total).all()
    assert np.isfinite(appended_lap).all()
    diff = float(np.abs(appended_total - appended_lap).max())
    assert diff > 1e-6, "Laplace path should differ from total-variance path"


def test_chain_masks_committed_action(tmp_path):
    env = _build_env(tmp_path)
    rng = np.random.default_rng(1)
    n, D, A = 64, 16, env.n_patches
    z_t = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    a_t = rng.integers(0, A, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, D)).astype(np.float32)
    model = train_dynamics(
        z_t, a_t, z_next, n_actions=A, embed_dim=D,
        epochs=1, batch_size=16, lr=1e-3, device="cpu", action_dropout_p=0.0,
    )
    posterior = fit_last_layer_laplace(
        model, torch.from_numpy(z_t), torch.from_numpy(a_t), prior_precision=1.0
    )

    wrapped = ProspectiveUncertaintyObsWrapper(
        env, dynamics=model, posterior=posterior, device="cpu"
    )
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped.reset()
    obs, _, _, _, _ = wrapped.step(0)
    appended = obs[base_dim:]
    assert appended[0] < -1e3
    assert appended[1] > -1e3
