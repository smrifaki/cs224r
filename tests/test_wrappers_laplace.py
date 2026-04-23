"""Tests for the Laplace-using path in ProspectiveUncertaintyObsWrapper.

When the wrapper is constructed with a LaplacePosterior, the appended
slot must be the BALD-aligned epistemic log-variance from the
last-layer Laplace approximation, not the total predictive
log-variance from the dynamics model's logvar_head. We pin two
properties:

  1. The wrapper accepts a LaplacePosterior and produces finite,
     correctly-shaped observations.
  2. With and without the posterior, the appended slot is
     materially different (sanity check that the posterior path
     is actually being used).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foveated.envs.feature_wrappers import ProspectiveUncertaintyObsWrapper
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.models.dynamics import ForwardDynamics
from foveated.models.dynamics_bayesian import fit_last_layer_laplace


@dataclass
class _MockBackbone:
    embed_dim: int = 384
    device: str = "cpu"

    def low_res(self, image, low_size):
        n = 16
        return torch.zeros(self.embed_dim, n, n), torch.ones(self.embed_dim) * 0.1

    def patch_features(self, image, grid_h, grid_w):
        return torch.ones(self.embed_dim, grid_h, grid_w) * 0.2

    def classify(self, pooled):
        logits = torch.zeros(1000)
        logits[0] = 10.0
        return logits.squeeze()


def _build_env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=384, grid_h=7, grid_w=7, max_patches=4, backbone_name="mock",
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


def _build_dynamics_and_posterior(env):
    torch.manual_seed(0)
    dyn = ForwardDynamics(
        embed_dim=env.cfg.embed_dim, n_actions=env.n_patches,
        hidden_dim=32, depth=2,
    )
    z = torch.randn(64, env.cfg.embed_dim)
    a = torch.randint(0, env.n_patches, (64,))
    posterior = fit_last_layer_laplace(dyn, z, a, prior_precision=1.0)
    return dyn, posterior


def test_laplace_wrapper_produces_finite_obs(tmp_path):
    env = _build_env(tmp_path)
    dyn, post = _build_dynamics_and_posterior(env)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped = ProspectiveUncertaintyObsWrapper(env, dynamics=dyn, posterior=post, device="cpu")
    obs, _ = wrapped.reset()
    assert obs.shape == (base_dim + env.n_actions,)
    assert np.isfinite(obs).all()


def test_laplace_vs_total_variance_differ(tmp_path):
    env = _build_env(tmp_path)
    dyn, post = _build_dynamics_and_posterior(env)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped_total = ProspectiveUncertaintyObsWrapper(env, dynamics=dyn, posterior=None, device="cpu")
    obs_total, _ = wrapped_total.reset()

    env2 = _build_env(tmp_path)
    wrapped_laplace = ProspectiveUncertaintyObsWrapper(env2, dynamics=dyn, posterior=post, device="cpu")
    obs_lap, _ = wrapped_laplace.reset()

    appended_total = obs_total[base_dim:]
    appended_lap = obs_lap[base_dim:]
    diff = float(np.abs(appended_total - appended_lap).max())
    assert diff > 1e-6, "Laplace path should differ from total-variance path"


def test_laplace_wrapper_masks_committed_action(tmp_path):
    env = _build_env(tmp_path)
    dyn, post = _build_dynamics_and_posterior(env)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped = ProspectiveUncertaintyObsWrapper(env, dynamics=dyn, posterior=post, device="cpu")
    wrapped.reset()
    obs, _, _, _, _ = wrapped.step(0)
    appended = obs[base_dim:]
    assert appended[0] < -1e3, "committed action 0 should be masked under the Laplace path"
    assert appended[1] > -1e3
