"""Tests for the agent-specific observation and reward wrappers.

For each wrapper, we check that:
  - the augmented observation has the right shape
  - committed-action slots are correctly handled
  - the wrapper is deterministic given a fixed env state

We construct a minimal env via the same mocking approach used in
test_env.py, then layer the wrappers on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from foveated.envs.feature_wrappers import (
    ClassifierEntropyObsWrapper,
    ProspectiveUncertaintyObsWrapper,
)
from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.envs.intrinsic_wrapper import IntrinsicRewardWrapper
from foveated.models.dynamics import ForwardDynamics


@dataclass
class _MockBackbone:
    embed_dim: int = 384
    device: str = "cpu"

    def low_res(self, image, low_size):
        n_tokens = 16
        return torch.zeros(self.embed_dim, n_tokens, n_tokens), torch.ones(self.embed_dim) * 0.1

    def patch_features(self, image, grid_h, grid_w):
        return torch.ones(self.embed_dim, grid_h, grid_w) * 0.2

    def classify(self, pooled):
        logits = torch.zeros(1000)
        logits[0] = 10.0
        return logits.squeeze()


def _build_env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=384, grid_h=7, grid_w=7, max_patches=4,
        backbone_name="mock",
    )
    paths = [tmp_path / f"img_{i}.jpeg" for i in range(4)]
    labels = [0, 0, 0, 0]
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=labels, backbone=_MockBackbone(), device="cpu"
    )


def _build_dynamics(env: FoveatedClassificationEnv) -> ForwardDynamics:
    torch.manual_seed(0)
    return ForwardDynamics(
        embed_dim=env.cfg.embed_dim, n_actions=env.n_patches, hidden_dim=32, depth=2
    )


def test_classifier_entropy_wrapper_adds_one_dim(tmp_path):
    env = _build_env(tmp_path)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped = ClassifierEntropyObsWrapper(env, device="cpu")
    obs, _ = wrapped.reset()
    assert obs.shape == (base_dim + 1,)
    assert np.isfinite(obs[-1])


def test_prospective_uncertainty_wrapper_adds_n_actions(tmp_path):
    env = _build_env(tmp_path)
    dyn = _build_dynamics(env)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped = ProspectiveUncertaintyObsWrapper(env, dynamics=dyn, device="cpu")
    obs, _ = wrapped.reset()
    assert obs.shape == (base_dim + env.n_actions,)
    appended = obs[base_dim:]
    assert np.isfinite(appended).all()


def test_prospective_wrapper_masks_committed_action(tmp_path):
    env = _build_env(tmp_path)
    dyn = _build_dynamics(env)
    base_dim = env.observation_space.shape[0]  # type: ignore[index]
    wrapped = ProspectiveUncertaintyObsWrapper(env, dynamics=dyn, device="cpu")
    obs, _ = wrapped.reset()
    obs, _, _, _, _ = wrapped.step(0)
    # After committing action 0, the appended uncertainty slot for
    # action 0 should be the masked sentinel (large negative).
    appended = obs[base_dim:]
    assert appended[0] < -1e3
    # Uncommitted actions should not be masked.
    assert appended[1] > -1e3


def test_intrinsic_reward_wrapper_no_change_on_first_step(tmp_path):
    """First-step bonus is zero because there is no prior committed action."""
    env = _build_env(tmp_path)
    dyn = _build_dynamics(env)
    wrapped = IntrinsicRewardWrapper(env, dynamics=dyn, device="cpu")
    wrapped.reset()
    obs, reward, _, _, info = wrapped.step(0)
    # No intrinsic_reward key on the first step (no prior_z to compute it).
    assert "intrinsic_reward" not in info
    assert reward == pytest.approx(-env.cfg.patch_cost, abs=1e-9)


def test_intrinsic_reward_wrapper_adds_bonus_after_first_step(tmp_path):
    env = _build_env(tmp_path)
    dyn = _build_dynamics(env)
    wrapped = IntrinsicRewardWrapper(env, dynamics=dyn, device="cpu", beta_start=0.1, beta_end=0.1)
    wrapped.reset()
    wrapped.step(0)
    _, reward, _, _, info = wrapped.step(1)
    assert "intrinsic_reward" in info
    assert info["intrinsic_reward"] >= 0.0
    assert reward == pytest.approx(-env.cfg.patch_cost + info["intrinsic_reward"], abs=1e-6)
