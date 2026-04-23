"""Tests for FoveatedClassificationEnv.

The env is the substrate everything else sits on (PPO, dynamics model,
evaluation). Tests pin its API contract: observation shape, action
space, episode termination behavior, reward structure, goal-conditioning
slot mapping.

We mock the BackboneAdapter and the dataset loader so the tests are
CPU-only and finish in well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from foveated.envs.foveated_env import (
    CORRUPTION_NAMES,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)


@dataclass
class _MockBackbone:
    """Stand-in for BackboneAdapter that returns deterministic tensors."""

    embed_dim: int = 384
    device: str = "cpu"

    def low_res(self, image, low_size):
        n_tokens = 16
        fmap = torch.zeros(self.embed_dim, n_tokens, n_tokens)
        pooled = torch.ones(self.embed_dim) * 0.1
        return fmap, pooled

    def patch_features(self, image, grid_h, grid_w):
        return torch.ones(self.embed_dim, grid_h, grid_w) * 0.2

    def classify(self, pooled):
        # Argmax always 0 so accuracy is 1 if label is 0, 0 otherwise.
        logits = torch.zeros(1000)
        logits[0] = 10.0
        return logits.squeeze()


@pytest.fixture
def env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=384,
        grid_h=7,
        grid_w=7,
        max_patches=4,
        patch_cost=0.02,
        backbone_name="mock",
    )
    paths = [tmp_path / f"img_{i}.jpeg" for i in range(8)]
    labels = [0] * 8  # always label 0 so mock classify scores accuracy 1.0
    # Touch dummy files so load_and_corrupt finds them; the mock backbone
    # does not actually read pixel content, but PIL.Image.open is called
    # inside reset, so the files must exist. We monkeypatch load_and_corrupt
    # to skip PIL entirely.
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(cfg=cfg, image_paths=paths, labels=labels, backbone=_MockBackbone(), device="cpu")


def test_observation_shape(env):
    obs, info = env.reset()
    cfg = env.cfg
    expected = cfg.embed_dim + env.n_patches + env.goal_dim
    assert obs.shape == (expected,)
    assert "goal_slot" in info


def test_action_space(env):
    assert env.action_space.n == env.n_patches + (1 if env.cfg.allow_stop_action else 0)


def test_step_increments_commit_mask(env):
    env.reset()
    n_before = int(env.committed_mask.sum())
    _, _, term, trunc, info = env.step(0)
    assert info["committed_patches"] == [0]
    assert int(env.committed_mask.sum()) == n_before + 1
    assert env.committed_mask[0] == 1.0


def test_repick_terminates(env):
    env.reset()
    env.step(0)
    _, _, term, trunc, _ = env.step(0)
    assert term, "re-picking a committed patch should terminate"


def test_truncation_at_max_patches(env):
    env.reset()
    truncated = False
    for i in range(env.cfg.max_patches):
        _, _, term, trunc, _ = env.step(i)
        truncated = trunc
    assert truncated, "should truncate at max_patches"


def test_reward_includes_patch_cost(env):
    env.reset()
    _, reward, _, _, _ = env.step(0)
    # Mid-episode reward should be exactly -patch_cost.
    assert reward == pytest.approx(-env.cfg.patch_cost, abs=1e-9)


def test_terminal_reward_includes_classification(env):
    env.reset()
    final_reward = 0.0
    for i in range(env.cfg.max_patches):
        _, r, _, trunc, _ = env.step(i)
        final_reward = r
        if trunc:
            break
    # Mock classifier always picks 0 == label, so accuracy is 1.0; final
    # step reward is 1.0 - patch_cost.
    assert final_reward == pytest.approx(1.0 - env.cfg.patch_cost, abs=1e-9)


def test_goal_slot_for_clean(env):
    obs, info = env.reset()
    # cfg.corruption_type is None -> slot 0
    assert info["goal_slot"] == 0


def test_goal_slot_for_known_corruption(env, tmp_path):
    # Build a fresh env with a corruption set.
    cfg = FoveatedEnvConfig(
        embed_dim=384, grid_h=7, grid_w=7, max_patches=4,
        corruption_type="snow", corruption_severity=3,
    )
    paths = [tmp_path / f"img_{i}.jpeg" for i in range(2)]
    e = FoveatedClassificationEnv(cfg=cfg, image_paths=paths, labels=[0, 0],
                                  backbone=_MockBackbone(), device="cpu")
    _, info = e.reset()
    expected_slot = 1 + CORRUPTION_NAMES.index("snow")
    assert info["goal_slot"] == expected_slot


def test_current_state_keys(env):
    env.reset()
    env.step(0)
    snap = env.current_state()
    assert set(snap.keys()) == {"z_t", "committed_mask", "committed_patches", "patch_embeds", "label"}
    assert snap["committed_patches"] == [0]
    assert isinstance(snap["z_t"], np.ndarray)
