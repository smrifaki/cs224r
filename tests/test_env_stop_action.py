"""Edge-case tests for FoveatedClassificationEnv with the optional
stop action enabled.

The default config has `allow_stop_action=False`. The Phase 1 spec
defers the stop-action ablation to Phase 5; this test pins the
behavior so the ablation does not regress when it is run.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from foveated.envs.foveated_env import (
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)


@dataclass
class _MockBackbone:
    embed_dim: int = 64
    device: str = "cpu"

    def low_res(self, image, low_size):
        return torch.zeros(self.embed_dim, 4, 4), torch.zeros(self.embed_dim)

    def patch_features(self, image, grid_h, grid_w):
        return torch.ones(self.embed_dim, grid_h, grid_w) * 0.1

    def classify(self, pooled):
        logits = torch.zeros(1000)
        logits[0] = 10.0
        return logits.squeeze()


def _build_env(tmp_path, allow_stop_action: bool):
    cfg = FoveatedEnvConfig(
        embed_dim=64, grid_h=7, grid_w=7, max_patches=8,
        backbone_name="mock", allow_stop_action=allow_stop_action,
    )
    paths = [tmp_path / "img_0.jpeg", tmp_path / "img_1.jpeg"]
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=[0, 0],
        backbone=_MockBackbone(), device="cpu",
    )


def test_stop_action_present(tmp_path):
    env = _build_env(tmp_path, allow_stop_action=True)
    assert env.action_space.n == env.n_patches + 1
    assert env.stop_action == env.n_patches


def test_stop_action_absent_by_default(tmp_path):
    env = _build_env(tmp_path, allow_stop_action=False)
    assert env.action_space.n == env.n_patches
    assert env.stop_action is None


def test_stop_action_terminates_immediately(tmp_path):
    env = _build_env(tmp_path, allow_stop_action=True)
    env.reset()
    _, reward, term, _, info = env.step(env.stop_action)
    assert term, "stop action must terminate the episode"
    assert info["committed_patches"] == []
    # Terminal reward is classification accuracy on the bare low-res
    # assembly (no patches committed). Mock classifier always picks 0.
    assert reward == pytest.approx(1.0, abs=1e-9)


def test_stop_action_after_one_commit(tmp_path):
    env = _build_env(tmp_path, allow_stop_action=True)
    env.reset()
    env.step(0)
    _, reward, term, _, info = env.step(env.stop_action)
    assert term
    assert info["committed_patches"] == [0]
    # Reward = accuracy - patch_cost * 1.
    assert reward == pytest.approx(1.0 - env.cfg.patch_cost, abs=1e-9)


def test_stop_action_indexing_consistent(tmp_path):
    """Action n_patches is the stop action; action n_patches + 1 is out of range."""
    env = _build_env(tmp_path, allow_stop_action=True)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(env.n_actions)
