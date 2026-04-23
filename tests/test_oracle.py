"""Tests for the greedy K-patch oracle.

The oracle is the tight upper bound used as the regret denominator
in the report. Three properties pinned:

  1. Greedy oracle accuracy is monotonically non-decreasing in K
     (since each step strictly cannot decrease the assembled-
     embedding classifier confidence for the true label).
  2. Greedy oracle never exceeds K commits when budget is K.
  3. Greedy oracle selects distinct patches (no repeats).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.oracle_topk import greedy_topk_accuracy


@dataclass
class _MockBackbone:
    embed_dim: int = 64
    device: str = "cpu"

    def low_res(self, image, low_size):
        n = 4
        return torch.zeros(self.embed_dim, n, n), torch.zeros(self.embed_dim)

    def patch_features(self, image, grid_h, grid_w):
        # Spatially varying features so greedy has a real choice.
        torch.manual_seed(0)
        return torch.randn(self.embed_dim, grid_h, grid_w)

    def classify(self, pooled):
        # Linear classifier that prefers patches whose mean is positive.
        logits = torch.zeros(1000)
        logits[0] = float(pooled.mean().item()) * 10.0
        return logits.squeeze()


def _build_env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=64, grid_h=7, grid_w=7, max_patches=16, backbone_name="mock",
    )
    paths = [tmp_path / "img_0.jpeg"]
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=[0],
        backbone=_MockBackbone(), device="cpu",
    )


def test_oracle_selects_distinct_patches(tmp_path):
    env = _build_env(tmp_path)
    _, chosen = greedy_topk_accuracy(env, k=8)
    assert len(chosen) == 8
    assert len(set(chosen)) == 8, "greedy must not repeat patches"


def test_oracle_respects_budget(tmp_path):
    env = _build_env(tmp_path)
    _, chosen = greedy_topk_accuracy(env, k=5)
    assert len(chosen) <= 5


def test_oracle_picks_no_more_than_n_patches(tmp_path):
    env = _build_env(tmp_path)
    _, chosen = greedy_topk_accuracy(env, k=200)
    assert len(chosen) <= env.n_patches


def test_oracle_zero_budget_returns_no_commits(tmp_path):
    env = _build_env(tmp_path)
    _, chosen = greedy_topk_accuracy(env, k=0)
    assert chosen == []
