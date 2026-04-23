"""Tests for the Lipschitz-constant estimator.

The estimator samples assembled embeddings and computes
||grad_e log_softmax_true_class(classify(e))||_2. We can verify the
gradient is correctly computed against a small linear classifier
whose Lipschitz constant we know analytically.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from foveated.envs.foveated_env import FoveatedClassificationEnv, FoveatedEnvConfig
from foveated.experiments.lipschitz_estimate import _embedding_grad_norm, estimate_constants


@dataclass
class _LinearBackbone:
    """Backbone where classify(e) = W @ e for a fixed W.

    For a linear logits = W e and log_softmax_k = w_k.e - logsumexp(W e),
    the gradient w.r.t. e is w_k - sum_j softmax(W e)_j w_j. We pick W
    such that the true class is class 0 with w_0 = unit vector along
    dim 0 and other w_j zero; then near typical inputs the gradient
    norm is bounded by ||w_0|| + ||sum_j softmax_j w_j||_2 = 1 + 0 = 1
    in the limit of perfect classification.
    """

    embed_dim: int = 8
    device: str = "cpu"

    def __post_init__(self):
        self.W = torch.zeros(1000, self.embed_dim)
        self.W[0, 0] = 1.0  # only class 0 has a non-zero weight along dim 0

    def low_res(self, image, low_size):
        return torch.zeros(self.embed_dim, 4, 4), torch.zeros(self.embed_dim)

    def patch_features(self, image, grid_h, grid_w):
        torch.manual_seed(0)
        return torch.randn(self.embed_dim, grid_h, grid_w) * 0.1

    def classify(self, pooled):
        return self.W @ pooled


def _build_env(tmp_path):
    cfg = FoveatedEnvConfig(
        embed_dim=8, grid_h=7, grid_w=7, max_patches=4, backbone_name="mock",
    )
    paths = [tmp_path / "img_0.jpeg"]
    import foveated.envs.foveated_env as fe

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    return FoveatedClassificationEnv(
        cfg=cfg, image_paths=paths, labels=[0],
        backbone=_LinearBackbone(), device="cpu",
    )


def test_embedding_grad_norm_finite_and_positive(tmp_path):
    env = _build_env(tmp_path)
    env.reset()
    import numpy as np

    embed = np.ones(env.cfg.embed_dim, dtype=np.float32)
    g = _embedding_grad_norm(env, embed)
    assert g > 0.0
    assert g < 10.0, "linear single-weight classifier should have small grad norm"


def test_estimate_constants_returns_finite(tmp_path):
    env = _build_env(tmp_path)
    L_hat, M_hat = estimate_constants(env, n_samples=16)
    assert L_hat >= 0.0
    assert M_hat == L_hat


def test_estimate_constants_more_samples_max_nondecreasing(tmp_path):
    env_small = _build_env(tmp_path)
    L_small, _ = estimate_constants(env_small, n_samples=8)
    env_big = _build_env(tmp_path)
    L_big, _ = estimate_constants(env_big, n_samples=64)
    assert L_big >= L_small - 1e-6, "max over a larger sample cannot be lower"
