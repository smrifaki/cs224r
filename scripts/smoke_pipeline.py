"""End-to-end CPU smoke pipeline.

Exercises the full project pipeline on synthetic data without
touching ImageNet or requiring a GPU. Verifies the pieces compose:
env constructs, dynamics trains, Laplace fits, wrappers apply,
evaluation runs, aggregation pools across seeds.

Not a benchmark; the numbers will be nonsense. The point is that
the code path runs end-to-end without crashing. If this script
returns 0 in less than a minute on CPU the pipeline is structurally
sound.

Run from the project root:

    python scripts/smoke_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import foveated.envs.foveated_env as fe
from foveated.algos.dynamics_train import train_dynamics
from foveated.algos.stats import (
    bootstrap_ci,
    cliffs_delta,
    paired_permutation_test,
)
from foveated.envs.feature_wrappers import (
    ClassifierEntropyObsWrapper,
    ProspectiveUncertaintyObsWrapper,
)
from foveated.envs.foveated_env import (
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)
from foveated.models.dynamics_bayesian import fit_last_layer_laplace


def ok(label: str) -> None:
    print(f"  [OK] {label}")


def fatal(label: str, exc: BaseException) -> None:
    print(f"  [FAIL] {label}: {exc}")
    traceback.print_exc()
    sys.exit(1)


@dataclass
class MockBackbone:
    embed_dim: int = 32
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


def main() -> None:
    print("=" * 50)
    print("smoke_pipeline.py")
    print("=" * 50)

    tmp = Path(tempfile.mkdtemp(prefix="foveated_smoke_"))
    print(f"  using tmp = {tmp}")

    print("\n[1/6] env construction")

    def _mock_load(_p, _cfg):
        return torch.zeros(3, _cfg.image_size, _cfg.image_size)

    fe.load_and_corrupt = _mock_load  # type: ignore[assignment]
    paths = [tmp / f"img_{i}.jpeg" for i in range(8)]
    cfg = FoveatedEnvConfig(
        embed_dim=32, grid_h=4, grid_w=4, max_patches=4, backbone_name="mock",
    )
    try:
        env = FoveatedClassificationEnv(
            cfg=cfg, image_paths=paths, labels=[0] * 8,
            backbone=MockBackbone(), device="cpu",
        )
        env.reset()
    except Exception as e:
        fatal("env construction", e)
        return  # for pyright
    ok("FoveatedClassificationEnv")

    print("\n[2/6] dynamics training (synthetic, 2 epochs)")
    n = 64
    rng = np.random.default_rng(0)
    z_t = rng.normal(0, 1, size=(n, 32)).astype(np.float32)
    a_t = rng.integers(0, env.n_patches, size=n).astype(np.int64)
    z_next = z_t + rng.normal(0, 0.1, size=(n, 32)).astype(np.float32)
    try:
        model = train_dynamics(
            z_t, a_t, z_next,
            n_actions=env.n_patches, embed_dim=32,
            epochs=2, batch_size=16, lr=1e-3, device="cpu",
        )
    except Exception as e:
        fatal("dynamics training", e)
        return
    ok("ForwardDynamics trained")

    print("\n[3/6] Laplace posterior fit")
    z_t_t = torch.from_numpy(z_t).float()
    a_t_t = torch.from_numpy(a_t).long()
    try:
        posterior = fit_last_layer_laplace(model, z_t_t, a_t_t, prior_precision=1.0)
    except Exception as e:
        fatal("Laplace fit", e)
        return
    assert posterior.posterior_cov.shape == (32, 512, 512)
    ok("LaplacePosterior")

    print("\n[4/6] feature wrappers")
    try:
        env_c = ProspectiveUncertaintyObsWrapper(env, dynamics=model, posterior=posterior, device="cpu")
        obs_c, _ = env_c.reset()
        assert np.isfinite(obs_c).all()
        ok("ProspectiveUncertaintyObsWrapper (Laplace)")

        env2 = FoveatedClassificationEnv(
            cfg=cfg, image_paths=paths, labels=[0] * 8,
            backbone=MockBackbone(), device="cpu",
        )
        env_d = ClassifierEntropyObsWrapper(env2, device="cpu")
        obs_d, _ = env_d.reset()
        assert np.isfinite(obs_d).all()
        ok("ClassifierEntropyObsWrapper")
    except Exception as e:
        fatal("wrappers", e)
        return

    print("\n[5/6] stats primitives")
    try:
        rng2 = np.random.default_rng(1)
        a = rng2.normal(0.6, 1, size=200)
        b = rng2.normal(0.0, 1, size=200)
        _, lo, hi = bootstrap_ci(a, rng=rng2)
        assert lo < hi
        _, p = paired_permutation_test(a, b, n_perm=500, rng=rng2)
        assert 0.0 <= p <= 1.0
        delta = cliffs_delta(a, b)
        assert -1.0 <= delta <= 1.0
    except Exception as e:
        fatal("stats", e)
        return
    ok("stats primitives")

    print("\n[6/6] full episode rollout with wrappers")
    try:
        obs, _ = env_c.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action = int(np.argmax(obs[-env_c.n_actions:]))
            obs, _, done, truncated, _ = env_c.step(action)
            steps += 1
            if steps > 20:
                break
    except Exception as e:
        fatal("full rollout", e)
        return
    assert steps <= env.cfg.max_patches + 1
    ok(f"rolled out {steps} steps with prospective uncertainty wrapper")

    print("\nall stages OK")


if __name__ == "__main__":
    main()
