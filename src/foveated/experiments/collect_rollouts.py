"""Random-policy rollouts to collect (z_t, a_t, z_{t+1}) triples.

Used by `algos.dynamics_train` to pretrain the dynamics model on clean
ImageNet before any PPO agent runs. Pretraining on a stationary (random)
policy keeps the dynamics model from drifting under PPO's non-stationary
state distribution.
"""
from __future__ import annotations

import numpy as np
import torch

from foveated.envs.foveated_env import FoveatedClassificationEnv


@torch.no_grad()
def collect_triples(
    env: FoveatedClassificationEnv,
    n_episodes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_t_buf: list[np.ndarray] = []
    a_t_buf: list[int] = []
    z_next_buf: list[np.ndarray] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        prev_z = obs[: env.cfg.embed_dim].copy()
        done = False
        truncated = False
        while not (done or truncated):
            a = int(rng.integers(env.n_patches))
            obs, _, done, truncated, _ = env.step(a)
            cur_z = obs[: env.cfg.embed_dim].copy()
            z_t_buf.append(prev_z)
            a_t_buf.append(a)
            z_next_buf.append(cur_z)
            prev_z = cur_z

    return (
        np.stack(z_t_buf).astype(np.float32),
        np.asarray(a_t_buf, dtype=np.int64),
        np.stack(z_next_buf).astype(np.float32),
    )
