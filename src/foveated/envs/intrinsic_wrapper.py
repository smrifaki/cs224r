"""Agent B's intrinsic-reward wrapper around the foveated env.

Adds beta * ||precision-weighted residual||^2 to the env reward, where the
residual is computed retrospectively from the previous step's commit. beta
anneals linearly to keep intrinsic from dominating extrinsic at training
end. See `algos.intrinsic_reward` for the bonus computation; this wrapper
only does the bookkeeping and step-by-step beta schedule.
"""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import torch

from foveated.algos.intrinsic_reward import intrinsic_reward
from foveated.envs.foveated_env import FoveatedClassificationEnv
from foveated.models.dynamics import ForwardDynamics


class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dynamics: ForwardDynamics,
        beta_start: float = 0.01,
        beta_end: float = 0.001,
        anneal_steps: int = 100_000,
        device: str = "cuda",
    ):
        super().__init__(env)
        self.dyn = dynamics.eval()
        self.device = device
        b = cast(FoveatedClassificationEnv, env.unwrapped)
        self.embed_dim = b.cfg.embed_dim
        self._n_patches = b.n_patches
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.global_step = 0
        self._last_z: torch.Tensor | None = None
        self._last_a: int | None = None

    def _beta(self) -> float:
        frac = min(self.global_step / self.anneal_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.global_step += 1
        if self._last_z is not None and self._last_a is not None:
            with torch.no_grad():
                z_obs = torch.from_numpy(obs[: self.embed_dim]).float().unsqueeze(0).to(self.device)
                z_hat, log_s = self.dyn(
                    self._last_z,
                    torch.tensor([self._last_a], device=self.device),
                )
                bonus = float(
                    intrinsic_reward(z_obs, z_hat, log_s, beta=self._beta()).item()
                )
                rew = float(rew) + bonus
                info["intrinsic_reward"] = bonus
        self._last_z = (
            torch.from_numpy(obs[: self.embed_dim]).float().unsqueeze(0).to(self.device)
        )
        self._last_a = int(action) if action < self._n_patches else None
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_z = None
        self._last_a = None
        return obs, info
