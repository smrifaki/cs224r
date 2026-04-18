"""Observation feature wrappers for Agents C and D.

Agent C: ProspectiveUncertaintyObsWrapper appends per-action predicted log-
variance u(a) from the dynamics model to the observation. The signal is
available at decision time WITHOUT committing the patch , the dynamics
model is queried for all candidate actions in a single batched forward
pass. This is the design refinement that resolves the TA's deployment-time
question. See docs/prospective_uncertainty_design.md.

Optionally, when constructed with a Laplace posterior, the wrapper exposes
the BALD-aligned EPISTEMIC log-variance instead of the total log-variance.
This is the BALD-correct feature for the bound in docs/regret_bound.md.

Agent D: ClassifierEntropyObsWrapper appends the entropy of the backbone
classifier's predicted distribution on the current assembled embedding.
Standard uncertainty baseline. The TA recommended this comparison at
proposal review. See docs/agent_d_entropy_baseline.md.
"""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np
import torch

from foveated.envs.foveated_env import FoveatedClassificationEnv
from foveated.models.dynamics import ForwardDynamics
from foveated.models.dynamics_bayesian import (
    LaplacePosterior,
    prospective_epistemic_uncertainty,
)


def _committed_mask_from_obs(obs: np.ndarray, embed_dim: int, n_patches: int) -> np.ndarray:
    return obs[embed_dim : embed_dim + n_patches]


class ProspectiveUncertaintyObsWrapper(gym.ObservationWrapper):
    """Append per-action prospective log-variance.

    If `posterior` is provided, the exposed signal is the EPISTEMIC log-
    variance under the last-layer Laplace approximation (BALD-aligned).
    Otherwise the signal is the total predictive log-variance straight
    from the dynamics model's logvar_head (aleatoric + epistemic mixed).
    The Laplace path is the recommended one; see docs/regret_bound.md.
    """

    def __init__(
        self,
        env: gym.Env,
        dynamics: ForwardDynamics,
        posterior: LaplacePosterior | None = None,
        device: str = "cuda",
    ):
        super().__init__(env)
        self.dyn = dynamics.eval()
        self.posterior = posterior
        self.device = device
        b = cast(FoveatedClassificationEnv, env.unwrapped)
        self.embed_dim = b.cfg.embed_dim
        self.n_patches = b.n_patches
        self.n_actions = b.n_actions

        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        new_dim = int(base.shape[0]) + self.n_actions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        z_t = torch.from_numpy(observation[: self.embed_dim]).float().unsqueeze(0).to(self.device)
        committed = _committed_mask_from_obs(observation, self.embed_dim, self.n_patches)
        mask = torch.from_numpy(committed).float().unsqueeze(0).to(self.device)
        if self.posterior is not None:
            u_patches = prospective_epistemic_uncertainty(
                self.dyn, self.posterior, z_t, committed_mask=mask
            ).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            u_patches = self.dyn.prospective_uncertainty(
                z_t, committed_mask=mask
            ).squeeze(0).cpu().numpy().astype(np.float32)
        if self.n_actions > self.n_patches:
            u = np.concatenate([u_patches, np.zeros(self.n_actions - self.n_patches, dtype=np.float32)])
        else:
            u = u_patches
        return np.concatenate([observation, u]).astype(np.float32)


class ClassifierEntropyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.device = device
        b = cast(FoveatedClassificationEnv, env.unwrapped)
        self.embed_dim = b.cfg.embed_dim
        self.n_patches = b.n_patches

        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        new_dim = int(base.shape[0]) + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        b = cast(FoveatedClassificationEnv, self.env.unwrapped)
        snap = b.current_state()
        if snap["committed_patches"]:
            high = snap["patch_embeds"][snap["committed_patches"]].mean(axis=0)
            embed = 0.5 * (snap["z_t"] + high)
        else:
            embed = snap["z_t"]
        with torch.no_grad():
            logits = b.backbone.classify(
                torch.from_numpy(embed).float().to(self.device)
            )
            log_p = torch.log_softmax(logits, dim=-1)
            entropy = float(-(log_p.exp() * log_p).sum().item())
        return np.concatenate([observation, np.array([entropy], dtype=np.float32)]).astype(np.float32)
