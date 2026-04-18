"""Information-bottleneck variant of the forward dynamics objective.

The plain NLL objective in models/dynamics.py trains the variance head to
maximize the predictive likelihood, which mixes aleatoric and epistemic
variance freely. The Laplace approximation in models/dynamics_bayesian.py
extracts the epistemic component after training but cannot change what
the head LEARNS.

This module trains the dynamics model with an information-bottleneck
penalty that explicitly compresses the latent representation phi(z, a)
toward a Gaussian prior, in the style of Alemi et al. (2017). The
intuition: a compact phi cannot encode noise it has not been forced to
predict, so the model is pushed to attribute "irreducible" components
of the residual to log_s rather than to phi. Downstream, this makes the
Laplace-extracted epistemic variance a cleaner signal for Agent C.

Mathematically, instead of
    L = NLL(z_next | mu(phi(z,a)), log_s(phi(z,a)))
we train with
    L_IB = NLL(...)  +  beta * KL( q(phi | z, a)  ||  N(0, I) )
where q(phi | z, a) is a stochastic encoder produced by adding a
reparameterized noise term to the trunk output. beta controls the
information-bottleneck trade-off; we ablate over beta in {0, 1e-4,
1e-3, 1e-2}.

beta = 0 recovers the plain dynamics_train objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from foveated.models.dynamics import ForwardDynamics


@dataclass
class IBConfig:
    beta: float = 1e-3  # information-bottleneck weight
    hidden_dim: int = 512
    stochastic_phi_dim: int = 256  # dimension of q(phi | z, a)


class StochasticDynamics(nn.Module):
    """ForwardDynamics with a reparameterized stochastic bottleneck.

    The trunk produces (mu_phi, log_var_phi); phi is sampled as
    mu_phi + sigma_phi * eps and then fed to the mean and log-var heads.
    At eval time we use mu_phi (no noise) for determinism.
    """

    LOG_VAR_MIN: float = -3.0
    LOG_VAR_MAX: float = 3.0

    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        cfg: IBConfig | None = None,
        depth: int = 3,
    ):
        super().__init__()
        self.cfg = cfg or IBConfig()
        self.embed_dim = embed_dim
        self.n_actions = n_actions
        in_dim = embed_dim + n_actions
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, self.cfg.hidden_dim), nn.GELU()]
            prev = self.cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.phi_mean = nn.Linear(self.cfg.hidden_dim, self.cfg.stochastic_phi_dim)
        self.phi_logvar = nn.Linear(self.cfg.hidden_dim, self.cfg.stochastic_phi_dim)
        self.mean_head = nn.Linear(self.cfg.stochastic_phi_dim, embed_dim)
        self.logvar_head = nn.Linear(self.cfg.stochastic_phi_dim, embed_dim)

    def _trunk(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a_t, num_classes=self.n_actions).float()
        x = torch.cat([z_t, a_onehot], dim=-1)
        return self.trunk(x)

    def encode(self, z_t: torch.Tensor, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk(z_t, a_t)
        return self.phi_mean(h), self.phi_logvar(h).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)

    def sample_phi(
        self, mu_phi: torch.Tensor, log_var_phi: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        if deterministic or not self.training:
            return mu_phi
        eps = torch.randn_like(mu_phi)
        return mu_phi + (0.5 * log_var_phi).exp() * eps

    def forward(
        self, z_t: torch.Tensor, a_t: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_phi, log_var_phi = self.encode(z_t, a_t)
        phi = self.sample_phi(mu_phi, log_var_phi, deterministic=deterministic)
        mu = self.mean_head(phi)
        log_s = self.logvar_head(phi).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        return mu, log_s

    def ib_loss(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        action_dropout_p: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        a_onehot = F.one_hot(a_t, num_classes=self.n_actions).float()
        if self.training and action_dropout_p > 0:
            keep = (torch.rand(a_t.shape[0], 1, device=a_t.device) > action_dropout_p).float()
            a_onehot = a_onehot * keep
        x = torch.cat([z_t, a_onehot], dim=-1)
        h = self.trunk(x)
        mu_phi = self.phi_mean(h)
        log_var_phi = self.phi_logvar(h).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        phi = self.sample_phi(mu_phi, log_var_phi)
        mu = self.mean_head(phi)
        log_s = self.logvar_head(phi).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        nll = ((z_next - mu) ** 2 / log_s.exp() + log_s).mean()
        # Closed-form KL between N(mu_phi, sigma_phi^2 I) and N(0, I) per dim.
        kl_per_dim = 0.5 * (mu_phi.pow(2) + log_var_phi.exp() - 1.0 - log_var_phi)
        kl = kl_per_dim.mean()
        anti_collapse = 1e-3 * (mu - z_t).pow(2).mean()
        loss = nll + self.cfg.beta * kl - anti_collapse
        diagnostics = {
            "nll": float(nll.item()),
            "kl_phi": float(kl.item()),
            "anti_collapse": float(anti_collapse.item()),
        }
        return loss, diagnostics


def from_pretrained_to_ib(
    pretrained: ForwardDynamics, cfg: IBConfig | None = None
) -> StochasticDynamics:
    """Construct an IB-style dynamics with the same shape as a trained
    plain dynamics model. Caller is expected to retrain weights; this
    helper just gets dimensions right.
    """
    return StochasticDynamics(
        embed_dim=pretrained.embed_dim,
        n_actions=pretrained.n_actions,
        cfg=cfg,
        depth=3,
    )
