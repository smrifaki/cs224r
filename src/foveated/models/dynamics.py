"""Forward dynamics on backbone embeddings with prospective uncertainty.

Two signals come out of this module:
    * Retrospective:  e_t = (z_obs - z_hat) * exp(-log_s / 2), computed AFTER
                      committing patch a_{t-1}. Used as the dynamics-model
                      training signal and as the Pathak-style intrinsic reward
                      for Agent B.
    * Prospective:    u_t(a) = mean(log_s(z_t, a)) over the embedding axis,
                      available at decision time WITHOUT committing the patch.
                      This is the signal Agent C exposes to the policy. It
                      resolves the deployment-time question the TA flagged at
                      proposal review.

Predictive coding form:
    (z_hat, log_s) = f_phi(z_t, a_t)
    L_NLL = mean[ (z_{t+1} - z_hat)^2 / exp(log_s) + log_s ]
    e = (z_obs - z_hat) * exp(-log_s / 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardDynamics(nn.Module):
    LOG_VAR_MIN: float = -3.0
    LOG_VAR_MAX: float = 3.0

    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_actions = n_actions
        in_dim = embed_dim + n_actions
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU()]
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, embed_dim)
        self.logvar_head = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self, z_t: torch.Tensor, a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_onehot = F.one_hot(a_t, num_classes=self.n_actions).float()
        x = torch.cat([z_t, a_onehot], dim=-1)
        h = self.trunk(x)
        z_hat = self.mean_head(h)
        log_s = self.logvar_head(h).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        return z_hat, log_s

    @torch.no_grad()
    def query_all_actions(
        self, z_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sweep every action and return (z_hat, log_s) for each.

        z_t:    (batch, embed_dim)
        return: (z_hat_all, log_s_all) each (batch, n_actions, embed_dim).

        Used at decision time to build a per-patch uncertainty map without
        committing any patch.
        """
        b = z_t.shape[0]
        n = self.n_actions
        z_rep = z_t.unsqueeze(1).expand(b, n, self.embed_dim).reshape(b * n, -1)
        a_rep = torch.arange(n, device=z_t.device).repeat(b)
        z_hat, log_s = self(z_rep, a_rep)
        return z_hat.view(b, n, -1), log_s.view(b, n, -1)

    @torch.no_grad()
    def prospective_uncertainty(
        self, z_t: torch.Tensor, committed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-action expected log-variance summary, available at decision time."""
        _, log_s_all = self.query_all_actions(z_t)
        score = log_s_all.mean(dim=-1)
        if committed_mask is not None:
            score = score.masked_fill(committed_mask.bool(), -1e4)
        return score

    def nll_loss(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next: torch.Tensor,
        action_dropout_p: float = 0.5,
    ) -> torch.Tensor:
        """Gaussian NLL with action dropout and anti-collapse term."""
        a_onehot = F.one_hot(a_t, num_classes=self.n_actions).float()
        if self.training and action_dropout_p > 0:
            keep = (torch.rand(a_t.shape[0], 1, device=a_t.device) > action_dropout_p).float()
            a_onehot = a_onehot * keep
        x = torch.cat([z_t, a_onehot], dim=-1)
        h = self.trunk(x)
        z_hat = self.mean_head(h)
        log_s = self.logvar_head(h).clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        nll = ((z_next - z_hat) ** 2 / log_s.exp() + log_s).mean()
        anti_collapse = 1e-3 * (z_hat - z_t).pow(2).mean()
        return nll - anti_collapse
