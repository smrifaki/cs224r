"""Foveated ImageNet / ImageNet-C Gym environment.

Episode = one image. At each step the agent commits one high-res patch from a
7x7 grid. Episode terminates on patch-budget exhaustion, on a re-pick (treated
as give-up), or on the optional stop action. Terminal reward = top-1 accuracy
of the backbone on the assembled (low-res + committed) representation, minus
the total per-patch cost paid so far.

Corruption is applied at the pixel level via the `imagenet_c` package, with a
Gaussian-noise fallback if the package is not installed.

Phase 1 spec is the canonical reference; this implementation tracks it
closely and supports the Phase 3 / Phase 4 additions for Agents B, C, D.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image

try:
    from imagenet_c import corrupt as imagenet_c_corrupt  # type: ignore[import-not-found]
    HAS_IMAGENET_C = True
except ImportError:
    imagenet_c_corrupt = None  # type: ignore[assignment]
    HAS_IMAGENET_C = False

try:
    _BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:
    _BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CORRUPTION_NAMES = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


@dataclass
class FoveatedEnvConfig:
    image_size: int = 224
    low_res_size: int = 56
    grid_h: int = 7
    grid_w: int = 7
    max_patches: int = 8  # K=8 horizon per phase1 spec
    patch_cost: float = 0.02
    backbone_name: str = "vit_small_patch16_224"
    embed_dim: int = 384
    n_classes: int = 1000
    corruption_type: str | None = None
    corruption_severity: int = 0
    seed: int = 0
    allow_stop_action: bool = False
    goal_corruption_slots: tuple[str, ...] = field(default_factory=lambda: CORRUPTION_NAMES)


def _to_tensor_normed(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _fallback_noise_corruption(arr: np.ndarray, severity: int) -> np.ndarray:
    if severity <= 0:
        return arr
    sigma = (0.04, 0.06, 0.08, 0.09, 0.10)[severity - 1]
    noise = np.random.normal(0, sigma * 255.0, size=arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def load_and_corrupt(path: Path, cfg: FoveatedEnvConfig) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(
        (cfg.image_size, cfg.image_size), _BILINEAR
    )
    arr = np.asarray(img)
    if cfg.corruption_type and cfg.corruption_severity > 0:
        if HAS_IMAGENET_C and imagenet_c_corrupt is not None:
            arr = imagenet_c_corrupt(
                arr, severity=cfg.corruption_severity, corruption_name=cfg.corruption_type
            )
        else:
            arr = _fallback_noise_corruption(arr, cfg.corruption_severity)
    return _to_tensor_normed(arr)


class BackboneAdapter:
    """Wraps a frozen backbone to expose (low-res, per-patch, classify).

    Constructor takes a timm-style module exposing forward_features and head.
    ViT vs CNN handling is selected by `is_vit`.
    """

    def __init__(self, backbone: torch.nn.Module, is_vit: bool, device: str):
        self.backbone = backbone.to(device).eval()
        self.is_vit = is_vit
        self.device = device

    @torch.no_grad()
    def _feature_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward_features = cast(Any, self.backbone).forward_features
        feats: torch.Tensor = forward_features(x.unsqueeze(0).to(self.device))
        if self.is_vit:
            cls = feats[:, 0]
            tokens = feats[:, 1:]
            n = int(tokens.shape[1])
            side = int(n ** 0.5)
            assert side * side == n, "non-square ViT grid"
            return tokens.permute(0, 2, 1).reshape(1, -1, side, side), cls
        return feats, feats.mean(dim=(2, 3))

    @torch.no_grad()
    def low_res(self, image: torch.Tensor, low_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        small = F.interpolate(
            image.unsqueeze(0), size=low_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        fmap, pooled = self._feature_map(small)
        return fmap.squeeze(0), pooled.squeeze(0)

    @torch.no_grad()
    def patch_features(
        self, image: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        fmap, _ = self._feature_map(image)
        return F.adaptive_avg_pool2d(fmap, (grid_h, grid_w)).squeeze(0)

    @torch.no_grad()
    def classify(self, pooled: torch.Tensor) -> torch.Tensor:
        head = cast(Any, self.backbone).head
        return head(pooled.unsqueeze(0)).squeeze(0)


class FoveatedClassificationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: FoveatedEnvConfig,
        image_paths: list[Path],
        labels: list[int],
        backbone: BackboneAdapter,
        device: str = "cuda",
    ):
        super().__init__()
        self.cfg = cfg
        self.image_paths = image_paths
        self.labels = labels
        self.backbone = backbone
        self.device = device

        self.n_patches = cfg.grid_h * cfg.grid_w
        self.n_actions = self.n_patches + (1 if cfg.allow_stop_action else 0)
        self.stop_action = self.n_patches if cfg.allow_stop_action else None

        self.goal_dim = len(cfg.goal_corruption_slots) + 1  # +1 for clean slot
        obs_dim = cfg.embed_dim + self.n_patches + self.goal_dim
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.rng = np.random.default_rng(cfg.seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.committed_mask = np.zeros(self.n_patches, dtype=np.float32)
        self.committed_patches: list[int] = []
        self.current_label: int = 0
        self.goal_one_hot = np.zeros(self.goal_dim, dtype=np.float32)
        self.low_res_embed = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        self.patch_embeds = np.zeros(
            (self.n_patches, self.cfg.embed_dim), dtype=np.float32
        )
        self.step_count = 0

    def _goal_slot(self) -> int:
        if self.cfg.corruption_type is None or self.cfg.corruption_severity == 0:
            return 0
        if self.cfg.corruption_type not in self.cfg.goal_corruption_slots:
            return 0
        return 1 + self.cfg.goal_corruption_slots.index(self.cfg.corruption_type)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()

        idx = int(self.rng.integers(len(self.image_paths)))
        image = load_and_corrupt(self.image_paths[idx], self.cfg)
        self.current_label = self.labels[idx]

        _, low_pooled = self.backbone.low_res(image, self.cfg.low_res_size)
        self.low_res_embed = low_pooled.cpu().numpy().astype(np.float32)

        patch_feats = self.backbone.patch_features(
            image, self.cfg.grid_h, self.cfg.grid_w
        )
        flat = patch_feats.permute(1, 2, 0).reshape(self.n_patches, -1)
        if flat.shape[1] != self.cfg.embed_dim:
            flat = F.adaptive_avg_pool1d(
                flat.unsqueeze(0).transpose(1, 2), self.cfg.embed_dim
            ).transpose(1, 2).squeeze(0)
        self.patch_embeds = flat.cpu().numpy().astype(np.float32)

        slot = self._goal_slot()
        self.goal_one_hot[slot] = 1.0
        return self._build_obs(), {"goal_slot": slot}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert 0 <= action < self.n_actions
        self.step_count += 1
        reward = 0.0
        terminated = False

        if (self.cfg.allow_stop_action and action == self.stop_action) or self.committed_mask[action] == 1.0:
            terminated = True
        else:
            self.committed_mask[action] = 1.0
            self.committed_patches.append(action)
            reward -= self.cfg.patch_cost

        truncated = self.step_count >= self.cfg.max_patches

        if terminated or truncated:
            acc = self._classify_accuracy()
            reward += acc

        return self._build_obs(), reward, terminated, truncated, {
            "committed_patches": list(self.committed_patches),
            "step_count": self.step_count,
        }

    def _build_obs(self) -> np.ndarray:
        return np.concatenate(
            [self.low_res_embed, self.committed_mask, self.goal_one_hot]
        ).astype(np.float32)

    def _classify_accuracy(self) -> float:
        if self.committed_patches:
            high = self.patch_embeds[self.committed_patches].mean(axis=0)
            embed = 0.5 * (self.low_res_embed + high)
        else:
            embed = self.low_res_embed
        logits = self.backbone.classify(
            torch.from_numpy(embed).float().to(self.device)
        )
        return float(int(logits.argmax().item()) == self.current_label)

    def current_state(self) -> dict[str, Any]:
        return {
            "z_t": self.low_res_embed.copy(),
            "committed_mask": self.committed_mask.copy(),
            "committed_patches": list(self.committed_patches),
            "patch_embeds": self.patch_embeds.copy(),
            "label": self.current_label,
        }
