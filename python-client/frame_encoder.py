from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class FrameStackFeatureExtractor(BaseFeaturesExtractor):
    """Shared per-frame encoder for flat stacked vector observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        frame_stack: int = 16,
        per_frame_features: int = 64,
        stacked_obs_dim: int | None = None,
        extra_features_dim: int = 0,
    ) -> None:
        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        self.extra_features_dim = max(0, int(extra_features_dim))
        self.stacked_obs_dim = int(stacked_obs_dim) if stacked_obs_dim else obs_dim - self.extra_features_dim
        if self.stacked_obs_dim <= 0 or self.stacked_obs_dim > obs_dim:
            self.stacked_obs_dim = obs_dim
            self.extra_features_dim = 0

        self.frame_stack = frame_stack if frame_stack > 1 and self.stacked_obs_dim % frame_stack == 0 else 1
        self.frame_size = self.stacked_obs_dim // self.frame_stack
        self.per_frame_features = per_frame_features

        self.frame_encoder = nn.Sequential(
            nn.Linear(self.frame_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, per_frame_features),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(per_frame_features * self.frame_stack, 512),
            nn.ReLU(),
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_features_dim, 128),
            nn.ReLU(),
        ) if self.extra_features_dim > 0 else None
        output_input_dim = 512 + (128 if self.extra_features_dim > 0 else 0)
        self.output_encoder = nn.Sequential(
            nn.Linear(output_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        stacked = observations[:, :self.stacked_obs_dim]
        frames = stacked.reshape(batch_size, self.frame_stack, self.frame_size)
        encoded = self.frame_encoder(frames.reshape(batch_size * self.frame_stack, self.frame_size))
        encoded = encoded.reshape(batch_size, self.frame_stack * self.per_frame_features)
        temporal = self.temporal_encoder(encoded)
        if self.extra_encoder is not None:
            extra = observations[:, self.stacked_obs_dim:self.stacked_obs_dim + self.extra_features_dim]
            temporal = torch.cat([temporal, self.extra_encoder(extra)], dim=1)
        return self.output_encoder(temporal)
