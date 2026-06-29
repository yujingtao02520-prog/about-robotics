"""CNN + MLP RGB/state-to-action baseline policy."""

from __future__ import annotations

import torch
from torch import nn


class CNNPolicy(nn.Module):
    """Encode RGB images and state vectors before predicting actions."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, image, state):
        """Run a forward pass with image tensor [B,3,H,W] and state [B,D]."""
        image_feature = self.image_encoder(image)
        state_feature = self.state_encoder(state)
        return self.head(torch.cat([image_feature, state_feature], dim=-1))
