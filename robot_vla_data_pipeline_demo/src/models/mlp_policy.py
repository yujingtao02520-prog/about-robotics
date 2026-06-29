"""MLP state-to-action baseline policy."""

from __future__ import annotations

from torch import nn


class MLPPolicy(nn.Module):
    """Predict action vectors from robot state vectors."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state):
        """Run a forward pass."""
        return self.net(state)
