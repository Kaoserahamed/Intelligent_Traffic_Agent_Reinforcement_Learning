"""Policy/value network for PPO."""

import torch
import torch.nn as nn


class _PolicyValueNet(nn.Module):
    """Shared backbone with separate policy and value heads."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        shared = self.shared(x)
        return self.policy_head(shared), self.value_head(shared)
