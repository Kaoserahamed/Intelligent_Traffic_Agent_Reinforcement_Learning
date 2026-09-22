"""Base classes for RL agents.

Every agent implements the :class:`BaseAgent` interface so the training loop
and CLI can treat Q-Learning, DQN, Double DQN and PPO uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from traffic_rl.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

log = get_logger("traffic_rl.agents")

State = np.ndarray  # shape (STATE_SIZE,) float32
Action = int


class BaseAgent(ABC):
    """Contract every learning agent must fulfil."""

    @abstractmethod
    def act(self, state: State, epsilon: float = 0.0) -> Action:
        """Choose an action given the current state (epsilon-greedy for RL)."""
        ...

    @abstractmethod
    def learn(self, state: State, action: Action, reward: float,
              next_state: State, done: bool) -> None:
        """Update the agent's internal model from a single transition."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the agent's learned parameters to *path*."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore the agent's learned parameters from *path*."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name (used in logs and plots)."""
        ...


class NeuralAgent(BaseAgent, ABC):
    """Base for agents that use a PyTorch neural network."""

    def __init__(self, config: Any) -> None:
        self._name = config.name
        from traffic_rl.utils.torch import default_device
        self._device = default_device()

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self):
        return self._device
