"""A shared, framework-agnostic experience replay buffer."""

from __future__ import annotations

import random
from collections import deque

import numpy as np

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-capacity ring buffer of ``(s, a, r, s', done)`` transitions.

    Used by the value-based agents (DQN / Double DQN).  Sampling returns
    batched ``numpy`` arrays so it can be consumed identically by both agents.
    """

    def __init__(self, capacity: int = 10000) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((np.asarray(state), action, float(reward),
                            np.asarray(next_state), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer length ({len(self.buffer)})"
            )
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        return bool(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
