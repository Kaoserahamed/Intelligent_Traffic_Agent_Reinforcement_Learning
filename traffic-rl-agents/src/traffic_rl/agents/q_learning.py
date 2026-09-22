"""Tabular Q-learning agent.

State is discretised and the Q-table is persisted as JSON (no pickle) for
security.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from traffic_rl.agents.base import Action, BaseAgent, State
from traffic_rl.config import ACTION_SIZE
from traffic_rl.io_utils import load_q_table, save_q_table
from traffic_rl.logging import get_logger

log = get_logger("traffic_rl.agents.q_learning")


class QLearningAgent(BaseAgent):
    """Tabular Q-learning with discrete state binning."""

    def __init__(self, config: Any,
                 state_bins: int | tuple[int, ...] = 5) -> None:
        self._name = config.name
        self._action_size = ACTION_SIZE
        self._epsilon = config.epsilon if config.epsilon is not None else 0.1
        self._alpha = float(config.hyperparams.get("alpha", 0.1))
        self._gamma = float(config.hyperparams.get("gamma", 0.95))
        self._epsilon_decay = float(config.hyperparams.get("epsilon_decay", 0.995))
        self._epsilon_min = float(config.hyperparams.get("epsilon_min", 0.01))
        self._state_bins = state_bins
        self._q_table: dict[tuple[int, ...], list[float]] = {}
        self._t = 0

    @property
    def name(self) -> str:
        return self._name

    def act(self, state: State, epsilon: float = 0.0) -> Action:
        if state.size == 0:
            return np.random.randint(0, self._action_size)
        discretised = self._discretise(state)
        if discretised not in self._q_table:
            self._q_table[discretised] = [0.0] * self._action_size
        if np.random.random() < epsilon:
            return np.random.randint(0, self._action_size)
        values = self._q_table[discretised]
        return int(np.argmax(values))

    def learn(self, state: State, action: Action, reward: float,
              next_state: State, done: bool) -> None:
        if state.size == 0 or next_state.size == 0:
            return
        s = self._discretise(state)
        s_next = self._discretise(next_state)
        if s not in self._q_table:
            self._q_table[s] = [0.0] * self._action_size
        if s_next not in self._q_table:
            self._q_table[s_next] = [0.0] * self._action_size

        q_current = self._q_table[s][action]
        if done:
            q_target = reward
        else:
            q_target = reward + self._gamma * max(self._q_table[s_next])
        self._q_table[s][action] = q_current + self._alpha * (q_target - q_current)
        self._t += 1

    def decay_epsilon(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def current_epsilon(self) -> float:
        return self._epsilon

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_q_table(self._q_table, path,
                     metadata={"alpha": self._alpha, "gamma": self._gamma,
                              "t": self._t})

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pkl":
            from traffic_rl.io_utils import load_legacy_q_table_pickle
            self._q_table = load_legacy_q_table_pickle(path)
        else:
            self._q_table = load_q_table(path)
        log.info("loaded Q-table (%d states) from %s", len(self._q_table), path)

    def _discretise(self, state: State) -> tuple[int, ...]:
        bins: tuple[int, ...]
        if isinstance(self._state_bins, int):
            bins = tuple([self._state_bins] * state.shape[0])
        else:
            bins = self._state_bins
        indices: list[int] = []
        for i, (val, n) in enumerate(zip(state, bins)):
            idx = min(int(val * n), n - 1)
            indices.append(idx)
        return tuple(indices)
