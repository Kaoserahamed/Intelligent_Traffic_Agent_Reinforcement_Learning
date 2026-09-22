"""Deep Q-Network agent with experience replay and target network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - conventional torch alias
import torch.optim as optim

from traffic_rl.agents.base import NeuralAgent
from traffic_rl.config import ACTION_SIZE, STATE_SIZE
from traffic_rl.logging import get_logger
from traffic_rl.utils.replay import ReplayBuffer

log = get_logger("traffic_rl.agents.dqn")


class DQNAgent(NeuralAgent):
    """Deep Q-Network with experience replay and target network."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._action_size = ACTION_SIZE
        self._state_size = STATE_SIZE
        self._lr = float(config.hyperparams.get("lr", 0.001))
        self._gamma = float(config.hyperparams.get("gamma", 0.95))
        self._epsilon = config.epsilon if config.epsilon is not None else 0.1
        self._epsilon_decay = float(config.hyperparams.get("epsilon_decay", 0.995))
        self._epsilon_min = float(config.hyperparams.get("epsilon_min", 0.01))
        self._batch_size = int(config.hyperparams.get("batch_size", 64))
        self._memory_size = int(config.hyperparams.get("memory_size", 10000))
        self._target_update = int(config.hyperparams.get("target_update", 100))

        self._memory = ReplayBuffer(capacity=self._memory_size)
        self._model = self._build_model()
        self._target_model = self._build_model()
        self._target_model.load_state_dict(self._model.state_dict())
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._t = 0

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self._state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._action_size),
        ).to(self._device)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if state.size == 0:
            return np.random.randint(0, self._action_size)
        if np.random.random() < epsilon:
            return np.random.randint(0, self._action_size)
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
            q_values = self._model(state_t)
            return int(q_values.argmax(dim=1).item())

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        self._memory.push(state, action, reward, next_state, done)
        if len(self._memory) < self._batch_size:
            return
        self._train_step()

    def _train_step(self) -> None:
        states, actions, rewards, next_states, dones = self._memory.sample(self._batch_size)
        states_t = torch.from_numpy(states).float().to(self._device)
        actions_t = torch.from_numpy(actions).long().to(self._device)
        rewards_t = torch.from_numpy(rewards).float().to(self._device)
        next_states_t = torch.from_numpy(next_states).float().to(self._device)
        dones_t = torch.from_numpy(dones).float().to(self._device)

        q_values = self._model(states_t)
        q_current = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._target_model(next_states_t)
            next_q_max = next_q_values.max(dim=1)[0]
            q_target = rewards_t + self._gamma * next_q_max * (1 - dones_t)

        loss = F.mse_loss(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._t += 1
        if self._t % self._target_update == 0:
            self._target_model.load_state_dict(self._model.state_dict())

    def decay_epsilon(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def current_epsilon(self) -> float:
        return self._epsilon

    def save(self, path: Path) -> None:
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "target_model_state_dict": self._target_model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "hyperparams": dict(self._hyperparams),
            "t": self._t,
            "epsilon": self._epsilon,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(str(path), map_location=self._device,
                                weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._t = checkpoint.get("t", 0)
        self._epsilon = checkpoint.get("epsilon", self._epsilon)
        log.info("loaded DQN model from %s", path)
