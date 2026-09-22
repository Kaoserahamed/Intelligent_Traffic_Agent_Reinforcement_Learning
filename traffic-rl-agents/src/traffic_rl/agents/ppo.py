"""Proximal Policy Optimization (PPO) agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from traffic_rl.agents.base import NeuralAgent
from traffic_rl.config import ACTION_SIZE, STATE_SIZE
from traffic_rl.logging import get_logger

log = get_logger("traffic_rl.agents.ppo")


class _PolicyValueNet(nn.Module):
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


class PPOAgent(NeuralAgent):
    """PPO agent with clipped surrogate objective."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._action_size = ACTION_SIZE
        self._state_size = STATE_SIZE
        self._lr = float(config.hyperparams.get("lr", 3e-4))
        self._gamma = float(config.hyperparams.get("gamma", 0.99))
        self._clip_epsilon = float(config.hyperparams.get("clip_epsilon", 0.2))
        self._ppo_epochs = int(config.hyperparams.get("ppo_epochs", 4))
        self._batch_size = int(config.hyperparams.get("batch_size", 64))
        self._entropy_coef = float(config.hyperparams.get("entropy_coef", 0.01))
        self._value_coef = float(config.hyperparams.get("value_coef", 0.5))
        self._model = _PolicyValueNet(
            self._state_size, self._action_size).to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._episode_states: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_log_probs: list[float] = []
        self._episode_rewards: list[float] = []
        self._episode_done: list[bool] = []
        self._t = 0

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if state.size == 0:
            return int(np.random.randint(0, self._action_size))
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
            logits, _ = self._model(state_t)
            probs = torch.softmax(logits, dim=1)
            d = torch.distributions.Categorical(probs)
            action = d.sample()
            log_prob = d.log_prob(action).item()
        self._episode_states.append(state.copy())
        self._episode_actions.append(int(action.item()))
        self._episode_log_probs.append(log_prob)
        return int(action.item())

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        self._episode_rewards.append(reward)
        self._episode_done.append(done)
        self._t += 1

    def decay_epsilon(self) -> None:
        pass

    def current_epsilon(self) -> float:
        return 0.0

    def end_episode(self) -> float:
        if not self._episode_rewards:
            return 0.0
        avg_reward = float(np.mean(self._episode_rewards))
        self._train_policy()
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_log_probs.clear()
        self._episode_rewards.clear()
        self._episode_done.clear()
        return avg_reward

    def _train_policy(self) -> None:
        states = np.stack(self._episode_states)
        actions = np.array(self._episode_actions)
        old_log_probs = np.array(self._episode_log_probs)
        rewards = np.array(self._episode_rewards)
        dones = np.array(self._episode_done, dtype=np.float32)

        rewards_to_go = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self._gamma * running * (1 - dones[t])
            rewards_to_go[t] = running

        advantages = rewards_to_go - np.mean(rewards_to_go)
        if np.std(advantages) > 0:
            advantages = advantages / np.std(advantages)

        states_t = torch.from_numpy(states).float().to(self._device)
        actions_t = torch.from_numpy(actions).long().to(self._device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(self._device)
        advantages_t = torch.from_numpy(advantages).float().to(self._device)
        returns_t = torch.from_numpy(rewards_to_go).float().to(self._device)

        n = len(states)
        indices = np.arange(n)
        for _ in range(self._ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                bidx = indices[start:end]
                bs = states_t[bidx]
                ba = actions_t[bidx]
                blp = old_log_probs_t[bidx]
                badv = advantages_t[bidx]
                bret = returns_t[bidx]

                logits, values = self._model(bs)
                d = torch.distributions.Categorical(logits=logits)
                new_log_probs = d.log_prob(ba)
                entropy = d.entropy().mean()

                ratio = torch.exp(new_log_probs - blp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - self._clip_epsilon,
                                   1 + self._clip_epsilon) * badv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(), bret)

                loss = (policy_loss +
                        self._value_coef * value_loss -
                        self._entropy_coef * entropy)

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
                self._optimizer.step()

        self._t += 1

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "hyperparams": {"lr": self._lr, "gamma": self._gamma,
                           "clip_epsilon": self._clip_epsilon},
            "t": self._t,
        }, str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(str(path), map_location=self._device,
                                weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._t = checkpoint.get("t", 0)
        log.info("loaded PPO model from %s", path)

