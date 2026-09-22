"""Double DQN - uses online network for action selection, target for evaluation."""

from __future__ import annotations

import torch

from traffic_rl.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN: uses the online network for action selection, target network
    for evaluation — reducing overestimation bias."""

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
            # Double DQN: select action with online network, evaluate with target
            next_actions = self._model(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self._target_model(next_states_t).gather(1, next_actions).squeeze(1)
            q_target = rewards_t + self._gamma * next_q_values * (1 - dones_t)

        loss = torch.nn.functional.mse_loss(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._t += 1
        if self._t % self._target_update == 0:
            self._target_model.load_state_dict(self._model.state_dict())
