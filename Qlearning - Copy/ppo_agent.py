import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.shared(state)
        logits = self.actor(x)
        value = self.critic(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, clip_epsilon=0.2,
                 lr=3e-4, batch_size=64, update_epochs=10, memory_size=5000):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.Transition = namedtuple('Transition',
            ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])
        self.memory = deque(maxlen=memory_size)

    def preprocess_state(self, state):
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
        state = np.array(state, dtype=np.float32)
        state[:4] = np.clip(state[:4] / 20.0, 0, 1)
        if len(state) > 4:
            state[4:] = np.clip(state[4:] / 5.0, 0, 1)
        return state

    def choose_action(self, state, deterministic=False):
        state = self.preprocess_state(state)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        dist, value = self.policy(state_tensor)
        action = dist.probs.argmax() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        transition = self.Transition(state, action, reward, next_state, done, log_prob, value)
        self.memory.append(transition)

    def learn(self, state, action, reward, next_state, done):
        action, log_prob, value = self.choose_action(state)
        self.store_transition(state, action, reward, next_state, done, log_prob, value)
        if len(self.memory) >= self.batch_size:
            self.update()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = list(self.memory)
        batch = self.Transition(*zip(*transitions))

        states = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(batch.log_prob, dtype=torch.float32).to(self.device)
        values = torch.tensor(batch.value, dtype=torch.float32).to(self.device)

        # Compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        for _ in range(self.update_epochs):
            dist, new_values = self.policy(states)
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        self.memory.clear()

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy.state_dict(), filepath)
        print(f"PPO model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            try:
                self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
                print(f"PPO model loaded from {filepath}")
            except Exception as e:
                print(f"Failed to load PPO model: {e}")
        else:
            print(f"No PPO model found at {filepath}")

    def get_stats(self):
        return {
            "q_table_size": len(self.memory),
            "epsilon": 0.0,
        }

    def set_train_mode(self, train=True):
        if train:
            self.policy.train()
        else:
            self.policy.eval()
