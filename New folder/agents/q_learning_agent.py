# agents/q_learning_agent.py

import numpy as np
import random
from agents.base_agent import BaseTrafficAgent

class QLearningAgent(BaseTrafficAgent):
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__()
        self.name = "Q_Learning" 
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
