# agents/dqn_agent.py

import random
import numpy as np
from agents.base_agent import BaseTrafficAgent
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class DQNTrafficAgent(BaseTrafficAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.01, batch_size=64, memory_size=2000):
        super().__init__()
        self.name = "DQN"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        model = models.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def update(self, state, action, reward, next_state):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        target = reward
        if next_state is not None:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
