from agents.base_agent import BaseTrafficAgent
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random

class DuelingDQNTrafficAgent(BaseTrafficAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__()
        self.name = "Dueling_DQN"
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        input_layer = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(64, activation='relu')(input_layer)

        value_fc = layers.Dense(64, activation='relu')(dense1)
        value = layers.Dense(1)(value_fc)

        advantage_fc = layers.Dense(64, activation='relu')(dense1)
        advantage = layers.Dense(self.action_size)(advantage_fc)

        q_values = layers.Add()([value, layers.Subtract()([advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)])])
        model = models.Model(inputs=input_layer, outputs=q_values)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
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
        target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
