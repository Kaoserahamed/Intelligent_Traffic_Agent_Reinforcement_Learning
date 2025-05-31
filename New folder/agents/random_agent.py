# agents/random_agent.py

import random
from agents.base_agent import BaseTrafficAgent

class RandomAgent(BaseTrafficAgent):
    def __init__(self, action_size):
        self.action_size = action_size
        super().__init__()
        self.name = "Random"

    def choose_action(self, state):
        return random.randint(0, self.action_size - 1)

    def update(self, state, action, reward, next_state):
        pass  # No learning
