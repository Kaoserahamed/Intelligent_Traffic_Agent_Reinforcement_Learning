# q_agent.py
import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = {}
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table"""
        if state is None:
            return tuple([0] * self.state_size)
        return tuple(np.array(state, dtype=int))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        key = self.get_state_key(state)
        
        # Exploration: random action
        if random.random() < self.epsilon or key not in self.q_table:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: best known action
        return np.argmax(self.q_table[key])
    
    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not seen before
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)
        
        # Q-learning update
        best_next_action = np.max(self.q_table[next_key])
        td_target = reward + self.gamma * best_next_action
        td_delta = td_target - self.q_table[key][action]
        self.q_table[key][action] += self.alpha * td_delta
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the Q-table to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'hyperparameters': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min
                }
            }, f)
    
    def load_model(self, filepath):
        """Load the Q-table from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
                print(f"Model loaded from {filepath}")
        else:
            print(f"No saved model found at {filepath}")