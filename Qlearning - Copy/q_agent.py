import numpy as np
import random
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_size=8, action_size=2, alpha=0.1, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # State discretization
        self.bins = 10
        
    def get_state_key(self, state):
        """Convert continuous state to discrete key"""
        if state is None or len(state) != self.state_size:
            return tuple([0] * self.state_size)
        
        # Discretize each state component
        discrete_state = []
        for i, value in enumerate(state):
            if i < 4:  # Queue lengths (0-20)
                bin_val = min(int(value / 2), self.bins - 1)
            else:  # Densities (0-5)
                bin_val = min(int(value), self.bins - 1)
            discrete_state.append(bin_val)
        
        return tuple(discrete_state)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_table[state_key]
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done=False):
        """Q-learning update"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Q-learning formula
        current_q = self.q_table[state_key][action]
        
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * next_max_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.alpha * (target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save Q-table"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'params': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min
                }
            }, f)
    
    def load_model(self, filepath):
        """Load Q-table"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                # Convert back to defaultdict
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                for key, value in data['q_table'].items():
                    self.q_table[key] = value
                
                self.epsilon = data['epsilon']
                print(f"Model loaded: {len(self.q_table)} states")
        else:
            print(f"No model found at {filepath}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'avg_q_value': np.mean([np.mean(q) for q in self.q_table.values()]) if self.q_table else 0
        }