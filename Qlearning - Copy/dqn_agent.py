import numpy as np
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_size=8, action_size=2, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for Traffic Light Control"""
    def __init__(self, state_size=8, action_size=2, lr=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, buffer_size=10000, target_update=100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Training counters
        self.steps_done = 0
        self.training_steps = 0
        
    def preprocess_state(self, state):
        """Preprocess state for neural network"""
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
        
        # Ensure state is the right size
        if len(state) != self.state_size:
            # Pad or truncate to correct size
            processed_state = np.zeros(self.state_size, dtype=np.float32)
            min_len = min(len(state), self.state_size)
            processed_state[:min_len] = state[:min_len]
            return processed_state
        
        # Normalize state values
        processed_state = np.array(state, dtype=np.float32)
        
        # Normalize queue lengths (typically 0-20) to 0-1
        processed_state[:4] = np.clip(processed_state[:4] / 20.0, 0, 1)
        
        # Normalize densities (typically 0-5) to 0-1
        if len(processed_state) > 4:
            processed_state[4:] = np.clip(processed_state[4:] / 5.0, 0, 1)
        
        return processed_state
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        self.steps_done += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if random.random() > self.epsilon:
            # Exploitation: choose best action
            state_tensor = torch.FloatTensor(self.preprocess_state(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
    
    def learn(self, state, action, reward, next_state, done=False):
        """Store experience and train the network"""
        # Store experience in replay buffer
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)
        
        self.memory.push(processed_state, action, reward, processed_next_state, done)
        
        # Train if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Update target network
        if self.training_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and parameters
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_steps': self.training_steps,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }, filepath)
        
        print(f"DQN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # Load network states
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load training state
                self.epsilon = checkpoint['epsilon']
                self.steps_done = checkpoint['steps_done']
                self.training_steps = checkpoint['training_steps']
                
                print(f"DQN model loaded from {filepath}")
                print(f"  Training steps: {self.training_steps}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
                print("Starting with fresh model...")
        else:
            print(f"No model found at {filepath}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'q_table_size': len(self.memory),  # Use replay buffer size instead
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'steps_done': self.steps_done,
            'avg_q_value': self._get_avg_q_value()
        }
    
    def _get_avg_q_value(self):
        """Calculate average Q-value (for monitoring)"""
        if len(self.memory) == 0:
            return 0.0
        
        try:
            # Sample a small batch to estimate average Q-value
            sample_size = min(100, len(self.memory))
            states, _, _, _, _ = self.memory.sample(sample_size)
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(states_tensor)
                return q_values.mean().item()
        except:
            return 0.0
    
    def set_train_mode(self, train=True):
        """Set training mode"""
        if train:
            self.q_network.train()
        else:
            self.q_network.eval()
    
    def get_memory_usage(self):
        """Get memory buffer usage"""
        return len(self.memory)