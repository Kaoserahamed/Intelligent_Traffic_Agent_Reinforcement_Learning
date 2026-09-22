"""Agent registry and factory for RL agents.

Import this module to trigger agent registration.
"""

from traffic_rl.agents.base import BaseAgent, NeuralAgent, State, Action
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.double_dqn import DoubleDQNAgent
from traffic_rl.agents.ppo import PPOAgent

__all__ = [
    "BaseAgent",
    "NeuralAgent",
    "State",
    "Action",
    "QLearningAgent",
    "DQNAgent",
    "DoubleDQNAgent",
    "PPOAgent",
]

