# TrafficRL Documentation

Reinforcement Learning for Traffic Signal Control — a production-grade Python
package for training and evaluating RL agents (Q-Learning, DQN, Double DQN, PPO)
in SUMO traffic simulations.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Configuration](#configuration)
- [RL Models](#rl-models)
- [Observability](#observability)
- [Safety & Security](#safety--security)
- [AWS Deployment](#aws-deployment)
- [Development](#development)
- [API Reference](#api-reference)

## Getting Started

TrafficRL trains reinforcement learning agents to control traffic signal
timings in SUMO (Simulation of Urban Mobility) intersections. The key idea:
instead of fixed-cycle or actuated timing plans, an RL agent learns to
dynamically select signal phases to minimize queue length, waiting time,
and delay while maximizing throughput and emergency-vehicle priority.

### Quick Start

```bash
# Train a DQN agent
traffic-train train --agent dqn --episodes 200 --seed 42

# Evaluate a trained model
traffic-train eval --agent dqn --model logs/train_dqn/best_model.pth --episodes 10
```

### Quick Start (Python API)

```python
from traffic_rl import resolve_config
from traffic_rl.config import AgentConfig, Config, EnvironmentConfig
from traffic_rl.sumo.environment import TrafficEnvironment
from traffic_rl.agents import DQNAgent
from traffic_rl.utils.random import seed_everything

config = resolve_config({"episodes": 100, "seed": 42})
env = TrafficEnvironment(config.environment)
agent = DQNAgent(config.agents[1])  # DQN agent
seed_everything(42)

for episode in range(config.training.episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, epsilon=0.1)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## Installation

```bash
pip install -e ".[dev,sumo]"
```

Requirements:
- Python 3.10+
- SUMO (for simulation runtime)
- PyTorch, NumPy, Pandas, Matplotlib

## Architecture Overview

See [Architecture Guide](architecture.md) for the full design.

```
traffic_rl/
├── src/traffic_rl/
│   ├── cli.py              # Command-line interface (train/eval)
│   ├── config.py           # Immutable config dataclasses
│   ├── config_loader.py    # YAML experiment config + validation
│   ├── settings.py         # Env-var based runtime settings
│   ├── metrics.py          # State, reward, metrics computation (pure functions)
│   ├── io_utils.py         # Safe model persistence (JSON + weights_only)
│   ├── logging.py          # Centralized logging
│   ├── agents/             # RL algorithms (Q-Learning, DQN, Double DQN, PPO)
│   ├── sumo/               # SUMO environment wrapper + controllers
│   ├── utils/              # Seeding, replay buffer, torch helpers
│   └── observability/      # Structured logging, metrics sinks, profiling
├── sim/                    # SUMO network, demand, and config files
├── configs/                # YAML experiment configurations
├── tests/                  # Unit, integration, and smoke tests
└── Dockerfile              # Containerized deployment
```

## Configuration

There are two configuration systems:

1. **Dataclass config** (`traffic_rl.config`): Used by the CLI for quick
   training/evaluation. Immutable, validated dataclasses.

2. **YAML experiment config** (`traffic_rl.config_loader`): Used for
   large-scale training and AWS deployment. Layered YAML files with pydantic
   validation and a config hash for reproducibility.

See [Configuration Guide](configuration.md) for full details.

## RL Models

TrafficRL implements four RL algorithms:

| Agent         | Type         | Description                                   | Model File       |
|---------------|--------------|-----------------------------------------------|------------------|
| Q-Learning    | Tabular      | Discrete state binning, Q-table as JSON       | `q_learning.py`  |
| DQN           | Neural       | MLP with experience replay + target network   | `dqn.py`         |
| Double DQN    | Neural       | Reduces overestimation bias                   | `double_dqn.py`  |
| PPO           | Neural       | Clipped surrogate objective, on-policy        | `ppo.py`         |

See [Models Guide](models.md) for reward function design and training details.

## Observability

Structured logging and metrics collection via the
`traffic_rl.observability` package:

- **Structured logging**: JSON or console format with context propagation
- **Metrics sinks**: JSONL, TensorBoard, SageMaker, CloudWatch
- **Events**: Stable event names and schema (see `events.py`)
- **Profiling**: Phase timers and step-rate meters

## Safety & Security

See [Safety Guide](safety.md) for full details on:

- Model checkpoint security (weights_only, JSON Q-tables)
- Runtime safety guardrails
- AI-specific evaluation gates and reproducibility
- Network security and dependency monitoring

## AWS Deployment

The project supports training on AWS SageMaker with:

- ECR for container images
- S3 for artifact storage
- IAM OIDC for credential-free CI/CD
- SageMaker managed spot training

See [Infrastructure](../infrastructure/README.md) for CloudFormation templates.

## Development

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup, testing, and
pull request guidelines.

## API Reference

- [API Reference](api.md)
- [Architecture Guide](architecture.md)
- [Configuration Guide](configuration.md)
- [Models Guide](models.md)
- [Safety Guide](safety.md)
