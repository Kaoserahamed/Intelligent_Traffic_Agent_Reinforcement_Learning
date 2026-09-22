# TrafficRL

Reinforcement Learning for Traffic Signal Control

A production-grade Python package for training and evaluating RL agents
(Q-Learning, DQN, Double DQN, PPO) in SUMO traffic simulations.

## Features

- **Multiple RL Algorithms**: Q-Learning, DQN, Double DQN, PPO
- **Secure Model Persistence**: JSON-based Q-table storage, safe PyTorch checkpoint loading
- **Reproducible Training**: Seed management for Python, NumPy, and PyTorch
- **Configurable Environments**: Immutable config dataclasses with validation
- **CLI Interface**: Easy-to-use command-line tools for training and evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/traffic_rl.git
cd traffic_rl

# Install in development mode
pip install -e ".[dev,sumo]"
```

## Requirements

- Python 3.10+
- SUMO installed and accessible (via `SUMO_HOME` or `PATH`)
- PyTorch
- NumPy, Pandas, Matplotlib

## Quick Start

### Training an Agent

```bash
# Train a DQN agent
traffic-train --agent dqn --episodes 200 --seed 42

# Train with GUI visualization
traffic-train --agent ppo --gui --episodes 100
```

### Evaluating an Agent

```bash
# Evaluate a trained model
traffic-eval --agent dqn --model logs/train_dqn/best_model.pth --episodes 10
```

## Project Structure

```
traffic_rl/
├── __init__.py          # Package entry point
├── __about__.py         # Package metadata (version)
├── cli.py               # Command-line interface
├── config.py            # Configuration dataclasses
├── defaults.py          # Default configuration factories
├── io_utils.py          # Safe model persistence (JSON/pickle)
├── logging.py           # Centralized logging configuration
├── metrics.py           # Traffic metrics and reward computation
├── agents/
│   ├── __init__.py      # Agent exports
│   ├── base.py          # Base agent classes
│   ├── q_learning.py    # Q-Learning agent
│   ├── dqn.py           # DQN agent
│   ├── double_dqn.py    # Double DQN agent
│   ├── ppo.py           # PPO agent
│   └── ppo_net.py       # PPO network architecture
├── sumo/
│   ├── __init__.py
│   ├── binary.py        # SUMO binary location
│   ├── controller.py    # Rule-based controllers
│   └── environment.py   # SUMO environment wrapper
└── utils/
    ├── __init__.py
    ├── random.py        # Seeding utilities
    ├── replay.py        # Experience replay buffer
    └── torch.py         # PyTorch device helpers
```

## Configuration

The package uses immutable, validated dataclasses for configuration:

- `EnvironmentConfig`: SUMO simulation parameters
- `TrainingConfig`: Training hyperparameters
- `AgentConfig`: Agent-specific configuration
- `Config`: Top-level application configuration

Override defaults via the CLI or programmatically:

```python
from traffic_rl import resolve_config

# Override episodes and seed
config = resolve_config({"episodes": 300, "seed": 123})
```

## Security

- **JSON-based Q-table storage**: No pickle for tabular agents
- **Safe PyTorch loading**: `torch.load(weights_only=True)` for neural agents
- **Restricted unpickler**: Legacy `.pkl` files use a restricted unpickler with allow-list

## Development

### Running Tests

```bash
pytest src/traffic_rl/tests/ -v --cov=traffic_rl
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/traffic_rl/

# Formatting
black src/
```

## License

MIT License - see LICENSE file for details.
