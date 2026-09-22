# TrafficRL

[![CI](https://github.com/Kaoserahamed/AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaoserahamed/AI/actions/workflows/ci.yml)
[![CD](https://github.com/Kaoserahamed/AI/actions/workflows/cd.yml/badge.svg)](https://github.com/Kaoserahamed/AI/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/traffic-rl-agents)](https://pypi.org/project/traffic-rl-agents/)
[![CodeQL](https://github.com/Kaoserahamed/AI/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Kaoserahamed/AI/actions/workflows/codeql-analysis.yml)
[![Security](https://img.shields.io/badge/Security-GHAS-green)](https://github.com/Kaoserahamed/AI/security)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa)](CODE_OF_CONDUCT.md)

**Reinforcement Learning for Traffic Signal Control** — a production-grade Python package for training and evaluating RL agents (Q-Learning, DQN, Double DQN, PPO) in SUMO traffic simulations.

- **Repository**: [github.com/Kaoserahamed/AI](https://github.com/Kaoserahamed/AI)
- **Package source**: [`traffic-rl-agents/`](traffic-rl-agents/) (this repo's installable project)
- **Documentation**: [`traffic-rl-agents/docs/`](traffic-rl-agents/docs/)
- **Contributing**: [CONTRIBUTING.md](traffic-rl-agents/CONTRIBUTING.md)
- **Security**: [SECURITY.md](traffic-rl-agents/SECURITY.md)
- **Changelog**: [CHANGELOG.md](traffic-rl-agents/CHANGELOG.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](traffic-rl-agents/CODE_OF_CONDUCT.md)

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [RL Algorithms](#rl-algorithms)
- [Evaluation](#evaluation)
- [Observability](#observability)
- [AWS Deployment](#aws-deployment)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multiple RL Algorithms**: Q-Learning, DQN, Double DQN, PPO (clipped surrogate)
- **Secure Model Persistence**: JSON-based Q-table storage with SHA-256 checksums, safe PyTorch checkpoint loading (`weights_only=True`)
- **Reproducible Training**: Seed management for Python, NumPy, and PyTorch; config hashing
- **Configurable Environments**: Immutable config dataclasses with validation
- **Structured Observability**: JSON logging, metrics sinks (JSONL, TensorBoard, CloudWatch, SageMaker)
- **YAML Experiment Config**: Layered configs for local/CI/staging/prod with pydantic validation
- **AWS Integration**: OIDC authentication, ECR, S3, SageMaker training and endpoints
- **CI/CD Automation**: Linting, type checking, testing, and security scanning
- **CLI Interface**: Easy-to-use command-line tools for training and evaluation
- **Safety First**: Agent safety guardrails, evaluation gates, domain randomization

## Repository Layout

```
AI/                          # repository root (this README)
├── README.md                # project overview and quick start
├── CODE_OF_CONDUCT.md       # contributor covenant
├── traffic-rl-agents/       # the installable Python package
│   ├── pyproject.toml       # packaging and tooling configuration
│   ├── Dockerfile           # container image for deployment
│   ├── .pre-commit-config.yaml  # pre-commit hooks
│   ├── .env.example         # environment variable template
│   ├── conftest.py          # root pytest configuration
│   ├── CHANGELOG.md         # version history
│   ├── CONTRIBUTING.md      # development guide
│   ├── SECURITY.md          # security policy
│   ├── configs/             # YAML experiment configurations
│   │   └── base.yaml, ci.yaml, local.yaml, prod.yaml, staging.yaml
│   ├── docs/                # full documentation
│   │   └── api.md, architecture.md, configuration.md, models.md, safety.md
│   ├── infrastructure/      # AWS CloudFormation templates
│   │   └── cloudformation.yaml, cloudformation-sagemaker.yaml
│   ├── sim/                 # SUMO network, demand, and config files
│   └── src/traffic_rl/      # package source
│       ├── cli.py           # command-line interface
│       ├── config.py        # configuration dataclasses
│       ├── config_loader.py # YAML experiment config + validation
│       ├── settings.py      # env-var based runtime settings
│       ├── io_utils.py      # safe model persistence
│       ├── logging.py       # centralized logging
│       ├── metrics.py       # traffic metrics and reward computation
│       ├── agents/          # RL algorithms (Q-Learning, DQN, Double DQN, PPO)
│       ├── sumo/            # SUMO environment wrapper + controllers
│       ├── utils/           # seeding, replay buffer, torch helpers
│       ├── observability/   # structured logging, metrics sinks, profiling
│       └── tests/           # unit and integration tests
├── .github/                 # CI/CD workflows and issue templates
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
└── (this README)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Kaoserahamed/AI.git
cd AI

# The installable package lives in the "traffic-rl-agents" folder
cd "traffic-rl-agents"

# Install in development mode
pip install -e ".[dev,sumo]"
```

## Requirements

- **Python** 3.10+
- **SUMO** installed and accessible (via `SUMO_HOME` or `PATH`) — for running simulations
- **PyTorch** — for neural network agents (DQN, Double DQN, PPO)
- **NumPy, Pandas, Matplotlib** — for numerical computation and plotting
- **Pydantic, PyYAML** — for configuration management
- **Boto3, SageMaker SDK** — for AWS deployment (optional, use `[cloud]` extra)

## Quick Start

All CLI usage goes through the `traffic-train` entry point, which exposes a `train` and an `eval` subcommand.

### Training an Agent

```bash
# Train a DQN agent
traffic-train train --agent dqn --episodes 200 --seed 42

# Train with GUI visualization
traffic-train train --agent ppo --gui --episodes 100

# Train using YAML config
traffic-train train --config configs/base.yaml --overlay configs/local.yaml
```

The same commands can be run from the CLI module without installing the console script:

```bash
python -m traffic_rl.cli train --agent dqn --episodes 200 --seed 42
```

### Evaluating an Agent

```bash
# Evaluate a trained model
traffic-train eval --agent dqn --model logs/train_dqn/best_model.pth --episodes 10

# Evaluate against baselines
traffic-train eval --agent ppo --model logs/PPO/best_ppo_model.pth \
  --baselines fixed_time actuated max_pressure
```

## Project Structure

Inside `traffic-rl-agents/src/traffic_rl/`:

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

This project uses two complementary configuration systems:

1. **Dataclass config** (`traffic_rl.config`): Used by the CLI for quick train/eval runs. Immutable, validated dataclasses with sensible defaults.

2. **YAML experiment config** (`traffic_rl.config_loader`): For large-scale training and AWS deployment. Layered YAML files (`base.yaml`, `<env>.yaml`) validated with pydantic models.

```python
from traffic_rl import resolve_config

# Override episodes and seed
config = resolve_config({"episodes": 300, "seed": 123})
```

### YAML Config Layering

```
configs/base.yaml      ← defaults for every knob
configs/<env>.yaml     ← overlay (local, ci, staging, prod)
.env / env vars        ← highest precedence for deployment knobs
```

```bash
# Local development
traffic-train train --config configs/base.yaml --overlay configs/local.yaml

# Production
traffic-train train --config configs/base.yaml --overlay configs/prod.yaml
```

See [docs/configuration.md](traffic-rl-agents/docs/configuration.md) for the full reference.

## Security

TrafficRL is designed with security as a priority:

- **Safe checkpoint loading**: `torch.load(weights_only=True)` for all neural checkpoints
- **JSON Q-tables**: No pickle for tabular agents; SHA-256 integrity verification
- **Restricted unpickler**: Legacy `.pkl` files use an explicit allow-list unpickler
- **Log redaction**: Credentials masked in structured logs via regex + secret matching
- **Dependency scanning**: pip-audit and bandit in CI
- **AWS OIDC**: No static AWS keys in CI/CD pipelines

See [SECURITY.md](traffic-rl-agents/SECURITY.md) for the full policy.

## Architecture

TrafficRL follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Entry Point                 │
├─────────────────────────────────────────────────────┤
│                 Training Loop                       │
├─────────────────────────────────────────────────────┤
│              RL Agents (Q-Learning, DQN, PPO)       │
├─────────────────────────────────────────────────────┤
│          Environment Interface (SUMO)                │
├─────────────────────────────────────────────────────┤
│           Metrics / State / Reward (pure)            │
├─────────────────────────────────────────────────────┤
│               SUMO Simulation Engine                 │
└─────────────────────────────────────────────────────┘
```

Key design principles:
- **Pure functions** for metrics and reward computation (no I/O, no external deps)
- **Pluggable agents** via a common `BaseAgent` interface
- **Reproducibility**: seeded RNG, config hashing, immutable configs
- **Observability**: structured logging with context propagation

See [docs/architecture.md](traffic-rl-agents/docs/architecture.md) for details.

## RL Algorithms

| Agent           | Type         | Persistence  | Description                          |
|-----------------|--------------|--------------|--------------------------------------|
| Q-Learning      | Tabular      | JSON         | Discrete state binning, Q-table      |
| DQN             | Neural       | `.pth`       | MLP + experience replay + target net |
| Double DQN      | Neural       | `.pth`       | Reduces overestimation bias         |
| PPO             | Neural       | `.pth`       | Clipped surrogate objective         |

The observation is a 28-dimensional state vector (vehicle counts, queue
lengths, densities across 4 incoming edges). The reward balances queue
reduction, waiting time, speed, throughput, and emergency priority.

See [docs/models.md](traffic-rl-agents/docs/models.md) for the full reward design.

## Evaluation

Production evaluation follows a rigorous protocol:

1. **Multi-seed evaluation**: 5 random seeds (1–5)
2. **Multi-split validation**: Val and test scenario splits
3. **Baseline comparison**: Fixed-time, actuated, and max-pressure controllers
4. **Quality gates** (enforced in `configs/prod.yaml`):
   - Minimum 10% improvement over fixed-time baseline
   - Average delay ≤ 45 seconds
   - 95th percentile queue ≤ 25 vehicles

## Observability

The `traffic_rl.observability` package provides:

- **Structured logging**: JSON or console format with context propagation
- **Metrics sinks**: JSONL, TensorBoard, SageMaker, CloudWatch (multi-sink with error isolation)
- **Events**: Stable event names and schema (run lifecycle, training, evaluation, etc.)
- **Profiling**: `PhaseTimer` and `StepRateMeter` for performance analysis
- **Structured test reporting**: JSON report for CI with environment metadata

See [docs/api.md](traffic-rl-agents/docs/api.md) for the full API reference.

## AWS Deployment

TrafficRL supports deployment on AWS:

- **ECR**: Containerized application image
- **S3**: Artifact storage with KMS encryption
- **SageMaker**: Managed training (spot), HPO tuning, and real-time endpoints
- **IAM**: OIDC federation for CI/CD (no static keys)
- **Secrets Manager / SSM**: Runtime secret resolution

Infrastructure is defined as code via CloudFormation:
- `infrastructure/cloudformation.yaml` — core (ECR, S3, IAM)
- `infrastructure/cloudformation-sagemaker.yaml` — SageMaker resources

See [docs/safety.md](traffic-rl-agents/docs/safety.md) for security considerations.

## Testing

Tests live in `traffic-rl-agents/src/traffic_rl/tests/`:

```bash
cd traffic-rl-agents

# All tests
pytest src/traffic_rl/tests/ -v --cov=traffic_rl

# With structured JSON report (used by CI)
pytest src/traffic_rl/tests/ --structured-report reports/test-report.json

# Skip SUMO/AWS integration tests
pytest src/traffic_rl/tests/ -m "not sumo and not aws"
```

### Test Markers

| Marker        | Description                                    |
|---------------|------------------------------------------------|
| `unit`        | Pure unit tests (fast, no external deps)       |
| `integration` | Tests requiring external services              |
| `sumo`        | Tests requiring a SUMO installation            |
| `aws`         | Tests requiring AWS credentials                |
| `performance` | Performance or timing-sensitive tests          |
| `gpu`         | Tests requiring a GPU                          |

## Code Quality

```bash
cd traffic-rl-agents

# Linting
ruff check src/

# Formatting
ruff format src/

# Type checking
mypy src/traffic_rl/

# Pre-commit hooks
pre-commit run --all-files
```

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](traffic-rl-agents/CONTRIBUTING.md)
for development setup, testing guidelines, and the pull request process.

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT License — see [LICENSE](LICENSE) or the license field in `traffic-rl-agents/pyproject.toml`.
