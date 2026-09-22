# Contributing to TrafficRL

Thank you for your interest in contributing! This document covers everything you
need to get started as a contributor.

## Table of Contents

- [Development Environment](#development-environment)
- [Quick Start](#quick-start)
- [Code Style](#code-style)
- [Testing](#testing)
- [Adding a New RL Agent](#adding-a-new-rl-agent)
- [Configuration Changes](#configuration-changes)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)
- [Release Process](#release-process)

## Development Environment

### Prerequisites

- Python 3.10 or 3.11
- [SUMO](https://sumo.dlr.de/) installed (for running simulations; not required for unit tests)
- [Docker](https://docs.docker.com/get-docker/) (for containerized runs)
- AWS CLI configured (for cloud deployment; not required for local development)

### Installation

```bash
# Clone and enter the repository
git clone https://github.com/Kaoserahamed/AI.git
cd AI/traffic-rl-agents

# Install in development mode with all dev dependencies
pip install -e ".[dev,sumo]"

# Copy the environment template (edit as needed for cloud runs)
cp .env.example .env
```

## Quick Start

```bash
# Run the test suite
pytest src/traffic_rl/tests/ -v --cov=traffic_rl

# Or use the structured report
pytest src/traffic_rl/tests/ --structured-report reports/test-report.json

# Validate configuration
python -c "from traffic_rl.config_loader import load_experiment_config; load_experiment_config()"

# Run a short training
python -m traffic_rl.cli train --agent dqn --episodes 10
```

## Code Style

This project uses:

- **Ruff** for linting and formatting
- **Mypy** for static type checking
- **Pre-commit** for git hooks (see `.pre-commit-config.yaml`)

```bash
# Format
ruff format src/

# Lint
ruff check src/

# Type check
mypy src/traffic_rl/
```

## Testing

Tests live in `src/traffic_rl/tests/` and are organized by module. Every new
feature or bug fix should include tests.

```bash
# All tests
pytest src/traffic_rl/tests/ -v

# With coverage report
pytest src/traffic_rl/tests/ --cov=traffic_rl --cov-report=term-missing

# Only fast unit tests (skip SUMO and AWS integration tests)
pytest src/traffic_rl/tests/ -m "not sumo and not aws"

# Structured JSON report (used by CI)
pytest src/traffic_rl/tests/ --structured-report reports/test-report.json
```

### Test markers

| Marker        | Description                          |
|---------------|--------------------------------------|
| `unit`        | Pure unit tests (fast, no dependencies) |
| `integration` | Tests requiring external services     |
| `sumo`        | Tests requiring a SUMO installation    |
| `aws`         | Tests requiring AWS credentials       |
| `performance` | Performance / timing-sensitive tests  |

## Adding a New RL Agent

1. Create `src/traffic_rl/agents/<name>.py` subclassing `BaseAgent` or `NeuralAgent`.
2. Register it in `src/traffic_rl/agents/__init__.py`.
3. Add the agent kind to `AgentConfig.kind` validation in `config.py`.
4. Add a CLI branch in `cli.py` if it has unique hyperparameters.
5. Add tests in `src/traffic_rl/tests/test_agents.py` (if not already covered).

## Configuration Changes

Experiment configuration uses YAML files in `configs/`. The layering model is:

1. `configs/base.yaml` — defaults for every environment
2. `configs/<env>.yaml` — overlay (`local`, `ci`, `staging`, `prod`)
3. Environment variables — highest precedence for deployment knobs
4. CLI overrides — final in-memory adjustments

All config is validated with pydantic models using `extra="forbid"`, so a typo
fails immediately in CI.

## Documentation

Documentation lives in `docs/`. Key files:

| File                  | Contents                                      |
|-----------------------|-----------------------------------------------|
| `docs/index.md`       | Documentation home                             |
| `docs/architecture.md`| System architecture and design decisions       |
| `docs/api.md`         | Public API reference                           |
| `docs/configuration.md`| Configuration and environment variables       |
| `docs/models.md`      | RL models and reward function design          |
| `docs/safety.md`      | Safety, security, and guardrails documentation|

## Pull Requests

1. Fork the repository and create a feature branch.
2. Make your changes, following the code style above.
3. Add or update tests as needed.
4. Ensure all CI checks pass.
5. Open a pull request with a clear description.

See `.github/pull_request_template.md` for the PR checklist.

## Release Process

Releases are automated via GitHub Actions (`release.yml`). To cut a new release:

```bash
git checkout main
git pull
git tag v0.2.0
git push origin v0.2.0
```

---

By contributing, you agree that your contributions will be licensed under the
MIT License that covers the project.
