# Configuration

TrafficRL has two complementary configuration systems:

1. **Dataclass config** (`traffic_rl.config`) — used by the CLI for quick
   train/eval runs. Immutable, validated dataclasses with sensible defaults.
2. **YAML experiment config** (`traffic_rl.config_loader`) — used for
   large-scale training, AWS deployment, and reproducibility. Layered YAML
   files validated with pydantic models.

## Dataclass Config (CLI)

The CLI uses frozen, validated dataclasses:

```python
from traffic_rl import resolve_config

config = resolve_config({"episodes": 300, "seed": 123})
```

### Dataclasses

| Class                | Purpose                                           |
|----------------------|---------------------------------------------------|
| `EnvironmentConfig`  | SUMO simulation parameters                        |
| `TrainingConfig`     | Training hyperparameters                          |
| `AgentConfig`        | Agent-specific configuration (kind, model path)   |
| `Config`             | Top-level application configuration               |

### CLI Overrides

```bash
traffic-train train --agent dqn --episodes 200 --seed 42
traffic-train train --agent ppo --gui --episodes 100
traffic-train train --agent double_dqn --max-steps 720
```

## YAML Experiment Config

For production training and AWS deployment, use YAML config files:

```bash
# Local development
python -m traffic_rl.cli train --config configs/base.yaml --overlay configs/local.yaml

# CI
python -m traffic_rl.cli train --config configs/base.yaml --overlay configs/ci.yaml

# Production (SageMaker)
python -m traffic_rl.cli train --config configs/base.yaml --overlay configs/prod.yaml
```

### Config Layering

```
configs/base.yaml      ← defaults for every knob
configs/<env>.yaml     ← overlay (local, ci, staging, prod)
.env / env vars        ← highest precedence for deployment knobs
```

### Config Sections

| Section        | Description                                      |
|----------------|--------------------------------------------------|
| `run`          | Experiment name, seed, episodes, schedule        |
| `environment`  | SUMO environment: network, timing, scale         |
| `observation`  | Which features the state vector carries          |
| `reward`       | Reward function weights and shaping              |
| `model`        | Neural architecture (hidden sizes, attention)    |
| `agent`        | Algorithm selection (ppo, dqn, etc.)             |
| `scenario`     | Scenario catalogue, splits, domain randomization |
| `evaluation`   | Multi-seed evaluation protocol and quality gates  |
| `tracking`     | Metrics sinks (jsonl, tensorboard, cloudwatch)   |
| `artifacts`    | Checkpoint retention and S3 upload               |

## Environment Variables

All deployment knobs are resolved from environment variables. See
`.env.example` for the complete reference, and `docs/configuration.md` for
generated documentation (produced from `traffic_rl.settings.describe_settings()`).

### Key Variables

```bash
# Core
TRAFFIC_RL_ENV=local           # local | dev | staging | prod | ci
TRAFFIC_RL_CONFIG=configs/base.yaml

# AWS (production uses OIDC — no static keys needed in CI)
AWS_REGION=us-east-1
AWS_ROLE_ARN=                  # OIDC role for CI/CD
S3_ARTIFACT_BUCKET=traffic-rl-artifacts

# SageMaker
SAGEMAKER_ROLE_ARN=
SAGEMAKER_INSTANCE_TYPE=ml.g5.2xlarge
SAGEMAKER_USE_SPOT=true

# SUMO
SUMO_HOME=/opt/sumo
SUMO_BINARY=/usr/bin/sumo

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Secrets Management

| Store          | Use Case                    |
|----------------|-----------------------------|
| `.env` (local) | Local development secrets   |
| SSM Parameter  | Non-secret config values    |
| Secrets Manager| Credentials, API keys      |
| GitHub OIDC    | CI/CD AWS authentication   |

Secrets are resolved at runtime via `${secret:name}` and `${ssm:/path}`
references in YAML configs. Values are never written to disk.
