# Architecture

This document describes the system architecture of TrafficRL and the design
decisions behind its components.

## High-Level Design

TrafficRL follows a **layered architecture** where each layer has a single
responsibility and can be tested in isolation:

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Entry Point                 │
│          (traffic_rl.cli, traffic-rain)             │
├─────────────────────────────────────────────────────┤
│                 Training Loop                        │
│     (cli._cmd_train, agent.learn)                    │
├─────────────────────────────────────────────────────┤
│              RL Agents (Pluggable)                   │
│  Q-Learning │ DQN │ Double DQN │ PPO                  │
├─────────────────────────────────────────────────────┤
│          Environment Interface                       │
│     (traffic_rl.sumo.environment)                    │
├─────────────────────────────────────────────────────┤
│           Metrics / State / Reward                  │
│      (traffic_rl.metrics — pure functions)           │
├─────────────────────────────────────────────────────┤
│               SUMO Simulation Engine                │
│        (traci / sumo binary via sim/*.xml)           │
└─────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns

- **Pure functions for metrics**: `metrics.py` contains no I/O and no external
  dependencies (only NumPy). Every function takes already-queried simulator
  data and returns plain Python data structures. This makes the logic
  trivially unit-testable without a SUMO installation.

- **Environment wrapper**: `sumo/environment.py` translates raw `traci` calls
  into a `WorldSnapshot` dataclass and delegates to `metrics.py`. This means
  the core RL logic never calls `traci` directly.

- **Agent interface**: All agents implement `BaseAgent`, so the training loop
  is agnostic to the algorithm. Switching from DQN to PPO requires only a CLI
  flag change.

### 2. Security by Design

- **No pickle for Q-tables**: Tabular agents persist to JSON with SHA-256
  checksums.
- **`weights_only=True` for torch**: Neural checkpoints use
  `torch.load(..., weights_only=True)` to prevent arbitrary code execution.
- **Restricted unpickler**: Legacy `.pkl` files are migrated through an
  allow-list unpickler that refuses any class outside a safe set.

### 3. Reproducibility

- **Seeding**: `seed_everything()` seeds Python, NumPy, and PyTorch RNGs.
- **Config hashing**: YAML configs are hashed via SHA-256 for every run.
- **Immutable configs**: Frozen dataclasses ensure config cannot change mid-run.

### 4. Observability

- **Context propagation**: `contextvars.ContextVar` carries run_id, agent,
  scenario, episode across threads and async tasks.
- **Structured logging**: JSON-formatted log records with a defined schema.
- **Metrics sinks**: Pluggable sinks (JSONL, TensorBoard, CloudWatch,
  SageMaker) with per-sink error isolation.
- **Profiling**: `PhaseTimer` and `StepRateMeter` answer "where does wall
  time go?" and "how many steps/second?".

## Component Deep Dive

### State Vector (28 dimensions)

The observation space encodes traffic conditions across 4 incoming edges:

- 5 normalized vehicle-type counts per edge (cars, buses, emergency, trucks, bikes)
- 1 normalized queue length per edge
- 1 normalized density per edge

Total: (5 + 1 + 1) × 4 = 28 features.

### Reward Function

The reward is a multi-objective function designed to optimize traffic flow:

| Component              | Weight | Description                          |
|------------------------|--------|--------------------------------------|
| Queue improvement      | 10.0   | Reduction in total queue length     |
| Waiting time reduction | 0.1    | Reduction in total waiting time      |
| Speed improvement      | 5.0    | Increase in average speed            |
| Throughput             | 20.0   | Vehicles that exited the network    |
| Phase switch penalty   | 5.0    | Discourages excessive phase changes  |
| Queue penalty          | 2.0    | For queues exceeding 15 vehicles     |
| Queue imbalance        | 2.0    | Penalizes unfairness across approaches |
| Low speed penalty      | 20.0   | When avg speed < 2.0 m/s             |
| Emergency priority     | 2.0    | Bonus for serving emergency vehicles |

Reward is clamped to [-100, 100] for training stability.

### Phase Control Model

The SUMO network has 4 green phases (one per approach). The agent selects
which phase to activate. Phase switches are rejected if the current green
phase is younger than `min_phase_duration` (default 10 seconds) to prevent
the "chattering" problem in real-world deployments.

## Deployment Architecture

### Local Development

```
CLI → Config → Environment (SUMO) → Agent learn() → save()
```

### AWS Production

```
SageMaker Training Job
  ├── Docker Container (ECR image)
  │   ├── configs/prod.yaml (loaded at startup)
  │   ├── sim/ (baked into image)
  │   ├── src/traffic_rl/ (baked into image)
  │
  ├── Secrets from AWS Secrets Manager / SSM Parameter Store
  ├── Artifacts uploaded to S3 (traffic-rl-artifacts bucket)
  ├── Metrics streamed to CloudWatch
  └── Logs to CloudWatch Logs (/traffic-rl/prod)
```

### CI/CD Pipeline

```
GitHub Push → CI Workflow
  ├── Lint (ruff)
  ├── Type Check (mypy)
  ├── Security (pip-audit, bandit)
  ├── Tests (pytest with structured report)
  └── (on tag) → CD Workflow
      ├── Build & Push Docker Image (ECR)
      ├── Create GitHub Release
      └── Deploy to SageMaker
```
