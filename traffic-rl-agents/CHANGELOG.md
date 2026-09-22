# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Observability package** (`traffic_rl.observability`): structured JSON/console logging,
  metrics sinks (JSONL, TensorBoard, SageMaker, CloudWatch), context propagation, profiling.
- **Configuration system** (`traffic_rl.config_loader`): YAML experiment config with layering,
  pydantic validation, reference resolution, and config hashing for reproducibility.
- **Settings management** (`traffic_rl.settings`): typed environment-based settings using pydantic-settings.
- **Structured test reporting**: pytest plugin emitting JSON reports with environment metadata.
- **YAML config files**: base, ci, local, staging, prod, logging configurations.
- **Environment template** (`.env.example`): complete reference of all environment variables.
- **CI/CD workflows**: linting, type checking, testing, security scanning, and release pipelines.
- **AWS infrastructure**: CloudFormation templates for ECR, IAM, and SageMaker resources.
- **Comprehensive documentation**: architecture, API reference, configuration, models, and safety docs.
- Reinforcement learning algorithms: Q-Learning, DQN, Double DQN, and PPO.
- Secure model persistence with JSON Q-tables and `weights_only=True` for PyTorch checkpoints.
- Restricted unpickler for legacy `.pkl` Q-tables (security boundary).

### Security

- All model checkpoints loaded with `torch.load(weights_only=True)`.
- Q-tables persisted as JSON with SHA-256 integrity verification.
- Legacy pickle files use an allow-list restricted unpickler.
- Sensitive values redacted from structured logs via regex patterns and resolved secrets.
