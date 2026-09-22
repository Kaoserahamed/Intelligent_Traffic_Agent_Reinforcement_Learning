# Security Policy

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report them to the maintainers via one of:

- Open a **private** security advisory on GitHub: https://github.com/Kaoserahamed/AI/security/advisories
- Email the maintainers directly

We will acknowledge receipt within 24 hours and provide a more detailed
response within 72 hours. We strive to fix all confirmed vulnerabilities
promptly and will publish a patch release as soon as a fix is available.

## Security Boundaries

### Model Checkpoint Loading

- **Neural network checkpoints** (`.pth`): loaded exclusively with
  `torch.load(..., weights_only=True)`. This prevents arbitrary code execution
  from tampered checkpoint files. The `weights_only=True` flag restricts
  deserialization to primitive types and tensors only.

- **Q-Learning tables** (`.json`): persisted as structured JSON with a
  SHA-256 integrity checksum. If the checksum does not match, the checkpoint
  is rejected. No pickle is used for tabular agents.

- **Legacy Q-tables** (`.pkl`): migrated through a restricted unpickler
  (`LegacyPickleUnpickler`) that uses an explicit allow-list of builtins and
  container types. Any class outside the allow-list raises `SafetyError`.

### Runtime Safety

This project implements reinforcement learning for **traffic signal control**.
Key safety considerations:

- **No real-world actuation**: agents make decisions in a *simulated* SUMO
  environment. Models are never deployed to control real-world traffic signals
  without rigorous validation in a staging environment.
- **Evaluation gates**: production config (`configs/prod.yaml`) enforces
  quality gates including minimum improvement over fixed-time baselines,
  maximum average delay, and maximum 95th-percentile queue length.
- **Multi-seed evaluation**: agents are evaluated across multiple random seeds
  to ensure robustness.
- **Domain randomization**: training includes demand scaling, weather, incidents,
  and sensor noise to improve generalization.
- **Emergency vehicle priority**: the reward function includes a 2x weight for
  emergency vehicle priority (`emergency_priority: 2.0`).

### Network Security

- All AWS interactions use **OIDC federation** — no long-lived AWS keys in CI.
- Secrets are stored in **AWS Secrets Manager** or **SSM Parameter Store**,
  never committed to the repository.
- A `.env.example` file documents all variables; `.env` is git-ignored.
- Log redaction filters scrub credentials from all log output.

### Dependency Monitoring

- **Dependabot** is enabled for both GitHub Actions and Python dependencies.
- **pip-audit** runs in CI to detect known CVEs in the dependency tree.
- **Bandit** performs static analysis for common Python security issues.

## AI-Specific Guardrails

Since this project trains RL agents, the following guardrails are in place:

- **Evaluation gates**: no model is promoted to production unless it meets
  strict performance thresholds (see `configs/prod.yaml`).
- **Reproducibility**: every run is seeded and the config is hashed for
  auditability.
- **Observability**: all metrics, events, and parameters are logged in
  structured JSON with correlation IDs.
- **Rollback**: model registry supports version pinning and rollback to any
  previously released checkpoint.

## What to Include in Vulnerability Reports

1. Description of the vulnerability
2. Steps to reproduce (or proof-of-concept)
3. Potential impact
4. Suggested fix (if any)

Thank you for helping keep TrafficRL secure!
