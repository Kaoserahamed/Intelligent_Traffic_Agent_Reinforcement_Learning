# Safety & Security

This document describes the safety mechanisms, security boundaries, and
guardrails built into TrafficRL.

## Model Checkpoint Security

### Neural Network Checkpoints (`.pth`)

All PyTorch checkpoints are loaded with `weights_only=True`:

```python
# Correct — safe loading
checkpoint = torch.load(path, weights_only=True)

# NEVER do this — allows arbitrary code execution
checkpoint = torch.load(path)
```

The `safe_torch_load()` function in `io_utils.py` normalizes all
deserialization failures to `CheckpointError`, so callers never interact
with pickle internals directly. See `src/traffic_rl/io_utils.py`.

### Q-Learning Tables (`.json`)

Tabular Q-tables are persisted as structured JSON with a SHA-256 integrity
checksum:

```json
{
  "format": "json-qtable-v2",
  "q_table": [{"state": [0,1,2], "values": [0.5, -0.3]}],
  "metadata": {
    "format_version": 2,
    "checksum_sha256": "abc123...",
    "saved_at": "2026-01-15T10:30:00+00:00",
    "alpha": 0.1,
    "gamma": 0.95,
    "t": 100
  }
}
```

If the checksum does not match (tampering detected), loading fails with a
`ValueError`. Checksum verification can be disabled only by explicitly
passing `verify_checksum=False`.

### Legacy Pickle Files (`.pkl`)

Legacy Q-table files are migrated through a **restricted unpickler**
(`LegacyPickleUnpickler`) that:

1. Maintains an explicit allow-list of safe builtins (dict, list, int, etc.)
2. Maintains an allow-list of safe module members (OrderedDict, defaultdict)
3. Bans dangerous operations: `system`, `popen`, `eval`, `exec`, `open`,
   `__import__`, `compile`
4. Refuses any class outside the allow-list

Any attempt to load an untrusted class raises `SafetyError`.

## Runtime Safety

TrafficRL controls **simulated** traffic signals. The following guardrails
ensure models cannot be accidentally deployed to a real-world system:

- **No real-world actuation**: The environment only connects to SUMO
  simulations via `traci`. No hardware, IoT, or real-world API integrations
  exist.
- **Evaluation gates**: Production config enforces quality gates before
  model promotion:
  - Minimum 10% improvement over fixed-time baseline
  - Average delay ≤ 45 seconds
  - 95th percentile queue ≤ 25 vehicles
- **Multi-seed evaluation**: Models are evaluated across 5 random seeds.
- **Domain randomization**: Training includes weather, incidents, and sensor
  noise to prevent overfitting to specific conditions.

## Log Security

Structured logging includes redaction filters that scrub credentials:

- **Key-based redaction**: Sensitive keys (`password`, `api_key`, `token`,
  `secret`, etc.) are masked regardless of nesting.
- **Pattern-based redaction**: AWS access keys, bearer tokens, and
  `Authorization: Bearer ...` patterns are masked via regex.
- **Secret values**: When secrets are resolved from AWS, their plaintext
  values are added to the redaction filter.

## Dependency Security

- **Dependabot** is enabled for GitHub Actions and Python dependencies.
- **pip-audit** checks the resolved dependency tree against CVE databases.
- **Bandit** performs static analysis for common Python security issues.
- All third-party dependencies are pinned with minimum version constraints.

## AWS Security

- **OIDC federation**: CI/CD authenticates to AWS via GitHub OIDC — no
  long-lived keys. Static keys are for local development only.
- **IAM least privilege**: CloudFormation templates define minimal IAM policies.
- **S3 encryption**: Artefact buckets use SSE-KMS with customer-managed keys.
- **Secrets Manager**: Credentials are stored in AWS Secrets Manager, never
  committed to the repository.

## What NOT to Do

| ❌ Do Not | ✅ Do Instead |
|-----------|---------------|
| Commit `.env` file | Use `.env.example` as a template |
| Hardcode credentials | Use `${secret:name}` references |
| Load checkpoints without `weights_only=True` | Use `safe_torch_load()` |
| Run untrained models in "production-like" settings | Validate against baselines first |
| Log raw metrics with PII | Use structured logging with redaction |
