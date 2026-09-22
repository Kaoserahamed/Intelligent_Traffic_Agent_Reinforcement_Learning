"""YAML experiment configuration: load, overlay, validate, hash.

Layering
--------
``configs/base.yaml``            defaults for every knob
``configs/<env>.yaml``           overlay (``local``, ``ci``, ``dev``, ``staging``, ``prod``)
environment variables             highest precedence for deployment knobs
``${ssm:/path}`` / ``${secret:name}``  resolved at load time (never stored on disk)

Everything is validated with pydantic models that use ``extra="forbid"``, so a
typo in a config file fails *immediately* in CI instead of silently training the
wrong thing.  :func:`config_hash` gives every run a stable fingerprint of the
effective configuration, which is what the artefacts and model registry store.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from traffic_rl.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

log = get_logger("traffic_rl.config_loader")

#: Matches ``${ssm:/path}``, ``${secret:name}`` and ``${env:NAME}`` references.
_REFERENCE_RE = re.compile(r"\$\{(ssm|secret|env):([^}]+)\}")

ReferenceResolver = Callable[[str, str], str]


class _StrictModel(BaseModel):
    """Base for every config section: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def project_configs_dir() -> Path:
    """Return the repository's ``configs/`` directory."""
    from traffic_rl.config import project_root

    return project_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path* (``{}`` for an empty file)."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {resolved} must contain a mapping at the top level")
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base* (overlay wins, lists replace)."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def resolve_references(
    value: Any,
    resolver: ReferenceResolver | None = None,
    *,
    _seen: tuple[str, ...] = (),
) -> Any:
    """Replace ``${ssm:...}``/``${secret:...}``/``${env:...}`` references.

    ``resolver`` is injected by :mod:`traffic_rl.cloud` (AWS Secrets Manager /
    SSM Parameter Store).  When it is ``None`` (offline runs, unit tests) only
    ``${env:NAME}`` is resolvable and cloud references are left untouched, so a
    local run can never silently depend on AWS.
    """
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            kind, name = match.group(1), match.group(2)
            if kind == "env":
                import os

                resolved = os.environ.get(name)
                if resolved is None:
                    raise KeyError(f"environment variable {name!r} is not set (referenced in config)")
                return resolved
            if resolver is None:
                log.debug("leaving %s reference %s unresolved (no resolver configured)", kind, name)
                return match.group(0)
            return resolver(kind, name)

        return _REFERENCE_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {key: resolve_references(item, resolver, _seen=_seen) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_references(item, resolver, _seen=_seen) for item in value]
    return value


def load_experiment_config(
    path: str | Path | None = None,
    *,
    overlay: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    resolver: ReferenceResolver | None = None,
) -> ExperimentConfig:
    """Load, layer, resolve and validate an experiment configuration.

    Parameters
    ----------
    path:
        Base config (defaults to ``configs/base.yaml`` in the repository).
    overlay:
        Optional environment overlay (``configs/ci.yaml``, ``configs/prod.yaml``).
    overrides:
        Final in-memory overrides (CLI flags, SageMaker hyper-parameters).
    resolver:
        Callable resolving ``${ssm:...}``/``${secret:...}`` references.
    """
    base_path = Path(path) if path else project_configs_dir() / "base.yaml"
    data = load_yaml(base_path)
    if overlay is not None:
        data = deep_merge(data, load_yaml(overlay))
    if overrides:
        data = deep_merge(data, dict(overrides))
    resolved = resolve_references(data, resolver)
    config = ExperimentConfig.model_validate(resolved)
    log.debug("loaded experiment config from %s (hash=%s)", base_path, config.config_hash())
    return config


def config_hash(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash of a resolved configuration."""
    payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunSection(_StrictModel):
    """Run identity and training schedule."""

    experiment: str = "traffic-rl"
    seed: int = 42
    episodes: int = 200
    save_interval: int = 25
    early_stopping_patience: int = 100
    checkpoint_keep_last: int = 3
    resume_from: str | None = None

    @field_validator("episodes", "save_interval", "checkpoint_keep_last")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class EnvironmentSection(_StrictModel):
    """SUMO environment settings (mirrors :class:`traffic_rl.config.EnvironmentConfig`)."""

    network: str = "sim/intersection.sumocfg"
    max_steps: int = 720
    step_length: int = 5
    sim_step_s: float = 1.0
    min_phase_duration: int = 10
    max_phase_duration: int = 60
    yellow_duration: int = 3
    scale: float = 1.0
    end_when_empty: bool = True
    time_to_teleport: int = 300
    collision_action: str = "teleport"
    gui: bool = False
    tripinfo_output: str | None = None
    summary_output: str | None = None

    @property
    def episode_seconds(self) -> float:
        return float(self.max_steps) * float(self.step_length)


class ObservationSection(_StrictModel):
    """Which features the observation vector carries."""

    include_vehicle_mix: bool = True
    include_pressure: bool = True
    include_time_of_day: bool = True
    include_phase: bool = True
    include_detector_mask: bool = True
    temporal_stack: int = 1
    normalize: bool = True


class RewardWeights(_StrictModel):
    """Multi-objective reward weights (documented in ``docs/models.md``)."""

    delay: float = 1.0
    queue: float = 0.5
    pressure: float = 0.3
    throughput: float = 1.0
    switch_penalty: float = 0.2
    emissions: float = 0.1
    fairness: float = 0.1
    emergency_priority: float = 2.0


class RewardSection(_StrictModel):
    """Reward composition."""

    clip: float = 100.0
    low_speed_threshold: float = 2.0
    weights: RewardWeights = Field(default_factory=RewardWeights)


class ModelSection(_StrictModel):
    """Neural architecture selection (see :mod:`traffic_rl.models.registry`)."""

    name: str = "attention_actor_critic"
    hidden_sizes: tuple[int, ...] = (256, 256)
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.0
    layer_norm: bool = True
    noisy: bool = False


class AgentSection(_StrictModel):
    """Algorithm selection and hyper-parameters (see :mod:`traffic_rl.agents`)."""

    kind: str = "ppo"
    hyperparams: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("agent.kind must be a non-empty string")
        return value


class SplitsSection(_StrictModel):
    """Train/validation/test proportions for the scenario library."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @field_validator("train", "val", "test")
    @classmethod
    def _in_unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("split fractions must be within [0, 1]")
        return value

    def normalised(self) -> dict[str, float]:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("split fractions must sum to a positive value")
        return {"train": self.train / total, "val": self.val / total, "test": self.test / total}


class DomainRandomizationSection(_StrictModel):
    """Stochastic scenario perturbations applied during training."""

    enabled: bool = True
    demand_scale: tuple[float, float] = (0.7, 1.3)
    vehicle_mix_jitter: float = 0.2
    weather_probability: float = 0.2
    incident_probability: float = 0.1
    sensor_noise_probability: float = 0.1
    seed: int | None = None

    @field_validator("demand_scale")
    @classmethod
    def _ordered_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low <= 0 or high < low:
            raise ValueError("demand_scale must be a positive, ordered (min, max) pair")
        return (low, high)


class ScenarioSection(_StrictModel):
    """Scenario catalogue selection and generation."""

    catalog: str = "configs/scenarios"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    splits: SplitsSection = Field(default_factory=SplitsSection)
    unseen_holdout: bool = True
    regenerate: bool = False
    domain_randomization: DomainRandomizationSection = Field(
        default_factory=DomainRandomizationSection
    )


class GatesSection(_StrictModel):
    """Quality gates evaluated by the CD pipeline before promotion."""

    min_improvement_vs_fixed_time: float = 0.0
    max_avg_delay_s: float = 0.0
    max_p95_queue: float = 0.0


class EvaluationSection(_StrictModel):
    """Evaluation protocol (multi-seed, multi-split, with baselines)."""

    episodes: int = 20
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    splits: list[str] = Field(default_factory=lambda: ["val", "test"])
    baselines: list[str] = Field(
        default_factory=lambda: ["fixed_time", "actuated", "max_pressure"]
    )
    gates: GatesSection = Field(default_factory=GatesSection)


class TrackingSection(_StrictModel):
    """Metrics sinks and upload behaviour."""

    sinks: list[str] = Field(default_factory=lambda: ["jsonl"])
    log_every_episode: bool = True
    upload_to_s3: bool = True


class ArtifactsSection(_StrictModel):
    """Checkpoint and artefact retention."""

    upload_to_s3: bool = True
    keep_local_runs: int = 20


class ExperimentConfig(_StrictModel):
    """Fully resolved, validated experiment configuration."""

    version: int = 1
    run: RunSection = Field(default_factory=RunSection)
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
    observation: ObservationSection = Field(default_factory=ObservationSection)
    reward: RewardSection = Field(default_factory=RewardSection)
    model: ModelSection = Field(default_factory=ModelSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    scenario: ScenarioSection = Field(default_factory=ScenarioSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    tracking: TrackingSection = Field(default_factory=TrackingSection)
    artifacts: ArtifactsSection = Field(default_factory=ArtifactsSection)

    # -- derived helpers -------------------------------------------------------
    def config_hash(self) -> str:
        """Return the SHA-256 fingerprint of the effective configuration."""
        return config_hash(self)

    def sink_list(self) -> list[str]:
        """Return the configured metrics sinks as a flat list."""
        return [name.strip().lower() for name in self.tracking.sinks if name.strip()]

    def summary(self) -> dict[str, Any]:
        """Return a compact, log-friendly summary of the run."""
        return {
            "experiment": self.run.experiment,
            "agent": self.agent.kind,
            "model": self.model.name,
            "episodes": self.run.episodes,
            "episode_seconds": self.environment.episode_seconds,
            "scenarios": self.scenario.include or ["<catalog default>"],
            "seed": self.run.seed,
            "config_hash": self.config_hash(),
        }