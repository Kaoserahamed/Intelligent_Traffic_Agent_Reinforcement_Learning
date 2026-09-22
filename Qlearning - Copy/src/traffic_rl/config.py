"""Runtime configuration for TrafficRL.

Configuration is expressed through frozen, validated dataclasses so that
mistakes fail loudly at construction time.  See :func:`get_default_config`
in :mod:`traffic_rl.defaults` for the default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# -----------------------------------------------------------------------------
# Domain constants — mirror the SUMO network in ``sim/``.
# -----------------------------------------------------------------------------

TL_ID: str = "1"  #: traffic-light junction id in the SUMO network.

INCOMING_EDGES: tuple[str, ...] = ("5to1", "2to1", "3to1", "4to1")  # S, E, N, W
OUTGOING_EDGES: tuple[str, ...] = ("1to5", "1to2", "1to3", "1to4")

#: Vehicle types from ``intersection.rou.xml``; order defines the state layout.
VEHICLE_TYPES: tuple[str, ...] = ("car", "bus", "emergency", "truck", "bike")

#: SUMO green phases (one per approach) for ``programID="1"`` in add.xml.
#  Legacy code used ``[0, 2, 4, 6]`` which targeted all-red phases; corrected.
GREEN_PHASES: tuple[int, ...] = (0, 3, 6, 9)

MIN_PHASE_DURATION: int = 10
MAX_PHASE_DURATION: int = 60
YELLOW_DURATION: int = 3
STEP_LENGTH: int = 30

#: 20 (vehicle counts) + 4 (queues) + 4 (densities) = 28.
STATE_SIZE: int = len(VEHICLE_TYPES) * len(INCOMING_EDGES) + len(INCOMING_EDGES) * 2
ACTION_SIZE: int = len(GREEN_PHASES)

DEFAULT_TIME_ALLOCATIONS: Mapping[str, float] = MappingProxyType(
    {"car": 2.0, "bus": 3.5, "emergency": 1.5, "truck": 4.0, "bike": 1.0}
)


def project_root() -> Path:
    """Return the repository root (the parent of ``src/``).

    Works correctly when the package is installed in development mode
    (``pip install -e .``) or when run from the source tree.
    """
    return Path(__file__).resolve().parent.parent.parent


def default_sim_dir() -> Path:
    return project_root() / "sim"


def default_config_path(name: str = "intersection.sumocfg") -> Path:
    return default_sim_dir() / name


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentConfig:
    """Immutable configuration for a SUMO traffic environment."""

    config_path: Path | str = field(default_factory=default_config_path)
    max_steps: int = 5000
    gui: bool = False
    seed: int | None = None
    step_length: int = STEP_LENGTH
    min_phase_duration: int = MIN_PHASE_DURATION
    max_phase_duration: int = MAX_PHASE_DURATION
    yellow_duration: int = YELLOW_DURATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}")
        if self.min_phase_duration > self.max_phase_duration:
            raise ValueError(
                f"min_phase_duration ({self.min_phase_duration}) must be"
                f" <= max_phase_duration ({self.max_phase_duration})"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"SUMO config not found: {self.config_path}. Ensure the `sim/`"
                " directory is present in the repository root."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters governing a single agent training run."""

    episodes: int = 200
    save_interval: int = 25
    early_stopping_patience: int = 100
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}"
            )


@dataclass(frozen=True)
class AgentConfig:
    """Registry entry describing how to instantiate a single agent."""

    name: str
    kind: str  # "q_learning" | "dqn" | "double_dqn" | "ppo" | "fixed"
    model_path: Path
    hyperparams: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    epsilon: float | None = None  # only relevant for evaluation / loading

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("agent.name must be a non-empty string")
        if self.kind not in {"q_learning", "dqn", "double_dqn", "ppo", "fixed"}:
            raise ValueError(
                f"unknown agent kind {self.kind!r}; expected one of "
                "q_learning, dqn, double_dqn, ppo, fixed"
            )
        if self.epsilon is not None and not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")


@dataclass(frozen=True)
class Config:
    """Top-level application configuration."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    agents: Sequence[AgentConfig] = field(default_factory=tuple)
    log_dir: Path = field(default_factory=lambda: project_root() / "logs")
    plot_dir: Path = field(default_factory=lambda: project_root() / "plots")

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", tuple(self.agents))
        if not self.agents:
            raise ValueError("Config.agents must contain at least one agent")
        for agent in self.agents:
            if not str(agent.model_path):
                raise ValueError(f"agent {agent.name!r} has an empty model_path")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def agent_dir(self, name: str) -> Path:
        """Return (and create) the directory used to store an agent's artefacts."""
        safe = name.replace(" ", "_").replace("/", "-")
        directory = self.log_dir / safe
        directory.mkdir(parents=True, exist_ok=True)
        return directory
