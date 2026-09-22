"""Runtime configuration for TrafficRL.

Configuration is expressed through frozen, validated dataclasses so that
mistakes fail loudly at construction time.  See :func:`get_default_config`
in :mod:`traffic_rl.defaults` for the default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

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

#: Simulated seconds between two agent decisions ("control interval").
#  Historic name kept for backwards compatibility.  It used to be treated as
#  *milliseconds* by the environment, which silently shrank every episode to
#  150 simulated seconds.
STEP_LENGTH: int = 5

#: SUMO integration step (``--step-length``) in simulated seconds.
SIM_STEP_S: float = 1.0

#: Default episode horizon expressed in agent decisions (720 x 5 s = 1 hour).
DEFAULT_MAX_STEPS: int = 720

#: Valid SUMO ``--collision.action`` values.
COLLISION_ACTIONS: tuple[str, ...] = ("none", "warn", "teleport", "remove")

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
    """Immutable configuration for a SUMO traffic environment.

    Timing model
    ------------
    ``sim_step_s``
        SUMO integration step (``--step-length``), typically 1.0 s.
    ``step_length``
        Simulated seconds between two agent decisions (control interval).
    ``max_steps``
        Number of agent decisions per episode.
    ``episode_seconds``
        ``max_steps * step_length``; passed to SUMO as ``--end``.
    ``scale``
        SUMO ``--scale`` factor used for demand/weather scaling (1.0 = nominal).
    """

    config_path: Path | str = field(default_factory=default_config_path)
    max_steps: int = DEFAULT_MAX_STEPS
    gui: bool = False
    seed: int | None = None
    step_length: int = STEP_LENGTH
    sim_step_s: float = SIM_STEP_S
    min_phase_duration: int = MIN_PHASE_DURATION
    max_phase_duration: int = MAX_PHASE_DURATION
    yellow_duration: int = YELLOW_DURATION
    #: SUMO ``--scale``: multiplier applied to all demand (weather/incidents).
    scale: float = 1.0
    #: Seconds a vehicle may be stuck before SUMO teleports it (jam breaker).
    time_to_teleport: int = 300
    #: One of :data:`COLLISION_ACTIONS`.
    collision_action: str = "teleport"
    #: End the episode early when no vehicles are left in the network.
    end_when_empty: bool = True
    #: Optional SUMO output files; when set, the environment records them in
    #: the run artefacts (used for real-world KPI extraction).
    tripinfo_output: Path | None = None
    summary_output: Path | None = None
    #: Extra ``-a/--additional-files`` entries (detectors, weather, incidents).
    additional_files: tuple[Path, ...] = ()
    #: Working directory for relative SUMO files (scenario runs).
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(
            self, "additional_files", tuple(Path(p) for p in self.additional_files)
        )
        for name in ("tripinfo_output", "summary_output", "run_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}")
        if self.sim_step_s <= 0:
            raise ValueError(f"sim_step_s must be positive, got {self.sim_step_s}")
        if self.step_length < self.sim_step_s:
            raise ValueError(
                f"step_length ({self.step_length}) must be >= sim_step_s ({self.sim_step_s})"
            )
        if self.min_phase_duration > self.max_phase_duration:
            raise ValueError(
                f"min_phase_duration ({self.min_phase_duration}) must be"
                f" <= max_phase_duration ({self.max_phase_duration})"
            )
        if self.yellow_duration < 0:
            raise ValueError(f"yellow_duration must be non-negative, got {self.yellow_duration}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.time_to_teleport < 0:
            raise ValueError(
                f"time_to_teleport must be non-negative, got {self.time_to_teleport}"
            )
        if self.collision_action not in COLLISION_ACTIONS:
            raise ValueError(
                f"collision_action {self.collision_action!r} must be one of {COLLISION_ACTIONS}"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"SUMO config not found: {self.config_path}. Ensure the `sim/`"
                " directory is present in the repository root."
            )

    # -- derived timing helpers ------------------------------------------------

    @property
    def decision_interval_s(self) -> int:
        """Preferred alias for :attr:`step_length` (seconds per decision)."""
        return self.step_length

    @property
    def episode_seconds(self) -> float:
        """Simulated wall-clock horizon of one episode, in seconds."""
        return float(self.max_steps) * float(self.step_length)

    @property
    def sub_steps(self) -> int:
        """Number of SUMO integration steps executed per agent decision."""
        return max(1, int(round(self.step_length / self.sim_step_s)))


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
            raise ValueError("agent.name must be non-empty")
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
