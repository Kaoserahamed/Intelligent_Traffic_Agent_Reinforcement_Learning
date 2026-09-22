"""Default configuration factories and override helper."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from traffic_rl.config import (
    AgentConfig,
    Config,
    EnvironmentConfig,
    TrainingConfig,
    project_root,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

#: Override keys accepted by :func:`resolve_config` for the environment block.
_ENVIRONMENT_OVERRIDE_KEYS: tuple[str, ...] = (
    "config_path",
    "max_steps",
    "gui",
    "seed",
    "step_length",
    "sim_step_s",
    "min_phase_duration",
    "max_phase_duration",
    "yellow_duration",
    "scale",
    "tripinfo_output",
    "summary_output",
)

#: Override keys accepted by :func:`resolve_config` for the training block.
_TRAINING_OVERRIDE_KEYS: tuple[str, ...] = (
    "episodes",
    "save_interval",
    "early_stopping_patience",
    "seed",
)


def _default_agents(log_dir: Path) -> Sequence[AgentConfig]:
    """Build the default agent registry used by the CLI."""
    base = log_dir
    return [
        AgentConfig(
            name="Q-Learning",
            kind="q_learning",
            model_path=base / "Q-Learning" / "best_q_model.json",
            hyperparams={"alpha": 0.1, "gamma": 0.95, "epsilon_decay": 0.995, "epsilon_min": 0.01},
        ),
        AgentConfig(
            name="DQN",
            kind="dqn",
            model_path=base / "DQN" / "best_dqn_model.pth",
            hyperparams={"lr": 0.001, "gamma": 0.95, "epsilon_decay": 0.995, "epsilon_min": 0.01},
        ),
        AgentConfig(
            name="DoubleDQN",
            kind="double_dqn",
            model_path=base / "DoubleDQN" / "best_double_dqn_model.pth",
            hyperparams={"lr": 0.001, "gamma": 0.99, "epsilon_decay": 0.995, "epsilon_min": 0.01},
        ),
        AgentConfig(
            name="PPO",
            kind="ppo",
            model_path=base / "PPO" / "best_ppo_model.pth",
            hyperparams={"lr": 3e-4, "gamma": 0.99, "clip_epsilon": 0.2},
        ),
    ]


@lru_cache(maxsize=1)
def get_default_config() -> Config:
    """Return the application's default, validated :class:`Config`.

    Cached because construction touches the filesystem (it ensures the log and
    plot directories exist).
    """
    log_dir = project_root() / "logs"
    return Config(
        environment=EnvironmentConfig(),
        training=TrainingConfig(),
        agents=_default_agents(log_dir),
        log_dir=log_dir,
        plot_dir=project_root() / "plots",
    )


def resolve_config(overrides: Mapping[str, Any] | None = None) -> Config:
    """Build a :class:`Config` applying optional overrides.

    Supported override keys:

    * environment: ``config_path``, ``max_steps``, ``gui``, ``seed``,
      ``step_length``, ``sim_step_s``, ``min_phase_duration``,
      ``max_phase_duration``, ``yellow_duration``, ``scale``,
      ``tripinfo_output``, ``summary_output``
    * training: ``episodes``, ``save_interval``, ``early_stopping_patience``,
      ``seed``
    * top level: ``log_dir``, ``plot_dir``, ``agents``

    Values are applied with :func:`dataclasses.replace`, so *every* supported
    field can be overridden independently and validation still runs exactly
    once.  (Previously ``{"seed": ...}`` alone was silently ignored because the
    training block was only rebuilt when one of the schedule keys changed.)
    """
    base = get_default_config()
    if not overrides:
        return base
    values = dict(overrides)

    environment = base.environment
    env_updates = {k: values[k] for k in _ENVIRONMENT_OVERRIDE_KEYS if k in values}
    if env_updates:
        environment = replace(base.environment, **env_updates)

    training = base.training
    train_updates = {k: values[k] for k in _TRAINING_OVERRIDE_KEYS if k in values}
    if train_updates:
        training = replace(base.training, **train_updates)

    agents = tuple(values["agents"]) if "agents" in values else base.agents
    log_dir = Path(values.get("log_dir", base.log_dir))
    plot_dir = Path(values.get("plot_dir", base.plot_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        environment=environment,
        training=training,
        agents=agents,
        log_dir=log_dir,
        plot_dir=plot_dir,
    )
