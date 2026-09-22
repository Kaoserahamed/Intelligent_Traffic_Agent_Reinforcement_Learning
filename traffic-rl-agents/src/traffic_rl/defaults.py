"""Default configuration factories and override helper."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from traffic_rl.config import (
    AgentConfig,
    Config,
    EnvironmentConfig,
    TrainingConfig,
    project_root,
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

    Supported override keys: ``episodes``, ``save_interval``,
    ``early_stopping_patience``, ``max_steps``, ``gui``, ``seed``,
    ``log_dir``, ``plot_dir``, ``agents``.
    """
    if not overrides:
        return get_default_config()
    base = get_default_config()
    overrides = dict(overrides)

    env = base.environment
    if any(k in overrides for k in ("max_steps", "gui", "seed")):
        env = EnvironmentConfig(
            config_path=env.config_path,
            max_steps=int(overrides.get("max_steps", env.max_steps)),
            gui=bool(overrides.get("gui", env.gui)),
            seed=overrides.get("seed", env.seed),
            step_length=env.step_length,
            min_phase_duration=env.min_phase_duration,
            max_phase_duration=env.max_phase_duration,
            yellow_duration=env.yellow_duration,
        )

    train = base.training
    if any(
        k in overrides
        for k in ("episodes", "save_interval", "early_stopping_patience")
    ):
        train = TrainingConfig(
            episodes=int(overrides.get("episodes", train.episodes)),
            save_interval=int(overrides.get("save_interval", train.save_interval)),
            early_stopping_patience=int(
                overrides.get("early_stopping_patience", train.early_stopping_patience)
            ),
            seed=overrides.get("seed", train.seed),
        )

    agents = base.agents
    if "agents" in overrides:
        agents = tuple(overrides["agents"])

    log_dir = Path(overrides.get("log_dir", base.log_dir))
    plot_dir = Path(overrides.get("plot_dir", base.plot_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return Config(environment=env, training=train, agents=agents, log_dir=log_dir, plot_dir=plot_dir)
