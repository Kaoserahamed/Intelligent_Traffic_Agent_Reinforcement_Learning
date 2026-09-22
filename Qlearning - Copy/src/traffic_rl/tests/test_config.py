"""Tests for configuration dataclasses and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_rl.config import (
    AgentConfig,
    Config,
    EnvironmentConfig,
    TrainingConfig,
    default_config_path,
)
from traffic_rl.defaults import get_default_config, resolve_config


class TestEnvironmentConfig:
    def test_default_config_path_exists(self):
        path = default_config_path()
        assert path.exists(), f"expected default sim at {path}"

    def test_invalid_max_steps_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            EnvironmentConfig(max_steps=0)

    def test_invalid_step_length_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            EnvironmentConfig(step_length=-1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="must be"):
            EnvironmentConfig(min_phase_duration=100, max_phase_duration=50)

    def test_nonexistent_config_path_raises(self, tmp_path: Path):
        bad = tmp_path / "missing.sumocfg"
        with pytest.raises(FileNotFoundError):
            EnvironmentConfig(config_path=bad)

    def test_valid_config_path_accepts_string(self, config_path: Path):
        cfg = EnvironmentConfig(config_path=str(config_path))
        assert cfg.config_path == config_path


class TestTrainingConfig:
    def test_invalid_episodes_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            TrainingConfig(episodes=0)

    def test_invalid_save_interval_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            TrainingConfig(save_interval=0)

    def test_negative_patience_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            TrainingConfig(early_stopping_patience=-1)


class TestAgentConfig:
    def test_valid_agent(self):
        cfg = AgentConfig(name="test", kind="dqn", model_path=Path("/tmp/model.pth"))
        assert cfg.kind == "dqn"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="unknown agent kind"):
            AgentConfig(name="x", kind="invalid", model_path=Path("/tmp/model.pth"))

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon must be in"):
            AgentConfig(name="x", kind="dqn", model_path=Path("/tmp/model.pth"),
                        epsilon=1.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="agent.name must be non-empty"):
            AgentConfig(name="", kind="dqn", model_path=Path("/tmp/model.pth"))


class TestConfig:
    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="Config.agents must contain at least"):
            Config(environment=EnvironmentConfig(), agents=[])

    def test_agents_must_be_tuple(self, config_path: Path):
        cfg = Config(
            environment=EnvironmentConfig(config_path=config_path),
            agents=[AgentConfig(name="test", kind="dqn", model_path=Path("/tmp/model.pth"))],
        )
        assert isinstance(cfg.agents, tuple)


class TestDefaultConfig:
    def test_get_default_config_returns_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, Config)
        assert len(cfg.agents) >= 4  # Q-Learning, DQN, DoubleDQN, PPO

    def test_default_log_and_plot_dirs_created(self, tmp_path: Path):
        base = get_default_config()
        cfg = resolve_config({"log_dir": tmp_path / "logs", "plot_dir": tmp_path / "plots"})
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "plots").exists()


class TestResolveConfig:
    def test_no_overrides_returns_default(self):
        a = get_default_config()
        b = resolve_config()
        assert a.environment.max_steps == b.environment.max_steps
        assert a.training.episodes == b.training.episodes

    def test_episodes_override(self):
        cfg = resolve_config({"episodes": 500})
        assert cfg.training.episodes == 500

    def test_max_steps_override(self):
        cfg = resolve_config({"max_steps": 10000})
        assert cfg.environment.max_steps == 10000

    def test_gui_override(self):
        cfg = resolve_config({"gui": True})
        assert cfg.environment.gui is True

    def test_seed_override(self):
        cfg = resolve_config({"seed": 123})
        assert cfg.training.seed == 123

    def test_multiple_overrides(self):
        cfg = resolve_config({"episodes": 300, "max_steps": 2000, "seed": 42})
        assert cfg.training.episodes == 300
        assert cfg.environment.max_steps == 2000
        assert cfg.training.seed == 42