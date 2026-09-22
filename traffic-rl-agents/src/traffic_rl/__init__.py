"""TrafficRL - Reinforcement Learning for Traffic Signal Control.

A production-grade package for training and evaluating RL agents
(Q-Learning, DQN, Double DQN, PPO) in SUMO traffic simulations.
"""

from __future__ import annotations

from importlib import import_module  # noqa: F401  (kept for backwards compatibility)
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Package metadata (loaded safely to avoid circular import issues)
# --------------------------------------------------------------------------- #


def _load_version() -> str:
    """Load __version__ from __about__.py without triggering import side-effects."""
    about_path = Path(__file__).resolve().parent / "__about__.py"
    namespace: dict[str, Any] = {}
    exec(compile(about_path.read_text(encoding="utf-8"), about_path.as_posix(), "exec"), namespace)
    return str(namespace.get("__version__", "0.0.0"))


__version__: str = _load_version()


# --------------------------------------------------------------------------- #
# Public API surface
# --------------------------------------------------------------------------- #

from .config import (  # noqa: E402
    AgentConfig,
    Config,
    EnvironmentConfig,
    TrainingConfig,
    default_config_path,
    default_sim_dir,
    project_root,
)
from .defaults import get_default_config, resolve_config  # noqa: E402

# ``traffic_rl.cli`` pulls in traci/argparse/agents; importing it eagerly would
# make every ``import traffic_rl`` (settings, metrics, plotting, ...) pay for it.
# It is resolved lazily through the module ``__getattr__`` below.
_CLI_LAZY_ATTRS = {"main"}


def __getattr__(name: str) -> Any:
    """Lazily resolve the ``main`` entry point of :mod:`traffic_rl.cli`."""
    if name in _CLI_LAZY_ATTRS:
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "AgentConfig",
    "Config",
    "EnvironmentConfig",
    "TrainingConfig",
    "default_config_path",
    "default_sim_dir",
    "project_root",
    "get_default_config",
    "resolve_config",
    "main",
]



