"""SUMO integration layer for TrafficRL.

Exports SUMO binary utilities, rule-based controllers, and the (legacy)
Gym-like traffic environment.  New code should prefer
:class:`traffic_rl.envs.sumo_env.SumoTrafficEnv`, which implements the
Gymnasium API and the richer observation/action specification.
"""

from traffic_rl.sumo.binary import (
    SUMONotFoundError,
    build_sumo_command,
    find_sumo_binary,
    resolve_sumo_config_path,
    sumo_version,
)
from traffic_rl.sumo.controller import (
    CountAccessor,
    FixedTimeController,
    PhaseRecord,
    SimpleDynamicController,
)
from traffic_rl.sumo.environment import TrafficEnvironment

__all__ = [
    "SUMONotFoundError",
    "build_sumo_command",
    "find_sumo_binary",
    "resolve_sumo_config_path",
    "sumo_version",
    "SimpleDynamicController",
    "FixedTimeController",
    "PhaseRecord",
    "CountAccessor",
    "TrafficEnvironment",
]

