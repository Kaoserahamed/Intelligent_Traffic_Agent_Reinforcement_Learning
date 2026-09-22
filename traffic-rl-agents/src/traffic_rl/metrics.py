"""Pure, side-effect-free traffic metric, state and reward computation.

Every function here takes already-queried simulator data (a
:class:`WorldSnapshot`) and returns plain Python data structures.  This keeps
the logic trivially unit-testable — no SUMO installation required.

The simulator wrapper (:mod:`traffic_rl.sumo.environment`) translates raw
``traci`` calls into a :class:`WorldSnapshot` and hands it to these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from traffic_rl.config import (
    INCOMING_EDGES,
    STATE_SIZE,
    VEHICLE_TYPES,
)


@dataclass
class WorldSnapshot:
    """A minimal, serialisable view of the simulated world at one step.

    The environment builds this from ``traci`` calls; tests build it by hand.
    """

    vehicle_ids: Sequence[str] = ()
    vehicle_speeds: Mapping[str, float] = field(default_factory=dict)
    vehicle_waiting: Mapping[str, float] = field(default_factory=dict)
    vehicle_types: Mapping[str, str] = field(default_factory=dict)
    edge_vehicle_counts: Mapping[str, int] = field(default_factory=dict)
    edge_vehicle_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    edge_lengths: Mapping[str, float] = field(default_factory=dict)
    sim_time: float = 0.0
    vehicles_exited: int = 0  # cumulative count of vehicles that left the network


# -----------------------------------------------------------------------------
# State vector
# -----------------------------------------------------------------------------


def empty_state() -> np.ndarray:
    """Return a zero vector of :data:`~traffic_rl.config.STATE_SIZE`."""
    return np.zeros(STATE_SIZE, dtype=np.float32)


def compute_state(snapshot: WorldSnapshot) -> np.ndarray:
    """Build the 28-dimensional RL state vector from a snapshot.

    Layout (per incoming edge): 5 normalised vehicle-type counts, a
    normalised queue length and a normalised density, for each of 4 edges.
    """
    state: list[float] = []
    for edge in INCOMING_EDGES:
        counts = {vt: 0 for vt in VEHICLE_TYPES}
        vehicles = snapshot.edge_vehicle_ids.get(edge, ())
        for vid in vehicles:
            vtype = snapshot.vehicle_types.get(vid)
            if vtype in counts:
                counts[vtype] += 1
        state.extend(counts[vt] / 10.0 for vt in VEHICLE_TYPES)

        queue = 0
        for vid in vehicles:
            if snapshot.vehicle_speeds.get(vid, 0.0) < 1.0:
                queue += 1
        state.append(min(queue / 20.0, 1.0))

        count = snapshot.edge_vehicle_counts.get(edge, 0)
        length = snapshot.edge_lengths.get(edge, 0.0)
        density = count / max(length / 100.0, 1.0)
        state.append(min(density / 5.0, 1.0))

        return np.array(state, dtype=np.float32)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


@dataclass
class TrafficMetrics:
    """Snapshot of traffic performance indicators."""

    total_vehicles: int = 0
    avg_waiting_time: float = 0.0
    avg_speed: float = 0.0
    total_waiting_time: float = 0.0
    queue_lengths: list[int] = field(default_factory=list)
    total_queue: int = 0
    throughput: int = 0


def empty_metrics() -> TrafficMetrics:
    return TrafficMetrics(queue_lengths=[0] * len(INCOMING_EDGES))


def compute_metrics(snapshot: WorldSnapshot) -> TrafficMetrics:
    """Compute aggregate traffic metrics from a snapshot."""
    vehicle_ids = list(snapshot.vehicle_ids)
    total_vehicles = len(vehicle_ids)
    if total_vehicles == 0:
        metrics = empty_metrics()
        metrics.throughput = snapshot.vehicles_exited
        return metrics

    total_waiting_time = float(
        sum(snapshot.vehicle_waiting.get(vid, 0.0) for vid in vehicle_ids)
    )
    total_speed = float(sum(snapshot.vehicle_speeds.get(vid, 0.0) for vid in vehicle_ids))
    avg_speed = total_speed / total_vehicles
    avg_waiting = total_waiting_time / total_vehicles

    queue_lengths: list[int] = []
    for edge in INCOMING_EDGES:
        vehicles = snapshot.edge_vehicle_ids.get(edge, ())
        queue = sum(1 for vid in vehicles if snapshot.vehicle_speeds.get(vid, 0.0) < 1.0)
        queue_lengths.append(queue)

    return TrafficMetrics(
        total_vehicles=total_vehicles,
        avg_waiting_time=avg_waiting,
        avg_speed=avg_speed,
        total_waiting_time=total_waiting_time,
        queue_lengths=queue_lengths,
        total_queue=sum(queue_lengths),
        throughput=snapshot.vehicles_exited,
    )


def metrics_to_dict(metrics: TrafficMetrics) -> dict:
    """Convert metrics to a plain dict (kept for backwards compatibility)."""
    return {
        "total_vehicles": metrics.total_vehicles,
        "avg_waiting_time": metrics.avg_waiting_time,
        "avg_speed": metrics.avg_speed,
        "total_waiting_time": metrics.total_waiting_time,
        "queue_lengths": list(metrics.queue_lengths),
        "total_queue": metrics.total_queue,
        "throughput": metrics.throughput,
    }


# -----------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------

# Tunable reward weights (explicit so the reward function is auditable).
_REWARD_QUEUE_IMPROVEMENT = 10
_REWARD_WAITING_IMPROVEMENT = 0.1
_REWARD_SPEED_IMPROVEMENT = 5
_REWARD_THROUGHPUT = 20
_REWARD_PENALTY_PER_SWITCH = 5
_REWARD_QUEUE_PENALTY_THRESHOLD = 15
_REWARD_QUEUE_PENALTY = 2
_REWARD_QUEUE_IMBALANCE = 2
_REWARD_LOW_SPEED_PENALTY = 20
_REWARD_LOW_SPEED_THRESHOLD = 2.0
_REWARD_CLAMP = 100


def compute_reward(old: TrafficMetrics, new: TrafficMetrics, phase_switched: bool) -> float:
    """Compute the scalar reward for a single environment transition.

    Mirrors the original ``traffic_env._calculate_reward`` but reads its inputs
    from :class:`TrafficMetrics`, making it deterministic and testable.
    """
    try:
        reward = 0.0
        reward += (old.total_queue - new.total_queue) * _REWARD_QUEUE_IMPROVEMENT
        reward += (old.total_waiting_time - new.total_waiting_time) * _REWARD_WAITING_IMPROVEMENT
        reward += (new.avg_speed - old.avg_speed) * _REWARD_SPEED_IMPROVEMENT
        if new.throughput > old.throughput:
            reward += (new.throughput - old.throughput) * _REWARD_THROUGHPUT
        if phase_switched:
            reward -= _REWARD_PENALTY_PER_SWITCH
        if new.total_queue > _REWARD_QUEUE_PENALTY_THRESHOLD:
            reward -= (new.total_queue - _REWARD_QUEUE_PENALTY_THRESHOLD) * _REWARD_QUEUE_PENALTY
        if new.queue_lengths:
            queue_balance = float(np.std(new.queue_lengths))
            reward -= queue_balance * _REWARD_QUEUE_IMBALANCE
        if new.avg_speed < _REWARD_LOW_SPEED_THRESHOLD:
            reward -= _REWARD_LOW_SPEED_PENALTY
        return float(max(min(reward, _REWARD_CLAMP), -_REWARD_CLAMP))
    except Exception:  # pragma: no cover - defensive
        return -50.0


