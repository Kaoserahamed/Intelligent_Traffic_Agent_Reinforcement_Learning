"""Rule-based traffic signal controllers.

Decoupled from ``traci``: green-time calculation takes a *count accessor*
callable, so it can be unit-tested without SUMO.  The environment injects the
real ``traci`` adapter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from traffic_rl.config import DEFAULT_TIME_ALLOCATIONS, INCOMING_EDGES, VEHICLE_TYPES

log = logging.getLogger("traffic_rl.sumo.controller")

CountAccessor = Callable[[str], Mapping[str, int]]


@dataclass
class PhaseRecord:
    """A single executed phase for historical analysis."""

    name: str
    edge: str
    sumo_phase: int
    planned_duration: float
    actual_duration: float
    start_step: int
    end_step: int
    forced_switch: bool
    vehicle_counts: Mapping[str, int] = field(default_factory=dict)


class SimpleDynamicController:
    """Allocates green time proportionally to vehicle composition per edge."""

    PHASE_CYCLE: tuple = (
        {"name": "South", "edge": "5to1", "sumo_phase": 0},
        {"name": "East", "edge": "2to1", "sumo_phase": 3},
        {"name": "North", "edge": "3to1", "sumo_phase": 6},
        {"name": "West", "edge": "4to1", "sumo_phase": 9},
    )

    MIN_GREEN = 10
    MAX_GREEN = 90
    BUFFER_FACTOR = 1.3
    STARTUP_SECONDS = 5.0

    def __init__(self, time_allocations: Mapping[str, float] | None = None) -> None:
        self.time_allocations: dict[str, float] = dict(
            time_allocations if time_allocations is not None else DEFAULT_TIME_ALLOCATIONS
        )
        self.edge_to_direction = {"5to1": "South", "2to1": "East",
                                  "3to1": "North", "4to1": "West"}
        self.incoming_edges: tuple[str, ...] = INCOMING_EDGES
        self.phases: tuple[int, ...] = tuple(p["sumo_phase"] for p in self.PHASE_CYCLE)
        self.stats: dict = {
            "allocation_name": "Default",
            "total_phases": 0,
            "phase_durations": [],
            "vehicle_counts_per_phase": [],
            "edge_traffic_history": {edge: [] for edge in self.incoming_edges},
        }
        self.phase_history: list = []

    def get_vehicle_counts_by_type(self, edge, get_counts) -> dict[str, int]:
        counts = {vt: 0 for vt in VEHICLE_TYPES}
        try:
            live = dict(get_counts(edge))
        except Exception as exc:  # pragma: no cover - adapter contract
            log.warning("count accessor failed for edge %s: %s", edge, exc)
            return counts
        for vtype in VEHICLE_TYPES:
            counts[vtype] = int(live.get(vtype, 0))
        return counts

    def calculate_dynamic_green_time(self, edge, get_counts):
        """Return (green_time_seconds, breakdown, total_vehicles) for edge."""
        vehicle_counts = self.get_vehicle_counts_by_type(edge, get_counts)
        direction = self.edge_to_direction.get(edge, "Unknown")
        total_time = 0.0
        total_vehicles = 0
        breakdown = {}
        for vtype, count in vehicle_counts.items():
            if count > 0:
                time_needed = count * self.time_allocations[vtype]
                total_time += time_needed
                total_vehicles += count
                breakdown[vtype] = time_needed
        if total_vehicles == 0:
            green_time = self.MIN_GREEN
            log.debug("%s Edge (%s): %ds (vehicles=%d)",
                      direction, edge, green_time, total_vehicles)
        else:
            raw_time = total_time * self.BUFFER_FACTOR
            if raw_time < self.STARTUP_SECONDS:
                raw_time = self.STARTUP_SECONDS
            green_time = min(max(raw_time, self.MIN_GREEN), self.MAX_GREEN)
            log.debug("%s Edge (%s): %.0fs (%s, vehicles=%d)",
                      direction, edge, green_time, breakdown, total_vehicles)
        return green_time, breakdown, total_vehicles

    def record_phase(self, phase, actual_duration, start_step, end_step,
                     vehicle_counts, forced_switch=False):
        record = PhaseRecord(
            name=str(phase["name"]), edge=str(phase["edge"]),
            sumo_phase=int(phase["sumo_phase"]),
            planned_duration=float(phase.get("planned_duration", actual_duration)),
            actual_duration=actual_duration, start_step=start_step,
            end_step=end_step, forced_switch=forced_switch,
            vehicle_counts=dict(vehicle_counts),
        )
        self.phase_history.append(record)
        self.stats["total_phases"] += 1
        self.stats["phase_durations"].append(actual_duration)
        self.stats["vehicle_counts_per_phase"].append(dict(vehicle_counts))
        self.stats["edge_traffic_history"].setdefault(str(phase["edge"]), []).append({
            "total_vehicles": sum(vehicle_counts.values()),
            "allocated_time": actual_duration,
        })

    def get_edge_statistics(self, edge):
        history = self.stats["edge_traffic_history"].get(edge)
        if not history:
            return None
        total_vehicles = sum(h["total_vehicles"] for h in history)
        total_time = sum(h["allocated_time"] for h in history)
        count = len(history)
        return {
            "edge": edge,
            "direction": self.edge_to_direction.get(edge, "Unknown"),
            "total_phases": count,
            "total_vehicles": total_vehicles,
            "total_allocated_time": total_time,
            "avg_vehicles_per_phase": total_vehicles / count if count else 0.0,
            "avg_time_per_phase": total_time / count if count else 0.0,
        }

    def print_stats_summary(self):
        if self.stats["total_phases"] == 0:
            log.warning("no phase data available")
            return
        durations = self.stats["phase_durations"]
        log.info("ALLOCATION PERFORMANCE (%s)", self.stats["allocation_name"])
        log.info("  allocations=%s", self.time_allocations)
        log.info("  total phases=%d  avg/min/max=%.1f/%.0f/%.0f s",
                 len(durations), float(np.mean(durations)), min(durations), max(durations))
        by_type = {vt: 0 for vt in self.time_allocations}
        for counts in self.stats["vehicle_counts_per_phase"]:
            for vt, c in counts.items():
                by_type[vt] = by_type.get(vt, 0) + c
        log.info("  vehicles processed: %s", by_type)


class FixedTimeController(SimpleDynamicController):
    """Fixed-green-time baseline controller, interchangeable with the dynamic one.

    Cycles through the four green phases with equal durations.  The phase
    sequence and durations are derived from the SUMO ``<tlLogic>`` (see
    ``sim/intersection.net.xml``) so the baseline matches what SUMO would run
    when ``programID="0"`` is active.
    """

    def __init__(self, fixed_time: int = 30,
                 time_allocations: Mapping[str, float] | None = None) -> None:
        super().__init__(time_allocations=time_allocations)
        self.fixed_time = int(fixed_time)
        self.time_allocations = {**self.time_allocations, "fixed_duration": self.fixed_time}
        self.stats["allocation_name"] = f"Fixed_{self.fixed_time}ds"

    def calculate_dynamic_green_time(self, edge, get_counts):
        counts = self.get_vehicle_counts_by_type(edge, get_counts)
        total_vehicles = sum(counts.values())
        self.stats["total_phases"] += 1
        self.stats["phase_durations"].append(self.fixed_time)
        log.info("Fixed time: %ds for all phases (edge=%s, vehicles=%d)",
                 self.fixed_time, edge, total_vehicles)
        return self.fixed_time, {}, total_vehicles

    def reset(self) -> None:
        """Reset per-run counters so the controller can be reused across episodes."""
        self.stats["total_phases"] = 0
        self.stats["phase_durations"] = []
        self.stats["vehicle_counts_per_phase"] = []
        self.stats["edge_traffic_history"] = {edge: [] for edge in self.incoming_edges}
        self.phase_history = []
