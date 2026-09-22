"""SUMO traffic environment for RL training.

Wraps the SUMO simulation via traci and provides a Gym-like interface
for RL agents.
"""

from __future__ import annotations

import numpy as np
import traci

from traffic_rl.config import INCOMING_EDGES, EnvironmentConfig
from traffic_rl.logging import get_logger
from traffic_rl.metrics import (
    TrafficMetrics,
    WorldSnapshot,
    compute_metrics,
    compute_reward,
    compute_state,
    metrics_to_dict,
)
from traffic_rl.sumo.binary import build_sumo_command, resolve_sumo_config_path

log = get_logger("traffic_rl.sumo.environment")


class TrafficEnvironment:
    """Gym-like environment wrapping a SUMO simulation.

    Timing model (see :class:`~traffic_rl.config.EnvironmentConfig`): SUMO is
    integrated with ``sim_step_s`` (default 1 s) and the agent observes/acts
    every ``step_length`` simulated seconds (default 5 s), i.e. ``sub_steps``
    SUMO steps elapse per decision.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config
        self._sim: traci.Simulation | None = None
        self._step_count = 0
        self._sim_time = 0.0
        self._current_phase = 0
        self._phase_start_time = 0.0
        self._last_metrics = None
        self._sumo_running = False
        self._vehicles_arrived = 0
        #: Cap on per-vehicle traci queries per snapshot (perf guard on big nets).
        self._max_vehicles_sampled = 200

    # -- introspection ---------------------------------------------------------

    @property
    def sim_time(self) -> float:
        """Simulated seconds elapsed in the current episode."""
        return self._sim_time

    @property
    def decision_count(self) -> int:
        """Number of agent decisions taken in the current episode."""
        return self._step_count

    @property
    def current_phase(self) -> int:
        """Index into :data:`~traffic_rl.config.GREEN_PHASES` currently active."""
        return self._current_phase

    @property
    def config(self) -> EnvironmentConfig:
        return self._config

    # -- gym-like API ----------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the simulation and return the initial state."""
        if seed is not None:
            np.random.seed(seed)
        self._close_sim()

        config_path = resolve_sumo_config_path(self._config.config_path, self._config.run_dir)
        cmd = build_sumo_command(
            config_path=config_path,
            gui=self._config.gui,
            seed=seed if seed is not None else self._config.seed,
            end=self._config.episode_seconds,
            step_length=self._config.sim_step_s,
            scale=self._config.scale,
            time_to_teleport=self._config.time_to_teleport,
            collision_action=self._config.collision_action,
            tripinfo_output=self._config.tripinfo_output,
            summary_output=self._config.summary_output,
            additional_files=self._config.additional_files,
        )
        log.info(
            "starting SUMO (%d decisions x %ds = %.0fs horizon, gui=%s)",
            self._config.max_steps,
            self._config.step_length,
            self._config.episode_seconds,
            self._config.gui,
        )
        try:
            traci.start(cmd)
            self._sim = traci
            self._step_count = 0
            self._sim_time = 0.0
            self._current_phase = 0
            self._phase_start_time = 0.0
            self._last_metrics = None
            self._vehicles_arrived = 0
            self._sumo_running = True
            state = self._get_state()
            self._last_metrics = self._compute_metrics()
            return state
        except Exception as e:
            log.error("failed to start SUMO: %s", e)
            self._sumo_running = False
            raise

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one decision step. Returns (state, reward, done, info)."""
        if not self._sumo_running:
            raise RuntimeError("Environment not started. Call reset() first.")
        prev_metrics = self._last_metrics
        switched = self._execute_action(action)

        for _ in range(self._config.sub_steps):
            self._sim.simulationStep()
            self._sim_time += float(self._config.sim_step_s)
            self._vehicles_arrived += int(self._sim.simulation.getArrivedNumber())
        self._step_count += 1

        state = self._get_state()
        metrics = self._compute_metrics()
        reward = compute_reward(prev_metrics, metrics, switched)
        self._last_metrics = metrics

        done = self._step_count >= self._config.max_steps
        if not done and self._config.end_when_empty:
            try:
                done = self._sim.simulation.getMinExpectedNumber() == 0
            except Exception:  # pragma: no cover - SUMO already terminated
                done = True
        info = {
            "step": self._step_count,
            "sim_time": self._sim_time,
            "metrics": metrics,
            "phase": self._current_phase,
            "phase_switched": switched,
        }
        return state, reward, done, info

    def close(self) -> None:
        self._close_sim()

    def __enter__(self) -> TrafficEnvironment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------------

    def _close_sim(self) -> None:
        if self._sumo_running and self._sim is not None:
            try:
                self._sim.close()
            except Exception as e:
                log.warning("error closing SUMO: %s", e)
            self._sumo_running = False
            self._sim = None

    def _execute_action(self, action: int) -> bool:
        """Apply a phase selection. Returns ``True`` when the phase changed.

        A switch is rejected while the current green phase is younger than
        ``min_phase_duration`` (measured in simulated seconds).
        """
        action = int(action)
        if action == self._current_phase:
            return False
        green_time = self._sim_time - self._phase_start_time
        if green_time < self._config.min_phase_duration:
            return False
        max_green = self._config.max_phase_duration
        if green_time >= max_green or green_time >= self._config.min_phase_duration:
            self._current_phase = action
            self._phase_start_time = self._sim_time
            return True
        return False

    def _get_state(self) -> np.ndarray:
        return compute_state(self._get_snapshot())

    def _get_snapshot(self) -> WorldSnapshot:
        """Translate raw ``traci`` calls into a :class:`WorldSnapshot`."""
        vehicle_ids: list[str] = []
        vehicle_speeds: dict[str, float] = {}
        vehicle_waiting: dict[str, float] = {}
        vehicle_types: dict[str, str] = {}
        edge_vehicle_counts: dict[str, int] = {}
        edge_vehicle_ids: dict[str, list[str]] = {}
        edge_lengths: dict[str, float] = {}

        if self._sim is not None:
            try:
                vehicle_ids = list(self._sim.vehicle.getIDList())
            except Exception as exc:  # pragma: no cover - SUMO terminated
                log.debug("could not list vehicles: %s", exc)
                vehicle_ids = []
            for vid in vehicle_ids[: self._max_vehicles_sampled]:
                try:
                    vehicle_speeds[vid] = self._sim.vehicle.getSpeed(vid)
                    vehicle_waiting[vid] = self._sim.vehicle.getWaitingTime(vid)
                    vehicle_types[vid] = self._sim.vehicle.getTypeID(vid)
                except Exception:
                    continue
            for edge in INCOMING_EDGES:
                try:
                    edge_vehicles = list(self._sim.edge.getLastStepVehicleIDs(edge))
                    edge_vehicle_ids[edge] = edge_vehicles
                    edge_vehicle_counts[edge] = len(edge_vehicles)
                    edge_lengths[edge] = float(self._sim.edge.getLength(edge))
                except Exception:
                    edge_vehicle_ids[edge] = []
                    edge_vehicle_counts[edge] = 0
                    edge_lengths[edge] = 100.0

        return WorldSnapshot(
            vehicle_ids=vehicle_ids,
            vehicle_speeds=vehicle_speeds,
            vehicle_waiting=vehicle_waiting,
            vehicle_types=vehicle_types,
            edge_vehicle_counts=edge_vehicle_counts,
            edge_vehicle_ids=edge_vehicle_ids,
            edge_lengths=edge_lengths,
            sim_time=self._sim_time,
            vehicles_exited=self._vehicles_arrived,
        )

    def _compute_metrics(self) -> TrafficMetrics:
        return compute_metrics(self._get_snapshot())

    def metrics_dict(self) -> dict:
        """Return the latest metrics as a plain dict (for logging/artefacts)."""
        if self._last_metrics is None:
            self._last_metrics = self._compute_metrics()
        return metrics_to_dict(self._last_metrics)
