"""SUMO traffic environment for RL training.

Wraps the SUMO simulation via traci and provides a Gym-like interface
for RL agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import traci

from traffic_rl.config import EnvironmentConfig
from traffic_rl.logging import get_logger
from traffic_rl.metrics import WorldSnapshot, compute_state
from traffic_rl.sumo.binary import build_sumo_command

log = get_logger("traffic_rl.sumo.environment")


class TrafficEnvironment:
    """Gym-like environment wrapping a SUMO simulation."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config
        self._sim: Optional[traci.Simulation] = None
        self._step_count = 0
        self._current_phase = 0
        self._phase_start_step = 0
        self._last_metrics = None
        self._sumo_running = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the simulation and return the initial state."""
        if seed is not None:
            np.random.seed(seed)
        self._close_sim()
        cmd = build_sumo_command(
            config_path=self._config.config_path,
            gui=self._config.gui,
            seed=seed,
            max_steps=self._config.max_steps,
            end=self._config.max_steps * self._config.step_length // 1000,
        )
        log.info("starting SUMO (%d steps, gui=%s)", self._config.max_steps,
                 self._config.gui)
        try:
            traci.start(cmd)
            self._sim = traci
            self._step_count = 0
            self._current_phase = 0
            self._phase_start_step = 0
            self._last_metrics = None
            self._sumo_running = True
            return self._get_state()
        except Exception as e:
            log.error("failed to start SUMO: %s", e)
            self._sumo_running = False
            raise

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step. Returns (state, reward, done, info)."""
        if not self._sumo_running:
            raise RuntimeError("Environment not started. Call reset() first.")
        prev_state = self._get_state()
        prev_metrics = self._last_metrics
        self._execute_action(action)
        self._sim.simulationStep()
        self._step_count += 1
        state = self._get_state()
        metrics = self._compute_metrics()
        self._last_metrics = metrics
        from traffic_rl.metrics import compute_reward
        reward = compute_reward(prev_metrics, metrics, action != self._current_phase)
        done = self._step_count >= self._config.max_steps
        info = {"step": self._step_count, "metrics": metrics, "phase": self._current_phase}
        return state, reward, done, info

    def close(self) -> None:
        self._close_sim()

    def _close_sim(self) -> None:
        if self._sumo_running and self._sim is not None:
            try:
                self._sim.close()
            except Exception as e:
                log.warning("error closing SUMO: %s", e)
            self._sumo_running = False
            self._sim = None

    def _execute_action(self, action: int) -> None:
        if action != self._current_phase:
            duration = self._step_count - self._phase_start_step
            if duration < self._config.min_phase_duration:
                return
            self._current_phase = action % 4
            self._phase_start_step = self._step_count

    def _get_state(self) -> np.ndarray:
        snapshot = self._get_snapshot()
        return compute_state(snapshot)

    def _get_snapshot(self) -> WorldSnapshot:
        vehicle_ids = []
        vehicle_speeds = {}
        vehicle_waiting = {}
        vehicle_types = {}
        edge_vehicle_counts = {}
        edge_vehicle_ids = {}
        edge_lengths = {}
        if self._sim is not None:
            vehicle_ids = self._sim.vehicle.getIDList()
            for vid in vehicle_ids[:50]:  # limit for performance
                try:
                    vehicle_speeds[vid] = self._sim.vehicle.getSpeed(vid)
                    vehicle_waiting[vid] = self._sim.vehicle.getWaitingTime(vid)
                    vehicle_types[vid] = self._sim.vehicle.getTypeID(vid)
                except Exception:
                    pass
            for edge in ["5to1", "2to1", "3to1", "4to1"]:
                try:
                    edge_vehicles = self._sim.edge.getLastStepVehicleIDs(edge)
                    edge_vehicle_ids[edge] = edge_vehicles
                    edge_vehicle_counts[edge] = len(edge_vehicles)
                    edge_lengths[edge] = self._sim.edge.getLength(edge)
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
            sim_time=float(self._step_count * self._config.step_length / 1000.0),
            vehicles_exited=0,
        )

    def _compute_metrics(self) -> dict:
        from traffic_rl.metrics import compute_metrics
        snapshot = self._get_snapshot()
        m = compute_metrics(snapshot)
        return {
            "total_vehicles": m.total_vehicles,
            "avg_waiting_time": m.avg_waiting_time,
            "avg_speed": m.avg_speed,
            "total_waiting_time": m.total_waiting_time,
            "queue_lengths": list(m.queue_lengths),
            "total_queue": m.total_queue,
            "throughput": m.throughput,
        }
