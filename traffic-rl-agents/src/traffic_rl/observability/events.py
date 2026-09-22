"""Stable event names and their required fields.

Structured logs are only useful if the *event names* and *field names* are a
contract.  :data:`EVENT_CATALOGUE` is that contract: it is asserted by the test
suite, emitted by the codebase, and rendered into the observability docs.

Every event record is a JSON object shaped as::

    {
      "timestamp": "2026-05-01T10:00:00+00:00",
      "level": "INFO",
      "logger": "traffic_rl.training.trainer",
      "event": "training.episode.end",
      "message": "...",
      "run_id": "...", "agent": "ppo", "episode": 12,
      ... event specific fields ...
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from traffic_rl.observability.context import get_context

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

# -- run lifecycle -------------------------------------------------------------
RUN_START = "run.start"
RUN_END = "run.end"
RUN_FAILED = "run.failed"

# -- training -----------------------------------------------------------------
TRAINING_EPISODE_START = "training.episode.start"
TRAINING_EPISODE_END = "training.episode.end"
TRAINING_UPDATE = "training.update"
TRAINING_CHECKPOINT_SAVED = "training.checkpoint.saved"
TRAINING_BEST_SAVED = "training.best.saved"
TRAINING_EARLY_STOP = "training.early_stop"
TRAINING_RESUMED = "training.resumed"
TRAINING_INTERRUPTED = "training.interrupted"

# -- evaluation / benchmarking -------------------------------------------------
EVAL_RUN_START = "eval.run.start"
EVAL_EPISODE_END = "eval.episode.end"
EVAL_REPORT_WRITTEN = "eval.report.written"
EVAL_GATE_FAILED = "eval.gate.failed"
EVAL_GATE_PASSED = "eval.gate.passed"
BENCHMARK_COMPARISON = "benchmark.comparison"

# -- scenarios -----------------------------------------------------------------
SCENARIO_GENERATED = "scenario.generated"
SCENARIO_SAMPLED = "scenario.sampled"
SCENARIO_CALIBRATED = "scenario.calibrated"
SCENARIO_SPLIT_WRITTEN = "scenario.split.written"

# -- simulation ----------------------------------------------------------------
SUMO_CONNECTION_UP = "sumo.connection.up"
SUMO_CONNECTION_DOWN = "sumo.connection.down"
SUMO_CRASH = "sumo.crash"
SUMO_EPISODE_SUMMARY = "sumo.episode.summary"

# -- models / artefacts --------------------------------------------------------
MODEL_REGISTERED = "model.registered"
MODEL_LOADED = "model.loaded"
ARTIFACT_UPLOADED = "artifact.uploaded"
ARTIFACT_DOWNLOADED = "artifact.downloaded"
ARTIFACT_SYNCED = "artifact.synced"

# -- infrastructure ------------------------------------------------------------
JOB_SUBMITTED = "job.submitted"
JOB_SUCCEEDED = "job.succeeded"
JOB_FAILED = "job.failed"
CONFIG_RESOLVED = "config.resolved"
SECRET_RESOLVED = "secret.resolved"

# -- errors / performance ------------------------------------------------------
ERROR_OCCURRED = "error.occurred"
PERF_STEP = "perf.step"
PERF_SUMMARY = "perf.summary"

#: Every event name, with the fields callers must provide.
EVENT_CATALOGUE: Mapping[str, Sequence[str]] = {
    RUN_START: ("run_id", "agent", "config_hash"),
    RUN_END: ("run_id", "status", "duration_s"),
    RUN_FAILED: ("run_id", "error_type", "error_message"),
    TRAINING_EPISODE_START: ("run_id", "episode"),
    TRAINING_EPISODE_END: ("run_id", "episode", "episode_reward", "steps"),
    TRAINING_UPDATE: ("run_id", "global_step", "loss"),
    TRAINING_CHECKPOINT_SAVED: ("run_id", "path", "episode"),
    TRAINING_BEST_SAVED: ("run_id", "path", "episode", "episode_reward"),
    TRAINING_EARLY_STOP: ("run_id", "episode", "best_reward", "patience"),
    TRAINING_RESUMED: ("run_id", "path", "episode"),
    TRAINING_INTERRUPTED: ("run_id", "signal", "episode"),
    EVAL_RUN_START: ("run_id", "agent", "episodes"),
    EVAL_EPISODE_END: ("run_id", "episode", "episode_reward"),
    EVAL_REPORT_WRITTEN: ("run_id", "path"),
    EVAL_GATE_FAILED: ("run_id", "reason"),
    EVAL_GATE_PASSED: ("run_id", "reason"),
    BENCHMARK_COMPARISON: ("run_id", "policies"),
    SCENARIO_GENERATED: ("scenario_id", "path", "seed"),
    SCENARIO_SAMPLED: ("scenario_id", "difficulty", "split"),
    SCENARIO_CALIBRATED: ("scenario_id", "geh_score", "rmse"),
    SCENARIO_SPLIT_WRITTEN: ("path", "n_train", "n_val", "n_test"),
    SUMO_CONNECTION_UP: ("run_id", "port", "scenario_id"),
    SUMO_CONNECTION_DOWN: ("run_id", "scenario_id"),
    SUMO_CRASH: ("run_id", "scenario_id", "error_message"),
    SUMO_EPISODE_SUMMARY: ("run_id", "episode", "sim_time", "throughput"),
    MODEL_REGISTERED: ("run_id", "model_package", "version"),
    MODEL_LOADED: ("run_id", "path", "agent"),
    ARTIFACT_UPLOADED: ("path", "uri", "bytes"),
    ARTIFACT_DOWNLOADED: ("uri", "path"),
    ARTIFACT_SYNCED: ("local_dir", "uri", "n_files"),
    JOB_SUBMITTED: ("job_name", "backend"),
    JOB_SUCCEEDED: ("job_name", "backend"),
    JOB_FAILED: ("job_name", "backend", "reason"),
    CONFIG_RESOLVED: ("config_hash", "config_path"),
    SECRET_RESOLVED: ("name", "source"),
    ERROR_OCCURRED: ("error_type", "error_message"),
    PERF_STEP: ("global_step", "steps_per_second"),
    PERF_SUMMARY: ("run_id", "steps_per_second", "wall_time_s"),
}

#: Fields that every event emit should resolve from the logging context.
CONTEXTUAL_FIELDS: tuple[str, ...] = (
    "run_id",
    "agent",
    "scenario_id",
    "episode",
    "global_step",
)


def validate_event(event: str, fields: Mapping[str, Any]) -> list[str]:
    """Return the required fields that are missing from *fields*.

    Used by the test suite (and available at runtime for defensive checks); it
    never raises, so logging can never break training.
    """
    required = EVENT_CATALOGUE.get(event)
    if required is None:
        return []
    return [name for name in required if name not in fields or fields[name] is None]


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Emit a structured event record on *logger*.

    The payload is attached to the record as ``structured`` so the JSON
    formatter flattens it into the JSON object while the console formatter keeps
    a readable single line.  Missing required fields are flagged rather than
    raising — observability must never break a training run.
    """
    resolved = {**get_context(), **fields}
    missing = validate_event(event, resolved)
    payload: dict[str, Any] = {"event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    if missing:
        payload["_missing_fields"] = missing
    logger.log(level, message or event, extra={"structured": payload}, exc_info=exc_info)
