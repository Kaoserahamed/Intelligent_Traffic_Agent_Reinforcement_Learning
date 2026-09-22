"""Observability: structured logging, metrics sinks and profiling.

Public API
----------
* :func:`configure_structured_logging` — JSON/console logging with redaction.
* :func:`bind_context` / :func:`run_context` — correlation ids for every record.
* :func:`log_event` + :data:`EVENT_CATALOGUE` — stable event contract.
* :class:`MultiSink`, :class:`JsonlMetricsSink`, :class:`TensorBoardSink`,
  :class:`SageMakerSink`, :class:`CloudWatchSink` — metrics destinations.
* :class:`PhaseTimer`, :class:`StepRateMeter` — performance instrumentation.
"""

from traffic_rl.observability.context import (
    CONTEXT_KEYS,
    bind_context,
    clear_context,
    context_snapshot,
    get_context,
    new_run_id,
    new_trace_id,
    run_context,
    unbind_context,
    utc_now_iso,
)
from traffic_rl.observability.events import EVENT_CATALOGUE, log_event, validate_event
from traffic_rl.observability.logging_config import (
    ConsoleFormatter,
    ContextFilter,
    JsonFormatter,
    RedactionFilter,
    build_formatter,
    configure_structured_logging,
    get_structured_logger,
    redact_mapping,
    redact_text,
)
from traffic_rl.observability.profiler import (
    PhaseTimer,
    StepRateMeter,
    process_rss_mb,
)
from traffic_rl.observability.sinks import (
    JsonArtifactSink,
    JsonlMetricsSink,
    MetricsSink,
    MultiSink,
    NullSink,
    SageMakerSink,
    TensorBoardSink,
)

__all__ = [
    "CONTEXT_KEYS",
    "EVENT_CATALOGUE",
    "ConsoleFormatter",
    "ContextFilter",
    "JsonArtifactSink",
    "JsonFormatter",
    "JsonlMetricsSink",
    "MetricsSink",
    "MultiSink",
    "NullSink",
    "PhaseTimer",
    "RedactionFilter",
    "SageMakerSink",
    "StepRateMeter",
    "TensorBoardSink",
    "bind_context",
    "build_formatter",
    "clear_context",
    "configure_structured_logging",
    "context_snapshot",
    "get_context",
    "get_structured_logger",
    "log_event",
    "new_run_id",
    "new_trace_id",
    "process_rss_mb",
    "redact_mapping",
    "redact_text",
    "run_context",
    "unbind_context",
    "utc_now_iso",
    "validate_event",
]
