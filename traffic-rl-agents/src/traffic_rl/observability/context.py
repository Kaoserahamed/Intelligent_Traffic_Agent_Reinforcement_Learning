"""Context propagation for structured logging and artefacts.

A single :class:`dict` is stored in a :class:`contextvars.ContextVar`, so every
log record, metric event and artefact write can be correlated with the run,
experiment, agent, scenario and episode it belongs to — across threads and
(asyncio) tasks without passing state around explicitly.

Usage
-----
>>> from traffic_rl.observability.context import bind_context, get_context
>>> bind_context(run_id="2026-05-01T10-00-00_abcd1234", agent="ppo")
>>> get_context()["agent"]
'ppo'
"""

from __future__ import annotations

import os
import socket
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping

    from traffic_rl.observability.typing import ContextDict

#: Context keys that are always present on a log record when bound.
CONTEXT_KEYS: tuple[str, ...] = (
    "run_id",
    "experiment",
    "agent",
    "scenario_id",
    "episode",
    "global_step",
    "split",
    "trace_id",
    "span",
    "host",
    "pid",
)

_CONTEXT: ContextVar[ContextDict] = ContextVar("traffic_rl_context")


def _new_dict() -> ContextDict:
    return {}


_CONTEXT.set(_new_dict())


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str | None = None) -> str:
    """Create a sortable, unique run identifier.

    Format: ``<UTC timestamp>_<8 hex chars>`` (optionally ``<prefix>_...``).
    Sortable as a string, which keeps S3 listings chronological.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{stamp}_{suffix}" if prefix else f"{stamp}_{suffix}"


def new_trace_id() -> str:
    """Return a new 16-hex-character correlation id."""
    return uuid.uuid4().hex[:16]


def get_context() -> ContextDict:
    """Return a *copy* of the current context (safe to mutate)."""
    return dict(_CONTEXT.get())


def bind_context(**values: Any) -> ContextDict:
    """Merge *values* into the context and return the new context.

    ``None`` values are ignored so callers can pass optional fields directly.
    """
    current = dict(_CONTEXT.get())
    current.update({k: v for k, v in values.items() if v is not None})
    _CONTEXT.set(current)
    return current


def unbind_context(*keys: str) -> ContextDict:
    """Remove *keys* from the context (missing keys are ignored)."""
    current = dict(_CONTEXT.get())
    for key in keys:
        current.pop(key, None)
    _CONTEXT.set(current)
    return current


def clear_context() -> None:
    """Drop every bound value (used between runs and in tests)."""
    _CONTEXT.set(_new_dict())


def default_context(**values: Any) -> ContextDict:
    """Return a context pre-populated with host/pid metadata."""
    base: ContextDict = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "trace_id": new_trace_id(),
    }
    base.update({k: v for k, v in values.items() if v is not None})
    return base


@contextmanager
def run_context(**values: Any) -> Iterator[ContextDict]:
    """Bind *values* for the duration of the ``with`` block, then restore.

    This is the recommended way to wrap a training/evaluation run::

        with run_context(run_id=new_run_id(), agent="ppo") as ctx:
            ...  # every log line carries ctx
    """
    token = _CONTEXT.set({**default_context(), **dict(_CONTEXT.get())})
    try:
        yield bind_context(**values) if values else get_context()
    finally:
        _CONTEXT.reset(token)


def context_snapshot(**extra: Any) -> ContextDict:
    """Return the current context merged with *extra* (for log records)."""
    snapshot = dict(_CONTEXT.get())
    snapshot.update({k: v for k, v in extra.items() if v is not None})
    return snapshot


def context_from_mapping(mapping: Mapping[str, Any]) -> ContextDict:
    """Project only the :data:`CONTEXT_KEYS` out of *mapping*."""
    return {key: mapping[key] for key in CONTEXT_KEYS if key in mapping}
