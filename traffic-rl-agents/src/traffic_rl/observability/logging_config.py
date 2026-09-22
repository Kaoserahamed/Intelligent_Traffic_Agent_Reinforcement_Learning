"""Structured logging: JSON records, human console output, redaction.

Design
------
* Standard library :mod:`logging` only (no extra runtime dependency).
* One :class:`JsonFormatter` for machine consumption (CloudWatch / Athena /
  OpenSearch / log-based metrics) and one :class:`ConsoleFormatter` for humans.
* Every record is enriched from :mod:`traffic_rl.observability.context`, so
  ``run_id``/``agent``/``episode`` are present without being passed around.
* :class:`RedactionFilter` scrubs credentials before they can reach disk or
  CloudWatch, using both regex patterns and the concrete secret values resolved
  from AWS Secrets Manager / SSM at run time.

Configuration precedence is resolved by :mod:`traffic_rl.settings`:
``configs/logging.yaml`` -> ``LOG_LEVEL``/``LOG_FORMAT`` env vars -> explicit
arguments to :func:`configure_structured_logging`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from traffic_rl.observability.context import context_snapshot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

#: Attributes present on every ``LogRecord``; anything else is "extra" data.
_RESERVED_ATTRS: frozenset[str] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
        "structured",
    }
)

#: Keys that must never be written to logs in clear text.
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "aws_access_key_id", "aws_secret_access_key", "aws_session_token",
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "authorization", "private_key", "client_secret", "webhook_url",
        "wandb_api_key", "mlflow_tracking_token", "hf_token",
    }
)

#: Regex fallbacks for credentials embedded in free-text messages.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(aws_?(access|secret)[a-z_]*['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9/+=]{16,})"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{20,})"),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9._\-]{16,})"),
    re.compile(
        r"(?i)((?:secret|token|api[_-]?key)['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9/+=_\-]{12,})"
    ),
)

_REDACTED = "***REDACTED***"

#: Logger names that are silenced by default (third-party noise).
NOISY_LOGGERS: tuple[str, ...] = (
    "traci", "matplotlib", "PIL", "botocore", "boto3", "urllib3", "s3transfer",
    "filelock", "git",
)


def redact_text(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Return *text* with credential-looking substrings masked."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}{_REDACTED}", redacted)
    for secret in extra_secrets:
        if secret and len(secret) >= 8:
            redacted = redacted.replace(secret, _REDACTED)
    return redacted


def redact_mapping(payload: Mapping[str, Any], extra_secrets: Iterable[str] = ()) -> dict:
    """Recursively mask values whose key looks sensitive (and free text)."""
    secrets = tuple(extra_secrets)
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in _SENSITIVE_KEYS:
            clean[key] = _REDACTED
        elif isinstance(value, dict):
            clean[key] = redact_mapping(value, secrets)
        elif isinstance(value, str):
            clean[key] = redact_text(value, secrets)
        else:
            clean[key] = value
    return clean


class RedactionFilter(logging.Filter):
    """Scrub credentials from ``msg``/``args``/extra payloads in place."""

    def __init__(self, extra_secrets: Sequence[str] = ()) -> None:
        super().__init__()
        self.extra_secrets = tuple(extra_secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(str(record.msg), self.extra_secrets)
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_text(str(a), self.extra_secrets) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_text(v, self.extra_secrets) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            structured = getattr(record, "structured", None)
            if isinstance(structured, dict):
                record.structured = redact_mapping(structured, self.extra_secrets)
        except Exception:  # pragma: no cover - never break logging
            return True
        return True


class ContextFilter(logging.Filter):
    """Attach the current observability context to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in context_snapshot().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects (one per ``\\n``)."""

    def __init__(self, *, include_extras: bool = True) -> None:
        super().__init__()
        self.include_extras = include_extras

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _iso_from_epoch(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key in _RESERVED_ATTRS or key in payload or key.startswith("_"):
                    continue
                payload[key] = _jsonify(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line output for terminals and CI logs."""

    DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(self.DEFAULT_FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            event = structured.get("event")
            extras = " ".join(f"{k}={v}" for k, v in structured.items() if k != "event")
            base = f"{base} | event={event} {extras}".rstrip()
        context_bits = [
            f"{key}={getattr(record, key)}"
            for key in ("run_id", "agent", "episode")
            if getattr(record, key, None) is not None
        ]
        if context_bits:
            base = f"{base} | {' '.join(context_bits)}"
        return base


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds")


def _jsonify(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_formatter(fmt: str = "json", **kwargs: Any) -> logging.Formatter:
    """Return a formatter for ``fmt`` in ``{"json", "console", "text"}``."""
    normalized = (fmt or "json").strip().lower()
    if normalized == "json":
        return JsonFormatter(**kwargs)
    return ConsoleFormatter()


def configure_structured_logging(
    *,
    level: int | str = logging.INFO,
    fmt: str = "json",
    log_file: str | Path | None = None,
    console: bool = True,
    extra_secrets: Sequence[str] = (),
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    use_queue: bool = False,
    silence_noisy: bool = True,
) -> logging.Logger:
    """Configure the ``traffic_rl`` logger tree for structured output.

    Idempotent: existing handlers on the package logger are removed first, so
    logging can be reconfigured once the run directory is known (the run id,
    agent and episode are then present on every subsequent record).

    Parameters
    ----------
    level:
        ``logging`` level or its name (``"DEBUG"``).
    fmt:
        ``"json"`` for machine-readable records (default) or ``"console"``.
    log_file:
        Optional rotating log file (``runs/<run_id>/logs/train.log.jsonl``).
    extra_secrets:
        Concrete secret values to mask (typically resolved from AWS).
    use_queue:
        Move file/stream writes to a background listener thread — recommended
        for long SUMO runs where per-step logging can dominate wall time.
    """
    package_logger = logging.getLogger("traffic_rl")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()

    package_logger.setLevel(_coerce_level(level))
    formatter = build_formatter(fmt)
    filters: list[logging.Filter] = [ContextFilter(), RedactionFilter(extra_secrets)]

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        _attach_filters(stream_handler, filters)
        package_logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        _attach_filters(file_handler, filters)
        package_logger.addHandler(file_handler)

    if use_queue:
        _install_queue_handler(package_logger, filters)

    package_logger.propagate = False

    if silence_noisy:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    return package_logger


def _attach_filters(handler: logging.Handler, filters: Sequence[logging.Filter]) -> None:
    for log_filter in filters:
        handler.addFilter(log_filter)


def _install_queue_handler(
    logger: logging.Logger,
    filters: Sequence[logging.Filter],
) -> logging.handlers.QueueListener:
    """Move blocking I/O off the training thread via a queue listener."""
    log_queue: queue.Queue[Any] = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    _attach_filters(queue_handler, filters)
    listeners = [
        handler
        for handler in logger.handlers
        if not isinstance(handler, logging.handlers.QueueHandler)
    ]
    listener = logging.handlers.QueueListener(
        log_queue, *listeners, respect_handler_level=True
    )
    listener.start()
    for handler in listeners:
        logger.removeHandler(handler)
    logger.addHandler(queue_handler)
    # Keep a reference so the listener is not garbage collected mid-run.
    logger.traffic_rl_queue_listener = listener  # type: ignore[attr-defined]
    return listener


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    text = str(level).strip().upper()
    if text.isdigit():
        return int(text)
    resolved = logging.getLevelName(text)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"unknown log level: {level!r}")


def get_structured_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``traffic_rl``."""
    if name and not name.startswith("traffic_rl"):
        return logging.getLogger(f"traffic_rl.{name}")
    return logging.getLogger(name or "traffic_rl")
