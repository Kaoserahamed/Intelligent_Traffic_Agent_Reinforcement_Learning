"""Metrics sinks: where training/evaluation numbers are written.

One interface, several destinations:

============== ==========================================================
Sink           Destination
============== ==========================================================
JSONL          ``runs/<run_id>/metrics/metrics.jsonl`` (one object/line)
JSON artefact  single-file snapshot for CI quality gates
TensorBoard    ``runs/<run_id>/tensorboard/`` for curves
SageMaker      stdout line-JSON, parsed by SageMaker metric definitions
CloudWatch     ``PutMetricData`` (dashboards + alarms)
Null           no-op (unit tests, offline dry runs)
Multi          fan-out, with per-sink error isolation
============== ==========================================================

Sinks are best-effort by design: observability must never abort a run, so a
failing destination is logged once and disabled rather than raised.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from traffic_rl.io_utils import save_json
from traffic_rl.logging import get_logger
from traffic_rl.observability.context import get_context, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

log = get_logger("traffic_rl.observability.sinks")


@runtime_checkable
class MetricsSink(Protocol):
    """Minimal contract every metrics destination implements."""

    def log_scalars(
        self,
        step: int,
        metrics: Mapping[str, float],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a set of scalar metrics at ``step``."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record the run's hyper-parameters once, for comparison."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


def _as_number(value: Any) -> Any:
    """Coerce numpy scalars to plain floats/ints for JSON and dashboards."""
    if isinstance(value, bool):
        return value
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:  # pragma: no cover - defensive
        return value
    return value


class NullSink:
    """Discard everything (default for tests and ``--dry-run``)."""

    def log_scalars(self, step, metrics, *, context=None) -> None:
        return None

    def log_params(self, params) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlMetricsSink:
    """Append newline-delimited JSON records (grep-able, S3/Athena friendly)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = 0

    def log_scalars(self, step, metrics, *, context=None) -> None:
        record: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "step": int(step),
            **{str(k): _as_number(v) for k, v in metrics.items()},
        }
        record.update(context or get_context())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        self._records += 1

    def log_params(self, params) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "type": "params",
            "params": dict(params),
            **get_context(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def close(self) -> None:
        log.debug("jsonl metrics sink closed (%d records) -> %s", self._records, self.path)

    def __len__(self) -> int:
        return self._records


class JsonArtifactSink:
    """Keep the latest metrics snapshot in memory and write it as one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.latest: dict[str, Any] = {}

    def log_scalars(self, step, metrics, *, context=None) -> None:
        self.latest = {
            "step": int(step),
            "metrics": {str(k): _as_number(v) for k, v in metrics.items()},
            "timestamp": utc_now_iso(),
        }

    def log_params(self, params) -> None:
        self.latest["params"] = dict(params)

    def close(self) -> None:
        if self.latest:
            save_json(self.latest, self.path)


class TensorBoardSink:
    """TensorBoard curves (``torch.utils.tensorboard``; optional dependency)."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self._writer: Any | None = None

    def _ensure_writer(self) -> Any | None:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
                log.info("tensorboard logging enabled: %s", self.log_dir)
            except Exception as exc:
                log.warning("tensorboard unavailable, disabling sink: %s", exc)
                self._writer = False
        return self._writer or None

    def log_scalars(self, step, metrics, *, context=None) -> None:
        writer = self._ensure_writer()
        if writer is None:
            return
        prefix = (context or get_context()).get("agent") or ""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else str(key)
            try:
                writer.add_scalar(tag, _as_number(value), int(step))
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("tensorboard add_scalar failed for %s: %s", tag, exc)

    def log_params(self, params) -> None:
        writer = self._ensure_writer()
        if writer is None:
            return
        try:
            writer.add_text("params", json.dumps(dict(params), indent=2, default=str), 0)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("could not write params to tensorboard: %s", exc)

    def close(self) -> None:
        if self._writer and self._writer is not False:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("error closing tensorboard writer: %s", exc)
        self._writer = None


class SageMakerSink:
    """Emit metric lines to stdout in the format SageMaker Training parses.

    SageMaker scrapes stdout for ``{...}`` JSON objects containing numeric
    values; the regex metric definitions live in
    ``src/traffic_rl/cloud/aws/job_specs/*.json``.
    """

    def __init__(self, stream: Any | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def log_scalars(self, step, metrics, *, context=None) -> None:
        payload = {
            "step": int(step),
            **{str(k): _as_number(v) for k, v in metrics.items()},
        }
        try:
            print(json.dumps(payload, default=str), file=self.stream, flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("sagemaker sink write failed: %s", exc)

    def log_params(self, params) -> None:
        try:
            print(json.dumps({"params": dict(params)}, default=str), file=self.stream, flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("sagemaker sink params write failed: %s", exc)

    def close(self) -> None:
        try:
            self.stream.flush()
        except Exception:  # pragma: no cover - defensive
            pass


class MultiSink:
    """Fan-out sink that isolates failures per destination."""

    def __init__(self, sinks: Sequence[MetricsSink] = ()) -> None:
        self.sinks: list[MetricsSink] = list(sinks)
        self._failed: set[int] = set()

    def add(self, sink: MetricsSink) -> None:
        self.sinks.append(sink)

    def log_scalars(self, step, metrics, *, context=None) -> None:
        for index, sink in enumerate(self.sinks):
            if index in self._failed:
                continue
            try:
                sink.log_scalars(step, metrics, context=context)
            except Exception as exc:
                self._failed.add(index)
                log.warning("metrics sink %s disabled after error: %s", type(sink).__name__, exc)

    def log_params(self, params) -> None:
        for index, sink in enumerate(self.sinks):
            if index in self._failed:
                continue
            try:
                sink.log_params(params)
            except Exception as exc:
                self._failed.add(index)
                log.warning("metrics sink %s disabled after error: %s", type(sink).__name__, exc)

    def close(self) -> None:
        for index, sink in enumerate(self.sinks):
            if index in self._failed:
                continue
            try:
                sink.close()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("error closing sink %s: %s", type(sink).__name__, exc)

    def __len__(self) -> int:
        return len(self.sinks)


def sink_from_env(env_var: str = "TRAFFIC_RL_SINKS") -> str:
    """Return the configured sink list (comma separated), default ``jsonl``."""
    return os.environ.get(env_var, "jsonl")
