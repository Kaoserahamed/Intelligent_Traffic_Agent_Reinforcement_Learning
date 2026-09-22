"""Tests for the observability package (logging, sinks, events, profiling)."""

from __future__ import annotations

import json
import logging

import pytest

from traffic_rl.observability.context import (
    CONTEXT_KEYS,
    bind_context,
    clear_context,
    context_snapshot,
    get_context,
    run_context,
    unbind_context,
)
from traffic_rl.observability.events import EVENT_CATALOGUE, log_event, validate_event
from traffic_rl.observability.logging_config import (
    ConsoleFormatter,
    JsonFormatter,
    build_formatter,
    configure_structured_logging,
    redact_text,
)
from traffic_rl.observability.profiler import PhaseTimer, StepRateMeter, process_rss_mb


class TestContext:
    """Tests for context propagation."""

    def test_bind_context(self):
        clear_context()
        ctx = bind_context(run_id="test-123", agent="ppo", episode=5)
        assert ctx["run_id"] == "test-123"
        assert ctx["agent"] == "ppo"
        assert ctx["episode"] == 5

    def test_get_context_returns_copy(self):
        clear_context()
        bind_context(run_id="abc")
        ctx1 = get_context()
        ctx1["run_id"] = "modified"
        ctx2 = get_context()
        assert ctx2["run_id"] == "abc"

    def test_context_snapshot(self):
        clear_context()
        bind_context(run_id="snap-1")
        snapshot = context_snapshot(extra_key="extra_value")
        assert snapshot["run_id"] == "snap-1"
        assert snapshot["extra_key"] == "extra_value"

    def test_clear_context(self):
        clear_context()
        bind_context(run_id="to-clear")
        clear_context()
        assert get_context() == {}

    def test_run_context_restores(self):
        clear_context()
        bind_context(run_id="outer")
        assert get_context()["run_id"] == "outer"
        with run_context(run_id="inner", agent="dqn"):
            assert get_context()["run_id"] == "inner"
            assert get_context()["agent"] == "dqn"
        assert get_context()["run_id"] == "outer"

    def test_unbind_context(self):
        clear_context()
        bind_context(run_id="r1", agent="ppo", episode=10)
        ctx = unbind_context("agent", "episode")
        assert "agent" not in ctx
        assert "episode" not in ctx
        assert ctx["run_id"] == "r1"

    def test_none_values_ignored(self):
        clear_context()
        bind_context(run_id="ok", agent=None, episode=None)
        ctx = get_context()
        assert ctx["run_id"] == "ok"
        assert "agent" not in ctx

    def test_context_keys_present(self):
        assert "run_id" in CONTEXT_KEYS
        assert "agent" in CONTEXT_KEYS
        assert "experiment" in CONTEXT_KEYS


class TestEvents:
    """Tests for the event catalogue and logging."""

    def test_event_catalogue_has_expected_events(self):
        assert EVENT_CATALOGUE["training.episode.end"]
        assert EVENT_CATALOGUE["run.start"]
        assert EVENT_CATALOGUE["eval.report.written"]
        assert EVENT_CATALOGUE["config.resolved"]

    def test_validate_event_missing_fields(self):
        missing = validate_event("training.episode.end", {"run_id": "123"})
        assert "episode" in missing
        assert "episode_reward" in missing

    def test_validate_event_complete(self):
        missing = validate_event(
            "training.episode.end",
            {"run_id": "123", "episode": 1, "episode_reward": -0.5, "steps": 100},
        )
        assert missing == []

    def test_validate_unknown_event(self):
        missing = validate_event("nonexistent.event", {})
        assert missing == []

    def test_log_event_emits_structured_record(self, caplog):
        clear_context()
        bind_context(run_id="evt-test", agent="dqn")
        logger = logging.getLogger("traffic_rl.test")
        with caplog.at_level(logging.INFO):
            log_event(
                logger,
                "training.episode.end",
                message="Episode complete",
                run_id="evt-test",
                episode=1,
                episode_reward=42.0,
                steps=100,
            )
        record = caplog.records[-1]
        assert record.structured["event"] == "training.episode.end"
        assert hasattr(record, "structured")
        assert record.structured["event"] == "training.episode.end"


class TestLoggingConfig:
    """Tests for structured logging and redaction."""

    def test_redact_secret_patterns(self):
        text = "aws_access_key: AKIAIOSFODNN7EXAMPLE1234"
        redacted = redact_text(text)
        assert "***REDACTED***" in redacted or "AKIAIOSFODNN7EXAMPLE1234" not in redacted

    def test_redact_bearer_token(self):
        text = "Authorization: Bearer abc123def456ghi789jkl012"
        redacted = redact_text(text)
        assert "***REDACTED***" in redacted or "abc123def456" not in redacted

    def test_redact_explicit_secret(self):
        text = "password=supersecret123"
        redacted = redact_text(text, extra_secrets=["supersecret123"])
        assert "supersecret123" not in redacted
        assert "***REDACTED***" in redacted

    def test_redact_short_secret_ignored(self):
        text = "token=ab"
        redacted = redact_text(text, extra_secrets=["ab"])
        assert "ab" in redacted

    def test_build_formatter_json(self):
        assert isinstance(build_formatter("json"), JsonFormatter)

    def test_build_formatter_console(self):
        assert isinstance(build_formatter("console"), ConsoleFormatter)

    def test_build_formatter_invalid(self):
        """Unknown formats fall back to console formatter."""
        formatter = build_formatter(fmt="bogus")
        assert isinstance(formatter, ConsoleFormatter)

    def test_configure_logging_returns_logger(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = configure_structured_logging(
            level="DEBUG", fmt="console", log_file=str(log_file), console=False
        )
        assert isinstance(logger, logging.Logger)
        logger.info("test message")
        assert log_file.exists()

    def test_json_formatter_output(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="traffic_rl.test", level=logging.INFO, pathname="",
            lineno=1, msg="Test message", args=(), exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"


class TestProfiler:
    """Tests for performance instrumentation."""

    def test_phase_timer_context(self):
        timer = PhaseTimer()
        with timer.phase("test.phase"):
            pass
        report = timer.report()
        assert "test.phase" in report
        assert report["test.phase"]["count"] == 1

    def test_phase_timer_timed_decorator(self):
        timer = PhaseTimer()

        @timer.timed("decorated")
        def my_func():
            return 42

        assert my_func() == 42
        assert "decorated" in timer.report()

    def test_step_rate_meter_average(self):
        meter = StepRateMeter()
        meter.tick(10)
        assert meter.steps == 10
        assert meter.average >= 0.0

    def test_step_rate_meter_snapshot(self):
        meter = StepRateMeter()
        for _ in range(5):
            meter.tick()
        snapshot = meter.snapshot()
        assert "steps_per_second" in snapshot
        assert "wall_time_s" in snapshot
        assert snapshot["global_step"] == 5.0

    def test_step_rate_meter_instantaneous(self):
        meter = StepRateMeter()
        assert meter.instantaneous == 0.0
        meter.tick()
        meter.tick()
        assert meter.instantaneous >= 0.0

    def test_process_rss_mb_returns_float(self):
        rss = process_rss_mb()
        assert isinstance(rss, float)
        assert rss >= 0.0


class TestSinks:
    """Tests for metrics sink implementations."""

    def test_jsonl_sink_writes_records(self, tmp_path):
        from traffic_rl.observability.sinks import JsonlMetricsSink

        path = tmp_path / "metrics.jsonl"
        sink = JsonlMetricsSink(path)
        sink.log_scalars(0, {"loss": 0.5, "reward": 10.0})
        sink.log_scalars(1, {"loss": 0.3, "reward": 15.0})
        sink.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        record0 = json.loads(lines[0])
        assert record0["loss"] == 0.5
        assert record0["step"] == 0

    def test_null_sink_no_op(self):
        from traffic_rl.observability.sinks import NullSink

        sink = NullSink()
        sink.log_scalars(0, {"loss": 0.5})
        sink.log_params({"lr": 0.001})
        sink.close()

    def test_multi_sink_fanout(self, tmp_path):
        from traffic_rl.observability.sinks import JsonlMetricsSink, MultiSink

        path1 = tmp_path / "sink1.jsonl"
        path2 = tmp_path / "sink2.jsonl"
        sink = MultiSink([JsonlMetricsSink(path1), JsonlMetricsSink(path2)])
        sink.log_scalars(0, {"loss": 0.5})
        sink.close()
        assert path1.exists() and path2.exists()
