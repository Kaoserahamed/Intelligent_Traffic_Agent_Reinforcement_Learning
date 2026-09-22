"""Pytest plugin that emits a structured JSON report for CI.

Usage
-----
Registered automatically through the repository-level ``conftest.py``::

    pytest --structured-report reports/test-report.json

The report is the machine-readable counterpart of the human pytest output: CI
uploads it as an artefact, diffs it between runs and uses it for flaky-test
detection.  It contains:

* ``environment`` — python/torch/SUMO versions, platform, git sha/branch
* ``summary`` — totals, pass rate, wall time, exit status
* ``tests`` — one record per test (nodeid, outcome, phase, duration, markers,
  failure message and location)
* ``slowest`` — the N slowest tests
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pytest

DEFAULT_REPORT_ENV = "TEST_REPORT_PATH"

#: Markers recognised in the report (kept in sync with pyproject markers).
KNOWN_MARKERS = frozenset(
    {"unit", "integration", "e2e", "slow", "sumo", "gpu", "aws", "network", "performance"}
)


@dataclass
class TestRecord:
    """Outcome of a single pytest test item."""

    nodeid: str
    outcome: str  # passed | failed | error | skipped | xfailed | xpassed
    duration_s: float = 0.0
    phases: dict[str, str] = field(default_factory=dict)
    markers: list[str] = field(default_factory=list)
    message: str = ""
    location: str = ""
    when_failed: str = ""


@dataclass
class StructuredReport:
    """Whole-session report."""

    run_id: str
    started_at: str
    duration_s: float
    exit_status: int
    summary: dict[str, Any]
    environment: dict[str, Any]
    tests: list[dict[str, Any]]
    slowest: list[dict[str, Any]]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return target


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the plugin's command-line options."""
    group = parser.getgroup("traffic-rl")
    group.addoption(
        "--structured-report",
        action="store",
        default=os.environ.get(DEFAULT_REPORT_ENV),
        metavar="PATH",
        help="Write a structured JSON test report to PATH (env: TEST_REPORT_PATH).",
    )
    group.addoption(
        "--report-run-id",
        action="store",
        default=os.environ.get("CI_RUN_ID", ""),
        help="Identifier stamped into the structured report (usually the CI run id).",
    )


class StructuredTestReporter:
    """Collect test outcomes and write one JSON document at session end."""

    def __init__(self, path: str, run_id: str = "") -> None:
        self.path = Path(path)
        self.run_id = run_id or os.environ.get("CI_RUN_ID", "local")
        self.records: dict[str, TestRecord] = {}
        self.started = time.time()
        self.started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # -- pytest hooks ----------------------------------------------------------

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        record = self.records.get(report.nodeid)
        if record is None:
            record = TestRecord(
                nodeid=report.nodeid,
                outcome="unknown",
                markers=_markers(report),
                location=str(getattr(report, "location", "")),
            )
            self.records[report.nodeid] = record
        record.phases[report.when] = report.outcome
        record.duration_s += float(getattr(report, "duration", 0.0) or 0.0)
        if report.failed:
            record.outcome = "error" if report.when in {"setup", "teardown"} else "failed"
            record.when_failed = report.when
            record.message = _short_message(report)
        elif report.skipped and record.outcome == "unknown":
            record.outcome = "skipped"
            record.message = _short_message(report)
        elif record.outcome == "unknown":
            record.outcome = report.outcome
        if report.when == "teardown" and record.outcome in {"passed", "unknown"}:
            record.outcome = "passed" if report.passed else record.outcome

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        tests = sorted(self.records.values(), key=lambda r: r.nodeid)
        slowest = sorted(tests, key=lambda r: r.duration_s, reverse=True)[:10]
        report = StructuredReport(
            run_id=self.run_id,
            started_at=self.started_iso,
            duration_s=round(time.time() - self.started, 3),
            exit_status=int(exitstatus),
            summary=_summarize(tests),
            environment=collect_environment(),
            tests=[asdict(record) for record in tests],
            slowest=[
                {"nodeid": r.nodeid, "outcome": r.outcome, "duration_s": round(r.duration_s, 4)}
                for r in slowest
            ],
        )
        try:
            written = report.write(self.path)
            print(f"\n[traffic-rl] structured test report -> {written}", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - never fail a run for a report
            print(f"[traffic-rl] could not write structured report: {exc}", file=sys.stderr)


def pytest_configure(config: pytest.Config) -> None:
    """Attach the reporter when ``--structured-report`` is provided."""
    path = config.getoption("--structured-report", default=None)
    if not path:
        return
    run_id = config.getoption("--report-run-id", default="") or "local"
    reporter = StructuredTestReporter(path=str(path), run_id=str(run_id))
    config.pluginmanager.register(reporter, "traffic-rl-structured-reporter")


def _markers(report: pytest.TestReport) -> list[str]:
    keywords = getattr(report, "keywords", None)
    if not keywords:
        return []
    return sorted(set(keywords) & KNOWN_MARKERS)


def _short_message(report: pytest.TestReport) -> str:
    text = getattr(report, "longreprtext", "") or ""
    if not text:
        return ""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[:12])[:2000]


def _summarize(tests: list[TestRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in tests:
        counts[record.outcome] = counts.get(record.outcome, 0) + 1
    total = len(tests)
    passed = counts.get("passed", 0)
    return {
        "total": total,
        "passed": passed,
        "failed": counts.get("failed", 0),
        "errors": counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "xfailed": counts.get("xfailed", 0),
        "xpassed": counts.get("xpassed", 0),
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "duration_s": round(sum(r.duration_s for r in tests), 3),
    }


def _safe_version(module_name: str, attr: str = "__version__") -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, attr, "unknown"))
    except Exception:
        return "unavailable"


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def collect_environment() -> dict[str, Any]:
    """Return environment metadata for the report (never raises)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": _safe_version("torch"),
        "numpy": _safe_version("numpy"),
        "traci": _safe_version("traci"),
        "pytest": _safe_version("pytest"),
        "sumo_home": os.environ.get("SUMO_HOME", ""),
        "traffic_rl_env": os.environ.get("TRAFFIC_RL_ENV", "dev"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "ci_run_id": os.environ.get("CI_RUN_ID", ""),
        "ci_workflow": os.environ.get("GITHUB_WORKFLOW", ""),
    }
