"""Locate the SUMO binaries and build a ``traci`` command line.

SUMO must be installed on the host and either its ``bin`` directory must be on
``PATH`` or the ``SUMO_HOME`` environment variable must point to the install
directory.  When neither is available we raise a clear, actionable error
instead of failing deep inside ``traci.start``.

The command builder is deliberately pure: it only formats an argument list, so
unit tests can assert on it without SUMO being installed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from traffic_rl.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

log = get_logger("traffic_rl.sumo.binary")

SUMO_HOME_ENV = "SUMO_HOME"
SUMO_BINARY_ENV = "SUMO_BINARY"
SUMO_GUI_BINARY_ENV = "SUMO_GUI_BINARY"


class SUMONotFoundError(RuntimeError):
    """Raised when the SUMO binaries cannot be located on the host."""


def _exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _binary_filename(gui: bool = False) -> str:
    """Return the SUMO executable filename for the current platform."""
    return f"sumo-gui{_exe_suffix()}" if gui else f"sumo{_exe_suffix()}"


def find_sumo_binary(gui: bool = False) -> Path:
    """Return the path to the ``sumo`` (or ``sumo-gui``) executable.

    Resolution order:

    1. ``SUMO_BINARY`` / ``SUMO_GUI_BINARY`` explicit override.
    2. ``SUMO_HOME`` environment variable -> ``bin/<binary>``.
    3. Executable found on ``PATH`` (via :func:`shutil.which`).

    Parameters
    ----------
    gui:
        Look for ``sumo-gui`` instead of the headless ``sumo`` binary.
    """
    override = os.environ.get(SUMO_GUI_BINARY_ENV if gui else SUMO_BINARY_ENV)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise SUMONotFoundError(
            f"{SUMO_GUI_BINARY_ENV if gui else SUMO_BINARY_ENV} points at "
            f"{candidate}, which does not exist."
        )

    filename = _binary_filename(gui=gui)
    sumo_home = os.environ.get(SUMO_HOME_ENV)
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / filename
        if candidate.exists():
            return candidate
        log.debug("SUMO_HOME=%s has no %s", sumo_home, filename)

    located = shutil.which(filename)
    if located:
        return Path(located)

    raise SUMONotFoundError(
        "Could not find the SUMO executable.\n"
        "Either:\n"
        f"  * set the {SUMO_HOME_ENV} environment variable to your SUMO install "
        "directory, or\n"
        f"  * set {SUMO_BINARY_ENV}/{SUMO_GUI_BINARY_ENV} to an explicit path, or\n"
        "  * add your SUMO `bin/` directory to the PATH.\n"
        f"  Searched PATH and ${SUMO_HOME_ENV}={sumo_home or '<unset>'} for {filename}"
    )


def sumo_version(sumo_binary: Path | None = None) -> str | None:
    """Return SUMO's reported version string, or ``None`` when unavailable.

    Best-effort only: used for run manifests and reproducibility metadata.
    """
    import subprocess

    try:
        binary = sumo_binary or find_sumo_binary()
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, SUMONotFoundError) as exc:
        log.debug("could not query SUMO version: %s", exc)
        return None
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if not output:
        return None
    for line in output.splitlines():
        if "SUMO" in line:
            return line.strip()
    return output.splitlines()[0].strip()


def build_sumo_command(
    config_path: Path,
    *,
    gui: bool = False,
    seed: int | None = None,
    max_steps: int | None = None,
    end: float | None = None,
    step_length: float = 1.0,
    scale: float = 1.0,
    time_to_teleport: int = 300,
    collision_action: str = "teleport",
    tripinfo_output: Path | None = None,
    summary_output: Path | None = None,
    additional_files: Sequence[Path] = (),
    remote_port: int | None = None,
    random_departure: bool = False,
    extra_args: Iterable[str] = (),
    quit_on_end: bool = True,
) -> list[str]:
    """Build the command-line argument list passed to ``traci.start``.

    ``end`` is the simulation end time in **simulated seconds**.  When omitted
    it is derived from ``max_steps`` and ``step_length`` (``max_steps *
    step_length``); the historic implementation divided by 1000, which
    silently truncated every episode to 150 simulated seconds.
    """
    binary = find_sumo_binary(gui=gui)
    if end is None:
        if max_steps is None:
            raise ValueError("either `end` (simulated seconds) or `max_steps` must be given")
        end = float(max_steps) * float(step_length)

    cmd: list[str] = [
        str(binary),
        "-c",
        str(config_path),
        "--no-warnings",
        "true",
        "--start",
        "true",
        "--step-length",
        str(step_length),
        "--time-to-teleport",
        str(time_to_teleport),
        "--collision.action",
        str(collision_action),
        "--scale",
        str(scale),
        "--end",
        str(end),
    ]
    if quit_on_end:
        cmd += ["--quit-on-end", "true"]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if random_departure:
        cmd += ["--random-departure", "true"]
    if additional_files:
        cmd += ["--additional-files", ",".join(str(p) for p in additional_files)]
    if tripinfo_output is not None:
        cmd += ["--tripinfo-output", str(tripinfo_output)]
    if summary_output is not None:
        cmd += ["--summary-output", str(summary_output)]
    if remote_port is not None:
        cmd += ["--remote-port", str(int(remote_port))]
    if gui:
        cmd += ["--delay", "100"]
    cmd += [str(arg) for arg in extra_args]
    return cmd


def resolve_sumo_config_path(config_path: Path, run_dir: Path | None) -> Path:
    """Resolve a SUMO config path for launching.

    ``traci`` resolves the ``-c`` argument from the process working directory,
    so a config generated for a scenario run must be expanded to an absolute
    path when the run directory differs from the CWD.
    """
    path = Path(config_path)
    if run_dir is not None and not path.is_absolute():
        path = Path(run_dir) / path
    return path
