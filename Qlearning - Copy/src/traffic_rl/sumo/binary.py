"""Locate the SUMO binaries and start a ``traci`` connection.

SUMO must be installed on the host and either its bin directory must be on
``PATH`` or the ``SUMO_HOME`` environment variable must point to the install
directory.  When neither is available we raise a clear, actionable error
instead of failing deep inside ``traci.start``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SUMO_HOME_ENV = "SUMO_HOME"


class SUMONotFoundError(RuntimeError):
    """Raised when the SUMO binaries cannot be located on the host."""


def find_sumo_binary() -> Path:
    """Return the path to the ``sumo`` executable.

    Resolution order:
    1. ``SUMO_HOME`` environment variable -> ``bin/sumo``.
    2. ``SUMO_HOME`` environment variable -> ``bin/sumo-gui`` for GUI mode.
    3. Executable found on ``PATH`` (via :func:`shutil.which`).
    """
    sumo_home = os.environ.get(SUMO_HOME_ENV)
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / _binary_name()
        if candidate.exists():
            return candidate

    located = shutil.which(_binary_name())
    if located:
        return Path(located)

    raise SUMONotFoundError(
        "Could not find the SUMO executable. Either:\n"
        f"  * set the {SUMO_HOME_ENV} environment variable to your SUMO install "
        "directory, or\n"
        "  * add your SUMO `bin/` directory to the PATH.\n"
        f"  Searched PATH and ${SUMO_HOME_ENV}={sumo_home or '<unset>'}"
    )


def _binary_name() -> str:
    return "sumo-gui.exe" if os.name == "nt" else "sumo-gui"

def _non_gui_name() -> str:
    return "sumo.exe" if os.name == "nt" else "sumo"


def build_sumo_command(config_path: Path, *, gui: bool, seed: int | None,
                       max_steps: int, end: int | None = None) -> list[str]:
    """Build the command-line argument list passed to ``traci.start``."""
    binary = find_sumo_binary() if gui else _non_gui_binary()
    cmd = [
        str(binary),
        "-c", str(config_path),
        "--no-warnings", "true",
        "--start", "true",
        "--quit-on-end", "true",
        "--step-length", "1.0",
        "--time-to-teleport", "300",
        "--end", str(end if end is not None else max_steps),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if gui:
        cmd += ["--delay", "100"]
    return cmd


def _non_gui_binary() -> Path:
    sumo_home = os.environ.get(SUMO_HOME_ENV)
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / _non_gui_name()
        if candidate.exists():
            return candidate
    located = shutil.which(_non_gui_name())
    if located:
        return Path(located)
    raise SUMONotFoundError(
        "Could not find the `sumo` executable. Set SUMO_HOME or add SUMO's "
        "`bin/` directory to PATH."
    )


def _non_gui_name() -> str:
    return "sumo.exe" if os.name == "nt" else "sumo"
