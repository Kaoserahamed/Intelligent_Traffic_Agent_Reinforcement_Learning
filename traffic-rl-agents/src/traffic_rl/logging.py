"""Centralised logging configuration.

Call :func:`configure_logging` once at application bootstrap (the CLI does
this for you).  Every other module should obtain its logger with::

    import logging
    log = logging.getLogger("traffic_rl.my_module")

and emit ``log.info(...)`` / ``log.error(...)`` rather than printing.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

_PACKAGE_LOGGER = "traffic_rl"
_DEFAULT_FMT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(verbose: bool = False, log_file: str | None = None) -> logging.Logger:
    """Configure the root ``traffic_rl`` logger.

    Parameters
    ----------
    verbose:
        If ``True`` use ``DEBUG`` level, otherwise ``INFO``.
    log_file:
        Optional path to an additional file handler (useful for capturing
        per-run training logs).
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger(_PACKAGE_LOGGER)
    root.setLevel(level)

    # Avoid attaching duplicate handlers on repeated calls (e.g. in tests).
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT, _DATE_FMT))
        root.addHandler(handler)

    if log_file:
        add_file_handler(log_file, level=level)

    # Don't propagate to the root logger (avoids duplicate output if the
    # application has configured the root logger elsewhere).
    root.propagate = False
    return root


def add_file_handler(log_file: str, level: int = logging.INFO) -> None:
    """Attach a file handler to the package logger."""
    root = logging.getLogger(_PACKAGE_LOGGER)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT, _DATE_FMT))
    root.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``traffic_rl``.

    Passing ``"traffic_rl"`` or ``None`` returns the package logger itself.
    """
    if name and not name.startswith(_PACKAGE_LOGGER):
        name = f"{_PACKAGE_LOGGER}.{name}" or _PACKAGE_LOGGER
    return logging.getLogger(name or _PACKAGE_LOGGER)


def silence_third_party(loggers: Iterable[str] = ("traci", "matplotlib", "PIL")) -> None:
    """Quiet noisy third-party loggers that flood stdout during training."""
    for name in loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
