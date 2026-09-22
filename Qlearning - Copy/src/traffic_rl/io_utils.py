"""Safe persistence helpers for agents.

Security note
-------------
The legacy codebase loaded checkpoints with ``pickle.load`` /
``torch.load`` *without* ``weights_only``.  Both deserialise arbitrary Python
objects and therefore execute attacker-controlled code if a checkpoint is
tampered with.  This module closes that hole:

* Q-Learning tables are persisted as **JSON** (no pickle at all), with a
  legacy ``.pkl`` → ``.json`` migration helper that uses a restricted
  :class:`pickle.Unpickler`.
* Torch checkpoints use ``torch.load(..., weights_only=True)`` and only return
  the standard checkpoint dict.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_rl.logging import get_logger

log = get_logger("io_utils")

# Restrict the torch ``SafeUnpickler`` to a small allow-list of modules that
# are legitimately used inside checkpoints.
_TORCH_WEIGHTS_ONLY = True


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses to load any class other than plain containers."""

    _ALLOWED = {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
    }

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle {module}.{name}: not in allow-list."
        )


def safe_torch_load(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load a torch checkpoint safely.

    Uses ``weights_only=True`` (torch >= 1.13) so only tensors and primitive
    types are restored — never arbitrary Python objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    log.debug("loading torch checkpoint (weights_only=True): %s", path)
    return torch.load(str(path), map_location=map_location, weights_only=_TORCH_WEIGHTS_ONLY)


def safe_torch_save(obj: dict, path: str | Path) -> None:
    """Persist a checkpoint dict with ``torch.save``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))
    log.debug("saved torch checkpoint: %s", path)


# -----------------------------------------------------------------------------
# Q-table (JSON based)
# -----------------------------------------------------------------------------

_QTable = dict[tuple[int, ...], list[float]]


def save_q_table(q_table: _QTable, path: str | Path, *, metadata: dict | None = None) -> None:
    """Persist a Q-table as JSON.

    Keys are integer tuples (state discretisation); the on-disk representation
    uses JSON arrays which ``json`` round-trips safely and cannot execute code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "json-qtable-v1",
        "metadata": metadata or {},
        "q_table": [{"state": list(k), "values": list(v)} for k, v in q_table.items()],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.debug("saved Q-table (%d states) to %s", len(q_table), path)


def load_q_table(path: str | Path) -> _QTable:
    """Load a Q-table written by :func:`save_q_table`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q-table not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    q_table: _QTable = {}
    for entry in payload["q_table"]:
        q_table[tuple(entry["state"])] = [float(x) for x in entry["values"]]
    log.debug("loaded Q-table (%d states) from %s", len(q_table), path)
    return q_table


def load_legacy_q_table_pickle(path: str | Path) -> _QTable:
    """Migrate a legacy ``.pkl`` Q-table using a restricted unpickler.

    Raises :class:`pickle.UnpicklingError` if the file references any non
    allow-listed class — this is the security boundary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q-table not found: {path}")
    with path.open("rb") as handle:
        data = _RestrictedUnpickler(handle).load()
    q_table: _QTable = {}
    for key, values in data["q_table"].items():
        q_table[tuple(int(x) for x in key)] = [float(x) for x in np.asarray(values).ravel()]
    log.debug("migrated legacy pickle Q-table (%d states) from %s", len(q_table), path)
    return q_table


def q_table_to_json_compat(q_table: _QTable) -> dict:
    """Convert a Q-table to a JSON-serialisable mapping (for in-memory use)."""
    return {"q_table": [{"state": list(k), "values": list(v)} for k, v in q_table.items()]}


def save_json(obj: Any, path: str | Path) -> None:
    """Best-effort JSON serialiser for arbitrary agent state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# Re-export for convenience.
__all__ = [
    "safe_torch_load",
    "safe_torch_save",
    "save_q_table",
    "load_q_table",
    "load_legacy_q_table_pickle",
    "q_table_to_json_compat",
    "save_json",
]
