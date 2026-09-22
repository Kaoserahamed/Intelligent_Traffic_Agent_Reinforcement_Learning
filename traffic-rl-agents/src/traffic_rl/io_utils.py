"""Safe persistence helpers for agents and checkpoints.

Security note
-------------
The legacy codebase loaded checkpoints with ``pickle.load`` /
``torch.load`` *without* ``weights_only``.  Both deserialise arbitrary Python
objects and therefore execute attacker-controlled code if a checkpoint is
tampered with.  This module closes that hole:

* Q-Learning tables are persisted as **JSON** (no pickle at all) with a
  SHA-256 integrity checksum, plus a legacy ``.pkl`` → ``.json`` migration
  helper driven by a restricted unpickler with an explicit allow-list.
* Torch checkpoints use ``torch.load(..., weights_only=True)`` and only ever
  return the standard checkpoint dict.  Failures are normalised to
  :class:`CheckpointError` so callers never have to catch pickle internals.

Every write is atomic (temp file + :func:`os.replace`) so an interrupted job can
never leave a half-written model behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_rl.logging import get_logger

log = get_logger("io_utils")

#: Torch checkpoints are always loaded with ``weights_only=True`` (torch >= 1.13).
_TORCH_WEIGHTS_ONLY = True

#: Schema version of the JSON Q-table payload.
Q_TABLE_FORMAT: str = "json-qtable-v2"
Q_TABLE_FORMAT_VERSION: int = 2

#: Schema version of torch checkpoints written by :func:`save_torch_checkpoint`.
CHECKPOINT_FORMAT_VERSION: int = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be deserialised safely."""


class SafetyError(ValueError):
    """Raised when untrusted data references a forbidden class."""


#: Builtins that may legitimately appear inside a tabular Q-table pickle.
_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "int",
        "float",
        "bool",
        "str",
        "bytes",
        "bytearray",
        "complex",
        "NoneType",
        "range",
        "slice",
        "object",
        "type",
    }
)

#: Allow-listed container classes keyed by module.
_SAFE_MODULE_MEMBERS: dict[str, frozenset[str]] = {
    "collections": frozenset({"OrderedDict", "defaultdict", "deque"}),
}


def _restricted_loads(obj: Any, globals_: Any = None, locals_: Any = None) -> Any:
    """Refuse to execute code objects coming from an untrusted pickle.

    ``pickle`` can encode raw byte-code that the interpreter would ``exec``.
    Mappings are merged into ``locals_`` instead (the only "execution" any
    legitimate Q-table needs); anything else is rejected outright.
    """
    if isinstance(obj, bytes | bytearray):
        raise SafetyError("refusing to exec a code object from an untrusted pickle")
    if isinstance(obj, dict):
        if locals_ is not None and hasattr(locals_, "update"):
            locals_.update(obj)
            return locals_
        return obj
    if globals_:  # never execute arbitrary payloads with a populated namespace
        raise SafetyError("refusing to exec untrusted payload with a non-empty globals dict")
    return obj


class LegacyPickleUnpickler(pickle.Unpickler):
    """Unpickler that refuses to load any class outside a small allow-list.

    This is the security boundary for legacy ``.pkl`` Q-tables: no I/O, no
    subprocess, no ``eval``/``exec`` and no arbitrary import may be reached
    while restoring a table.
    """

    _SAFE_BUILTINS: frozenset[str] = _SAFE_BUILTINS
    _SAFE_MODULES: dict[str, frozenset[str]] = _SAFE_MODULE_MEMBERS
    _BANNED: frozenset[str] = frozenset(
        {"system", "popen", "spawn", "fork", "eval", "exec", "open", "compile", "__import__"}
    )

    def find_class(self, module: str, name: str) -> Any:
        if name in self._BANNED:
            raise SafetyError(f"refusing to unpickle unsafe class {module}.{name}")
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return super().find_class(module, name)
        if name in self._SAFE_MODULES.get(module, frozenset()):
            return super().find_class(module, name)
        raise SafetyError(f"refusing to unpickle unsafe class {module}.{name}")


#: Backwards-compatible private alias.
_RestrictedUnpickler = LegacyPickleUnpickler


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file in the same directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:  # pragma: no cover - cleanup on failure
        Path(handle.name).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file (streamed, large files safe)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_torch_load(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load a torch checkpoint safely.

    Uses ``weights_only=True`` (torch >= 1.13) so only tensors and primitive
    types are restored — never arbitrary Python objects.  Any deserialisation
    failure (corrupt file, unsafe global, truncated archive) is normalised to
    :class:`CheckpointError` so callers never catch pickle internals.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    log.debug("loading torch checkpoint (weights_only=True): %s", path)
    try:
        payload = torch.load(
            str(path), map_location=map_location, weights_only=_TORCH_WEIGHTS_ONLY
        )
    except Exception as exc:
        raise CheckpointError(
            f"refusing to load unsafe or corrupt checkpoint {path} "
            f"({type(exc).__name__}): {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointError(
            f"checkpoint {path} did not contain a mapping "
            f"(got {type(payload).__name__})"
        )
    return payload


def save_torch_checkpoint(obj: dict, path: str | Path, *, metadata: dict | None = None) -> Path:
    """Persist a checkpoint dict (tensors + primitives only) with provenance.

    The write is atomic and the payload always carries a ``metadata`` mapping
    with the schema version, a UTC timestamp and the caller-supplied metadata,
    so a truncated artefact is detected before training resumes from it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(obj).__name__}")
    payload = dict(obj)
    meta: dict = dict(payload.get("metadata") or {})
    meta.update(metadata or {})
    meta.setdefault("format_version", CHECKPOINT_FORMAT_VERSION)
    meta["saved_at"] = _utc_now_iso()
    payload["metadata"] = meta

    tmp = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, str(tmp))
        os.replace(tmp, path)
    except BaseException:  # pragma: no cover - cleanup on failure
        tmp.unlink(missing_ok=True)
        raise
    log.debug("saved torch checkpoint: %s", path)
    return path


def safe_torch_save(obj: dict, path: str | Path) -> None:
    """Backwards-compatible alias for :func:`save_torch_checkpoint`."""
    save_torch_checkpoint(obj, path)


# -----------------------------------------------------------------------------
# Q-table (JSON based)
# -----------------------------------------------------------------------------

_QTable = dict[tuple[int, ...], list[float]]


def _canonical_q_table_entries(q_table: _QTable) -> list[dict[str, list[int] | list[float]]]:
    """Return a deterministic, JSON-serialisable view of the Q-table."""
    return [
        {"state": [int(x) for x in key], "values": [float(v) for v in value]}
        for key, value in sorted(q_table.items(), key=lambda item: item[0])
    ]


def _q_table_checksum(entries: list[dict]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_q_table(q_table: _QTable, path: str | Path, *, metadata: dict | None = None) -> Path:
    """Persist a Q-table as JSON with provenance and an integrity checksum.

    Keys are integer tuples (state discretisation); the on-disk representation
    uses JSON arrays, which ``json`` round-trips safely and can never execute
    code.  Reserved metadata keys (``format_version``, ``checksum_sha256``,
    ``saved_at``, ``n_states``) are always written, even if the caller passes
    their own ``metadata``.
    """
    path = Path(path)
    entries = _canonical_q_table_entries(q_table)
    meta: dict = {
        "format_version": Q_TABLE_FORMAT_VERSION,
        "saved_at": _utc_now_iso(),
        "n_states": len(entries),
        "checksum_sha256": _q_table_checksum(entries),
    }
    meta.update(metadata or {})
    payload = {"format": Q_TABLE_FORMAT, "metadata": meta, "q_table": entries}

    _atomic_write_text(path, json.dumps(payload, indent=2))
    log.debug("saved Q-table (%d states) to %s", len(entries), path)
    return path


def load_q_table(path: str | Path, *, verify_checksum: bool = True) -> _QTable:
    """Load a Q-table written by :func:`save_q_table`.

    Raises
    ------
    ValueError
        If the file is not valid JSON, is missing required keys, or fails the
        SHA-256 integrity check.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q-table not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not valid JSON: expected an object at the top level")
    if "q_table" not in payload:
        raise ValueError(f"{path} is not a valid Q-table: missing required key 'q_table'")

    entries = payload["q_table"]
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a valid Q-table: 'q_table' must be a list")

    meta = payload.get("metadata") or {}
    if verify_checksum and isinstance(meta, dict) and meta.get("checksum_sha256"):
        expected = meta["checksum_sha256"]
        actual = _q_table_checksum(entries)
        if actual != expected:
            raise ValueError(
                f"{path} failed its integrity check (checksum {actual[:12]} != {expected[:12]})"
            )

    q_table: _QTable = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "state" not in entry or "values" not in entry:
            raise ValueError(
                f"{path} is not a valid Q-table: entry {index} is missing "
                "required key 'state'/'values'"
            )
        q_table[tuple(int(x) for x in entry["state"])] = [float(x) for x in entry["values"]]
    log.debug("loaded Q-table (%d states) from %s", len(q_table), path)
    return q_table


def load_legacy_q_table_pickle(path: str | Path) -> _QTable:
    """Migrate a legacy ``.pkl`` Q-table using the restricted unpickler.

    Accepts both shapes seen in the wild: a raw ``{state: values}`` mapping and
    the ``{"q_table": {state: values}}`` wrapper.  Raises
    :class:`~traffic_rl.io_utils.SafetyError` (a ``ValueError``) if the file
    references any non allow-listed class — that is the security boundary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q-table not found: {path}")
    try:
        with path.open("rb") as handle:
            data = LegacyPickleUnpickler(handle).load()
    except SafetyError:
        raise
    except pickle.UnpicklingError as exc:  # pragma: no cover - defensive
        raise SafetyError(f"refusing to unpickle unsafe class in {path}: {exc}") from exc

    if isinstance(data, dict) and "q_table" in data:
        data = data["q_table"]
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a Q-table: expected a mapping, got {type(data).__name__}")

    q_table: _QTable = {}
    for key, values in data.items():
        key_tuple = (key,) if isinstance(key, int | np.integer) else tuple(int(x) for x in key)
        q_table[key_tuple] = [float(x) for x in np.asarray(values).ravel()]
    log.debug("migrated legacy pickle Q-table (%d states) from %s", len(q_table), path)
    return q_table


def q_table_to_json_compat(q_table: _QTable) -> dict:
    """Convert a Q-table to a JSON-serialisable mapping (for in-memory use)."""
    return {"q_table": [{"state": list(k), "values": list(v)} for k, v in q_table.items()]}


def save_json(obj: Any, path: str | Path) -> Path:
    """Atomically write *obj* as pretty-printed JSON (numpy/Path aware)."""
    path = Path(path)
    _atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default))
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON document, raising :class:`ValueError` on malformed input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# Re-export for convenience.
__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "Q_TABLE_FORMAT",
    "Q_TABLE_FORMAT_VERSION",
    "CheckpointError",
    "LegacyPickleUnpickler",
    "SafetyError",
    "load_json",
    "load_legacy_q_table_pickle",
    "load_q_table",
    "q_table_to_json_compat",
    "safe_torch_load",
    "safe_torch_save",
    "save_json",
    "save_q_table",
    "save_torch_checkpoint",
    "sha256_file",
]
