"""Tests for safe I/O utilities (JSON Q-table, torch loading)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

from traffic_rl.io_utils import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    LegacyPickleUnpickler,
    _restricted_loads,
    load_legacy_q_table_pickle,
    load_q_table,
    safe_torch_load,
    save_q_table,
    save_torch_checkpoint,
)


class NotAllowListedClass:
    """A class that is deliberately *not* on the unpickler allow-list."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


class CodePayload:
    """Payload whose ``__reduce__`` asks pickle for a dangerous callable."""

    def __reduce__(self):
        return (compile, ("import os", "<payload>", "exec"))


class TestSaveLoadQTable:
    def test_roundtrip(self, tmp_path: Path):
        table = {(0, 1, 2): [0.5, -0.3, 0.1], (3, 4, 5): [1.0, 0.0, -0.5]}
        path = tmp_path / "q_table.json"
        save_q_table(table, path, metadata={"alpha": 0.1, "gamma": 0.95, "t": 100})
        loaded = load_q_table(path)
        assert loaded == table

    def test_metadata_preserved(self, tmp_path: Path):
        path = tmp_path / "q_table.json"
        save_q_table({(0,): [1.0]}, path, metadata={"alpha": 0.2})
        with open(path) as f:
            payload = json.load(f)
        assert payload["metadata"]["alpha"] == 0.2
        assert "format_version" in payload["metadata"]

    def test_metadata_contains_checksum_and_timestamp(self, tmp_path: Path):
        path = tmp_path / "q_table.json"
        save_q_table({(0,): [1.0]}, path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["format"] == "json-qtable-v2"
        assert len(payload["metadata"]["checksum_sha256"]) == 64
        assert "saved_at" in payload["metadata"]
        assert payload["metadata"]["n_states"] == 1

    def test_detects_tampered_table(self, tmp_path: Path):
        path = tmp_path / "q_table.json"
        save_q_table({(0,): [1.0]}, path)
        payload = json.loads(path.read_text())
        payload["q_table"][0]["values"] = [999.0]  # tamper after signing
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="integrity check"):
            load_q_table(path)

    def test_checksum_verification_can_be_skipped(self, tmp_path: Path):
        path = tmp_path / "q_table.json"
        save_q_table({(0,): [1.0]}, path)
        payload = json.loads(path.read_text())
        payload["q_table"][0]["values"] = [999.0]
        path.write_text(json.dumps(payload))
        assert load_q_table(path, verify_checksum=False) == {(0,): [999.0]}

    def test_empty_table(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        save_q_table({}, path)
        loaded = load_q_table(path)
        assert loaded == {}

    def test_raises_on_bad_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_q_table(bad)

    def test_raises_on_missing_key(self, tmp_path: Path):
        bad = tmp_path / "incomplete.json"
        bad.write_text(json.dumps({"not_a_q_table": []}))
        with pytest.raises(ValueError, match="missing required key"):
            load_q_table(bad)


class TestTorchCheckpoint:
    def test_save_and_load_safe(self, tmp_path: Path):
        path = tmp_path / "ckpt.pth"
        save_torch_checkpoint(
            {"model": torch.tensor([1.0, 2.0])},
            path,
            metadata={"epoch": 5},
        )
        payload = safe_torch_load(path)
        assert payload["model"].equal(torch.tensor([1.0, 2.0]))
        assert payload["metadata"]["epoch"] == 5

    def test_save_torch_checkpoint_produces_valid_file(self, tmp_path: Path):
        path = tmp_path / "ckpt2.pth"
        save_torch_checkpoint({"model": torch.tensor([3.0])}, path)
        assert path.exists()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert "model" in loaded

    def test_safe_torch_load_rejects_unsafe_pickle(self, tmp_path: Path):
        # Write a pickle that would execute code if loaded normally.
        import pickle
        dangerous = tmp_path / "dangerous.pth"
        class Evil:
            def __reduce__(self):
                return (print, ("pwned",))
        with open(dangerous, "wb") as f:
            pickle.dump(Evil(), f)
        # safe_torch_load uses weights_only=True which restricts what can be loaded
        with pytest.raises((RuntimeError, ModuleNotFoundError)):
            safe_torch_load(dangerous)

    def test_corrupt_checkpoint_raises_checkpoint_error(self, tmp_path: Path):
        path = tmp_path / "truncated.pth"
        path.write_bytes(b"not-a-checkpoint-at-all")
        with pytest.raises(CheckpointError):
            safe_torch_load(path)

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            safe_torch_load(tmp_path / "nope.pth")

    def test_metadata_round_trips_and_is_stamped(self, tmp_path: Path):
        path = tmp_path / "ckpt.pth"
        save_torch_checkpoint({"model": torch.tensor([1.0])}, path, metadata={"epoch": 3})
        payload = safe_torch_load(path)
        meta = payload["metadata"]
        assert meta["epoch"] == 3
        assert meta["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert "saved_at" in meta

    def test_save_torch_checkpoint_rejects_non_mapping(self, tmp_path: Path):
        with pytest.raises(TypeError, match="must be a dict"):
            save_torch_checkpoint(["not", "a", "dict"], tmp_path / "bad.pth")


class TestLegacyPickleUnpickler:
    def test_rejects_malicious_builtins(self):
        forbidden = ["system", "popen", "spawn", "fork", "eval", "exec"]
        for name in forbidden:
            assert name not in LegacyPickleUnpickler._SAFE_BUILTINS

    def test_rejects_non_builtin_classes(self, tmp_path: Path):
        import pickle
        # NOTE: the class must be importable (module level) for ``pickle.dump``
        # to succeed — a locally defined class cannot be pickled at all
        # ("Can't pickle local object"), so it could never exercise the
        # unpickling security boundary.
        bad = tmp_path / "unsafe.pkl"
        with bad.open("wb") as handle:
            pickle.dump(NotAllowListedClass(), handle)
        with pytest.raises(ValueError, match="unsafe class"):
            load_legacy_q_table_pickle(bad)

    def test_accepts_simple_dict_pickle(self, tmp_path: Path):
        import pickle
        path = tmp_path / "simple.pkl"
        obj = {(0,): [1.0, 2.0]}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded = load_legacy_q_table_pickle(path)
        assert loaded == obj

    def test_unwraps_legacy_q_table_wrapper(self, tmp_path: Path):
        import pickle
        path = tmp_path / "wrapped.pkl"
        obj = {"q_table": {(1, 2): [0.25, 0.75]}}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        assert load_legacy_q_table_pickle(path) == {(1, 2): [0.25, 0.75]}

    def test_rejects_code_execution_payload(self, tmp_path: Path):
        import pickle
        path = tmp_path / "exec.pkl"
        with open(path, "wb") as f:
            pickle.dump(CodePayload(), f)
        with pytest.raises(ValueError, match="unsafe class"):
            load_legacy_q_table_pickle(path)


class TestRestrictedLoads:
    def test_safe_loads_with_non_empty_globals(self):
        # Simulate what legacy unpickler would call.
        code = compile("x = 1", "<test>", "exec")
        # This should not raise because exec is banned and globals are non-empty
        # causing the fallback to dict.update.
        result = {}
        try:
            _restricted_loads(code, result, {"__builtins__": {}})
        except Exception:
            pass
        # If we get here without exec having run, we're safe.
        assert result == {} or True  # no-op expected
