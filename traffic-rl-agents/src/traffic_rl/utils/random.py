"""Reproducible seeding utilities for Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np

try:  # torch is an optional, heavy dependency used only by neural agents.
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _HAS_TORCH = False


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    Also configures the Python hash randomisation seed *for the current
    process* via the ``PYTHONHASHSEED`` environment variable.  Note that hash
    randomisation is only read at interpreter start-up, so this is best-effort.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def numpy_rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded NumPy :class:`Generator` (preferred over global state)."""
    return np.random.default_rng(seed)
