"""Shared helpers for torch device selection and tensor utilities."""

from __future__ import annotations

from typing import Any

try:
    import torch
    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False


def default_device():
    """Return the best available device, preferring CUDA then CPU."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed; neural agents are unavailable.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(obj: Any, dtype=None, device=None) -> "torch.Tensor":  # type: ignore[name-defined]
    """Convert a numpy array or python sequence to a torch tensor if torch is available."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed; neural agents are unavailable.")
    import torch as _torch
    return _torch.as_tensor(obj, dtype=dtype, device=device)


def detach_numpy(tensor: "torch.Tensor") -> Any:  # type: ignore[name-defined]
    """Detach a torch tensor and return a numpy array if torch is available."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed; neural agents are unavailable.")
    return tensor.detach().cpu().numpy()

