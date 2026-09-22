"""Type aliases shared by the observability package."""

from __future__ import annotations

from typing import Any

#: A flat mapping of context key -> value (run_id, agent, episode, ...).
ContextDict = dict[str, Any]

__all__ = ["ContextDict"]
