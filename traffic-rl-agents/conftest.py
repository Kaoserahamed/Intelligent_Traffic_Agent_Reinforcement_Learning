"""Repository-level pytest configuration (the rootdir *and* top-level conftest).

Two jobs:

1. Make ``src/`` importable without ``pip install -e .`` (so ``pytest`` works
   from a fresh clone and on Windows without activation gymnastics).
2. Register the structured test-report plugin used by CI
   (``--structured-report reports/test-report.json``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest_plugins = ["traffic_rl.observability.test_reporter"]
