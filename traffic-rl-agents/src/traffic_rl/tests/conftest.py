"""Pytest fixtures shared across tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the package root is on sys.path for tests run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with a minimal SUMO configuration.

    The directory contains a valid ``.sumocfg``, an empty ``.net.xml`` and an
    empty ``.rou.xml`` so that :class:`~traffic_rl.sumo.environment.TrafficEnvironment`
    can be constructed without a real SUMO installation.
    """
    sim = tmp_path / "sim"
    sim.mkdir()
    simo_cfg = sim / "intersection.sumocfg"
    simo_cfg.write_text(
        "<configuration><input><net-file value='intersection.net.xml'/>"
        "<route-files value='intersection.rou.xml'/></input>"
        "<time><begin value='0'/><end value='100'/></time></configuration>"
    )
    (sim / "intersection.net.xml").write_text("<net></net>")
    (sim / "intersection.rou.xml").write_text("<routes></routes>")
    return sim


@pytest.fixture
def config_path(sim_dir: Path) -> Path:
    return sim_dir / "intersection.sumocfg"
