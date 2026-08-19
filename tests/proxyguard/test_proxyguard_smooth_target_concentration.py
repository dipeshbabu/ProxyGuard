from __future__ import annotations

import json
from pathlib import Path

from scripts.proxyguard.run_proxyguard_smooth_target_concentration import (
    SMOOTH_CONCENTRATION,
    run_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def test_smooth_target_driver_returns_all_methods() -> None:
    registry = json.loads(
        (ROOT / "registries/proxyguard_smooth_target_concentration_pilot.json").read_text(encoding="utf-8")
    )
    settings = registry["smooth_target_concentration"]
    settings["repetitions"] = 3
    settings["release_counts"] = [40]
    settings["target_sizes"] = [100]
    settings["ramp_widths"] = [0.6]
    settings["true_reliabilities"] = [0.8]

    frame = run_registry(registry)

    assert len(frame) == 4
    assert SMOOTH_CONCENTRATION in set(frame["Method"])
    assert set(frame["ValidationTrials"]) == {3}
