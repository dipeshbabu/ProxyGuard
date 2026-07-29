from __future__ import annotations

import json
from pathlib import Path

from scripts.proxyguard.run_proxyguard_conditional_shared_target import (
    run_study,
)


def test_conditional_shared_target_study_runs_small_grid() -> None:
    registry = json.loads(
        Path(
            "registries/proxyguard_conditional_shared_target_pilot.json"
        ).read_text(encoding="utf-8")
    )
    settings = registry["conditional_shared_target_study"]
    settings["repetitions"] = 2
    settings["release_counts"] = [20]
    settings["target_sizes"] = [100]
    settings["reliabilities"] = [0.8, 0.95]

    result = run_study(registry)

    assert result.shape[0] == 6
    assert set(result["Method"]) == {
        "Named-release Holm",
        "Conditional shared-target",
        "Oracle release labels",
    }
    assert result["ValidationRate"].between(0.0, 1.0).all()
