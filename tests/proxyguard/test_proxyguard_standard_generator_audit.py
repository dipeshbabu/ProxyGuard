from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.proxyguard.standard_generator_audit import _limits, _select_roles


def test_standard_generator_role_selection_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "Configuration": np.repeat(["a", "b", "c", "d"], 2),
            "MinimumMargin": [0.09, 0.10, 0.045, 0.05, 0.01, 0.02, -0.2, -0.1],
        }
    )

    selected = _select_roles(frame, 0.035)

    assert selected == {
        "high_fidelity": "a",
        "moderate_evidence": "b",
        "degraded": "d",
    }


def test_policy_limits_separate_degradation_from_absolute_ceiling() -> None:
    registry = {
        "claims": {
            "primary_policy": {
                "normalized_degradation_budget": 0.08,
                "absolute_ceilings": {
                    "classification_error": 0.25,
                    "brier": 0.20,
                    "normalized_cost5x": 0.18,
                },
            }
        }
    }

    limits = _limits(np.asarray([0.10, 0.15, 0.04]), registry)

    np.testing.assert_allclose(limits, np.asarray([0.18, 0.20, 0.12]))
