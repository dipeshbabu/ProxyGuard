from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.proxyguard.run_proxyguard_text_direct_audit import _losses, _select_roles


def test_text_audit_losses_are_bounded_and_directional() -> None:
    labels = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])

    losses = _losses(labels, probabilities)

    assert losses.shape == (4, 3)
    assert np.all((0.0 <= losses) & (losses <= 1.0))
    assert np.allclose(losses[:, 0], [0.0, 1.0, 1.0, 0.0])
    assert losses[1, 2] == 1.0
    assert losses[2, 2] == 1.0 / 3.0


def test_text_role_selection_is_deterministic_and_distinct() -> None:
    frame = pd.DataFrame(
        {
            "Configuration": ["high"] * 3 + ["middle"] * 3 + ["bad"] * 3,
            "MinimumScoreMargin": [0.02, 0.03, 0.04, 0.001, 0.003, 0.005, -0.2, -0.1, -0.05],
        }
    )

    roles = _select_roles(frame, target_margin=0.003)

    assert roles == {
        "high_signal": "high",
        "moderate_evidence": "middle",
        "degraded": "bad",
    }
