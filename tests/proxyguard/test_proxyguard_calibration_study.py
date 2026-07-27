import numpy as np

from scripts.proxyguard.run_proxyguard_calibration_study import (
    candidate_decisions,
    holm_rejections,
    invalid_probabilities,
)


def test_holm_rejections_stop_after_the_first_failed_step() -> None:
    decisions = holm_rejections(
        np.array(
            [
                [0.001, 0.020, 0.030],
                [0.030, 0.040, 0.050],
            ]
        ),
        alpha=0.05,
    )

    assert decisions[0].tolist() == [True, True, True]
    assert decisions[1].tolist() == [False, False, False]


def test_invalid_design_has_one_violated_requirement_per_candidate() -> None:
    probabilities = invalid_probabilities(
        candidates=5,
        requirements=3,
        tolerance=0.1,
        invalid_gap=0.01,
        safe_risk=0.02,
    )

    assert (probabilities > 0.1).sum(axis=1).tolist() == [1, 1, 1, 1, 1]


def test_candidate_decisions_require_all_requirements() -> None:
    counts = np.array([[[0, 0], [0, 20]]])
    decisions = candidate_decisions(
        counts=counts,
        audit_size=20,
        tolerance=0.2,
        alpha=0.05,
    )

    assert decisions["Point threshold"].tolist() == [[True, False]]
    assert not decisions["Per-proxy IUT"][0, 1]
