from __future__ import annotations

import numpy as np

from scripts.proxyguard.run_proxyguard_false_pass_diagnostic import (
    DIRECT,
    ORACLE,
    SCORE_ONLY,
    _draw_shared_target_means,
    run_confirmation,
    run_pilot,
)


SETTINGS = {
    "seed": 19,
    "repetitions": 3,
    "minimum_reliability": 0.8,
    "total_alpha": 0.05,
    "target_error_fraction": 0.9,
    "lower_bound": 0.0,
    "upper_bound": 1.0,
    "tolerance": 0.5,
    "valid_risk_low": 0.30,
    "valid_risk_high": 0.40,
    "invalid_risk_low": 0.5005,
    "invalid_risk_high": 0.54,
}


def test_shared_target_draws_preserve_shapes_and_bounds() -> None:
    means, valid = _draw_shared_target_means(
        rng=np.random.default_rng(41),
        releases=50,
        target_records=100,
        reliability=0.8,
        valid_risk_low=0.30,
        valid_risk_high=0.40,
        invalid_risk_low=0.5005,
        invalid_risk_high=0.54,
    )

    assert means.shape == (50, 1)
    assert valid.shape == (50,)
    assert np.all((means >= 0.0) & (means <= 1.0))


def test_pilot_and_confirmation_report_all_diagnostic_methods() -> None:
    pilot = run_pilot(
        {
            "false_pass_diagnostic": {
                **SETTINGS,
                "release_counts": [50],
                "target_sizes": [100],
                "normalized_slacks": [0.02],
            }
        }
    )
    assert set(pilot["Method"]) == {SCORE_ONLY, DIRECT, ORACLE}

    confirmation = run_confirmation(
        {
            "false_pass_diagnostic": {
                **SETTINGS,
                "selected_release_count": 50,
                "selected_target_records": 100,
                "selected_normalized_slack": 0.02,
            }
        }
    )
    assert set(confirmation["Method"]) == {SCORE_ONLY, DIRECT, ORACLE}
    score_only = confirmation.loc[
        confirmation["Method"] == SCORE_ONLY,
        "MeanLowerBound",
    ].item()
    corrected = confirmation.loc[
        confirmation["Method"] == DIRECT,
        "MeanLowerBound",
    ].item()
    assert score_only >= corrected


def test_confirmation_can_sweep_true_reliability() -> None:
    confirmation = run_confirmation(
        {
            "false_pass_diagnostic": {
                **SETTINGS,
                "selected_release_count": 50,
                "selected_target_records": 100,
                "selected_normalized_slack": 0.02,
                "true_reliabilities": [0.8, 0.95],
            }
        }
    )

    assert len(confirmation) == 6
    assert set(confirmation["TrueReliability"]) == {0.8, 0.95}
