from __future__ import annotations

import numpy as np
import pytest

from proxyguard.core import empirical_bernstein_badness_pvalue
from scripts.proxyguard.run_proxyguard_direct_multirequirement import (
    DIRECT,
    HYBRID,
    NAMED,
    ORACLE,
    _draw_summaries,
    _empirical_bernstein_pvalues_from_summaries,
    run_confirmation,
    run_pilot,
)


SETTINGS = {
    "seed": 17,
    "repetitions": 2,
    "mechanisms": 2,
    "minimum_reliability": 0.8,
    "total_alpha": 0.05,
    "lower_bounds": [-1.0, 0.0, 0.0],
    "upper_bounds": [1.0, 1.0, 2.0],
    "tolerances": [0.1, 0.35, 0.8],
    "valid_margin_low": 0.035,
    "valid_margin_high": 0.09,
    "margin_jitter": 0.008,
    "invalid_margin_low": 0.001,
    "invalid_margin_high": 0.05,
    "shared_noise_scale": 0.08,
    "requirement_noise_scale": 0.08,
    "interaction_noise_scale": 0.04,
}


def test_summary_pvalues_match_scalar_implementation() -> None:
    rng = np.random.default_rng(31)
    values = rng.uniform(-0.6, 0.4, size=(7, 80))
    means = values.mean(axis=1)[:, None]
    variances = values.var(axis=1, ddof=1)[:, None]
    vectorized = _empirical_bernstein_pvalues_from_summaries(
        means,
        variances,
        target_records=80,
        tolerances=np.array([0.5]),
        lower_bounds=np.array([-1.0]),
        upper_bounds=np.array([1.0]),
    )[:, 0]
    scalar = np.asarray(
        [
            empirical_bernstein_badness_pvalue(row, 0.5, -1.0, 1.0)
            for row in values
        ]
    )

    assert vectorized == pytest.approx(scalar, abs=1e-12)


def test_correlated_continuous_draws_respect_registered_ranges() -> None:
    means, variances, valid = _draw_summaries(
        rng=np.random.default_rng(73),
        settings=SETTINGS,
        releases=100,
        target_records=250,
        reliability=0.9,
    )

    assert means.shape == (100, 3)
    assert variances.shape == (100, 3)
    assert valid.shape == (100,)
    assert np.all(means > np.asarray(SETTINGS["lower_bounds"]))
    assert np.all(means < np.asarray(SETTINGS["upper_bounds"]))
    assert np.all(variances > 0.0)


def test_smoothed_bernoulli_draws_are_bounded_and_high_variance() -> None:
    settings = {
        **SETTINGS,
        "noise_model": "smoothed_bernoulli",
        "continuous_fraction": 0.02,
    }
    means, variances, valid = _draw_summaries(
        rng=np.random.default_rng(79),
        settings=settings,
        releases=50,
        target_records=250,
        reliability=0.9,
    )
    lower = np.asarray(settings["lower_bounds"])
    upper = np.asarray(settings["upper_bounds"])

    assert means.shape == (50, 3)
    assert valid.shape == (50,)
    assert np.all(means > lower)
    assert np.all(means < upper)
    assert np.all(variances > 0.0)
    normalized_variances = variances / np.square(upper - lower)
    assert float(normalized_variances.mean()) > 0.05


def test_pilot_and_confirmation_smoke() -> None:
    pilot_settings = {
        **SETTINGS,
        "planning_release_count": 20,
        "planning_target_records": 100,
        "planning_reliabilities": [0.8, 0.9],
        "named_release_error_fractions": [0.5],
        "direct_target_error_fractions": [0.75],
        "direct_normalized_slacks": [0.05],
    }
    pilot = run_pilot(
        {
            "direct_multirequirement_study": pilot_settings,
        }
    )
    assert set(pilot["Method"]) == {NAMED, DIRECT}

    confirmation_settings = {
        **SETTINGS,
        "release_counts": [20],
        "target_sizes": [100],
        "reliabilities": [0.8, 0.9],
        "selected_named_release_error_fraction": 0.5,
        "selected_direct_target_error_fraction": 0.75,
        "selected_direct_normalized_slack": 0.05,
    }
    confirmation = run_confirmation(
        {
            "direct_multirequirement_study": confirmation_settings,
        }
    )
    assert set(confirmation["Method"]) == {NAMED, DIRECT, HYBRID, ORACLE}
