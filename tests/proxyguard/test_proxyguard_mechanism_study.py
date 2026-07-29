from __future__ import annotations

import numpy as np

from scripts.proxyguard.run_proxyguard_collective_extension import (
    simulate_collective_setting,
)
from scripts.proxyguard.run_proxyguard_mechanism_study import (
    MECHANISM_METHODS,
    build_planning_table,
    holm_rejections,
    holm_release_counts,
    release_collective_simes_counts,
    release_collective_fisher_counts,
    release_partial_conjunction_counts,
    release_validation_counts,
    run_adaptive_study,
    run_mechanism_study,
)


def _registry(repetitions: int = 50) -> dict:
    return {
        "mechanism_reliability": {
            "total_alpha": 0.05,
            "release_error_share": 0.5,
            "primary_minimum_reliability": 0.8,
            "sensitivity_minimum_reliability": 0.9,
            "requirements": 2,
            "mechanisms": 2,
            "tolerance": 0.1,
            "safe_release_risk": 0.02,
            "bad_release_risk": 0.12,
            "boundary_reliability": 0.8,
            "valid_reliability": 0.98,
            "release_counts": [5, 10],
            "audit_sizes": [50],
            "repetitions": repetitions,
            "seed": 7,
        },
        "adaptive_search": {
            "total_alpha": 0.05,
            "requirements": 2,
            "tolerance": 0.1,
            "boundary_risk": 0.1,
            "nonbinding_risk": 0.02,
            "valid_risk": 0.02,
            "audit_size": 50,
            "rounds": [1, 3],
            "valid_arrival_rounds": [1, 3],
            "repetitions": repetitions,
            "seed": 8,
        },
    }


def test_vectorized_holm_stops_after_the_first_failure() -> None:
    rejected = holm_rejections(
        np.array(
            [
                [0.001, 0.02, 0.5],
                [0.02, 0.03, 0.04],
            ]
        ),
        alpha=0.05,
    )

    assert rejected[0].tolist() == [True, True, False]
    assert rejected[1].tolist() == [False, False, False]


def test_release_simulation_returns_one_count_per_mechanism() -> None:
    counts = release_validation_counts(
        rng=np.random.default_rng(2),
        repetitions=10,
        mechanisms=3,
        releases=4,
        requirements=2,
        audit_size=100,
        tolerance=0.1,
        safe_risk=0.02,
        bad_release_risk=0.12,
        reliability=0.8,
        release_alpha=0.025,
    )

    assert set(counts) == set(MECHANISM_METHODS)
    assert all(value.shape == (10, 3) for value in counts.values())
    assert all(np.logical_and(value >= 0, value <= 4).all() for value in counts.values())


def test_registered_studies_emit_every_method_and_setting() -> None:
    registry = _registry()
    mechanism = run_mechanism_study(registry)
    adaptive = run_adaptive_study(registry)

    assert set(mechanism["Method"]) == set(MECHANISM_METHODS)
    assert set(mechanism["Releases"]) == {5, 10}
    assert set(adaptive["Analysis"]) == {
        "False validation",
        "Valid-candidate power",
    }
    assert set(adaptive["Method"]) == {
        "Fixed alpha each round",
        "Quadratic alpha spending",
    }


def test_release_planning_accounts_for_reliability_and_multiplicity() -> None:
    planning = build_planning_table(_registry()).set_index(
        ["MinimumReliability", "Mechanisms"]
    )

    assert (
        planning.loc[(0.9, 2), "AllRecognizedGoodReleasesNeeded"]
        > planning.loc[(0.8, 2), "AllRecognizedGoodReleasesNeeded"]
    )
    assert (
        planning.loc[(0.8, 9), "AllRecognizedGoodReleasesNeeded"]
        > planning.loc[(0.8, 1), "AllRecognizedGoodReleasesNeeded"]
    )


def test_partial_conjunction_respects_partial_conjunction_formula() -> None:
    release_pvalues = np.array(
        [
            [
                [0.01, 0.02, 0.05, 0.20],  # sorted: 0.01,0.02,0.05,0.20
            ],
        ],
        dtype=float,
    )
    # For R=4:
    # k=1: min(4*0.01, 2*0.02, 4/3*0.05, 4/4*0.2) = 0.04 -> reject
    # k=2: min(3*0.02, 3/2*0.05, 3/3*0.2) = 0.06 -> reject
    # k=3: min(2*0.05, 2/2*0.2) = 0.10 -> not reject
    # so the largest k is 2.
    got = release_partial_conjunction_counts(release_pvalues, release_alpha=0.07)
    assert got.shape == (1, 1)
    assert got.tolist() == [[2]]


def test_collective_simes_count_is_not_looser_than_holm() -> None:
    release_pvalues = np.array(
        [
            [
                [0.001, 0.02, 0.03, 0.25],
                [0.10, 0.12, 0.14, 0.16],
                [0.001, 0.40, 0.50, 0.90],
            ]
        ],
        dtype=float,
    )
    alpha = 0.05
    partial = release_collective_simes_counts(release_pvalues, release_alpha=alpha)
    holm = holm_release_counts(release_pvalues, release_alpha=alpha)
    assert (partial >= holm).all()


def test_collective_fisher_count_shape_and_range() -> None:
    release_pvalues = np.array(
        [
            [
                [0.001, 0.005, 0.02, 0.10],
                [0.01, 0.02, 0.50, 0.90],
            ]
        ],
        dtype=float,
    )
    counts = release_collective_fisher_counts(
        release_pvalues,
        release_alpha=0.05,
    )
    assert counts.shape == (1, 2)
    assert ((counts >= 0) & (counts <= 4)).all()


def test_partial_conjunction_collective_score_is_more_conservative_than_invalid_formula() -> None:
    """Regression: the old R/k * p_(k) score was anti-conservative.

    The synthetic construction has one effectively perfect candidate (p=0) and two
    invalid candidates with independent Uniform p-values. The old formula rejects
    at k=2 with probability > 0.05 in this setting.
    """

    rng = np.random.default_rng(11)
    uniforms = rng.random((200_000, 2))
    sorted_uniforms = np.sort(uniforms, axis=1)
    release_pvalues = np.concatenate(
        [np.zeros((200_000, 1)), uniforms],
        axis=1,
    )
    release_pvalues = release_pvalues[:, None, :]
    counts = release_partial_conjunction_counts(
        release_pvalues,
        release_alpha=0.05,
    )

    # At least two valid releases inferred (k >= 2) under the fixed threshold.
    observed = (counts[:, 0] >= 2).mean()
    assert observed < 0.05

    old_formula_reject = (1.5 * sorted_uniforms[:, 0] <= 0.05).mean()
    assert old_formula_reject > 0.05


def test_collective_extension_records_count_gain_and_global_error() -> None:
    result = simulate_collective_setting(
        rng=np.random.default_rng(19),
        repetitions=50,
        mechanisms=2,
        releases=20,
        requirements=2,
        audit_size=250,
        tolerance=0.1,
        safe_risk=0.06,
        bad_release_risk=0.12,
        minimum_reliability=0.8,
        valid_reliability=0.95,
        total_alpha=0.05,
        release_error_share=0.5,
    ).set_index("Method")

    assert set(result.index) == {
        "Two-level ProxyGuard",
        "Collective partial-conjunction release evidence",
        "Fisher-tail partial-conjunction benchmark",
    }
    assert (
        result.loc[
            "Collective partial-conjunction release evidence",
            "MeanCollectiveCountGain",
        ]
        >= 0.0
    )
