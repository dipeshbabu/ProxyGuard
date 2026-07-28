from __future__ import annotations

import numpy as np

from scripts.proxyguard.run_proxyguard_mechanism_study import (
    MECHANISM_METHODS,
    build_planning_table,
    holm_rejections,
    holm_release_counts,
    release_collective_simes_counts,
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
    # k=1: 4*0.01 = 0.04 -> reject
    # k=2: 2*0.02 = 0.04 -> reject
    # k=3: 1.333*0.05 = 0.0667 -> reject
    # k=4: 1*0.20 = 0.20 -> not reject
    expected = np.array([[3]])
    got = release_partial_conjunction_counts(release_pvalues, release_alpha=0.07)
    assert got.shape == (1, 1)
    assert got.tolist() == expected.tolist()


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
