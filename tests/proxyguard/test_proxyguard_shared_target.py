from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BPoly
from scipy.stats import binom

from proxyguard.shared_target import (
    bernstein_restriction_matrix,
    bernstein_soft_step_overshoot,
    bernstein_step_minorant,
    bounded_kl_lower_bound,
    hybrid_reliability_lower_bound,
    plan_conditional_shared_target,
    project_cost_normalized_plan,
    recommend_cost_normalized_audit,
    shared_target_block_witness_lower_bound,
    shared_target_conditional_mean_lower_bound,
    shared_target_conditional_witness_lower_bound,
    shared_target_polynomial_lower_bound,
    shared_target_reliability_lower_bound,
    shared_target_tensor_polynomial_lower_bound,
    shared_target_witness_lower_bound,
    stratified_release_evidence,
    stratified_shared_target_conditional_witness_lower_bound,
    tensor_bernstein_design_minorant,
    tensor_bernstein_validity_minorant,
)


def test_block_witness_uses_two_axis_mcdiarmid_radius() -> None:
    losses = np.zeros((200, 5_000, 1), dtype=float)
    result = shared_target_block_witness_lower_bound(
        losses,
        tolerances=[0.5],
        slacks=[0.2],
        block_size=50,
        error_rate=0.05,
        block_seed=19,
    )

    expected_radius = np.sqrt(
        0.5 * (1.0 / 200 + 1.0 / 100) * np.log(1.0 / 0.05)
    )
    assert result.target_blocks == 100
    assert result.witness_mean == pytest.approx(1.0)
    assert result.concentration_radius == pytest.approx(expected_radius)
    assert result.witness_mean_lower_bound == pytest.approx(1.0 - expected_radius)
    assert result.reliability_lower_bound > 0.8


def test_block_witness_is_vacuous_when_invalid_ceiling_is_one() -> None:
    losses = np.zeros((20, 20, 1), dtype=float)
    result = shared_target_block_witness_lower_bound(
        losses,
        tolerances=[0.5],
        slacks=[1e-12],
        block_size=1,
        error_rate=0.05,
    )

    assert result.invalid_release_witness_ceiling == pytest.approx(1.0)
    assert result.reliability_lower_bound == 0.0


def test_cost_normalized_planner_selects_feasible_design() -> None:
    plan = recommend_cost_normalized_audit(
        total_budget=6_000.0,
        target_record_cost=5.0,
        release_cost=1.0,
        candidate_target_records=[500, 1_000],
        candidate_releases=[100, 500],
        tolerances=[0.3],
        candidate_slacks=[0.04, 0.06],
        expected_direct_score_probability=0.97,
        expected_named_recognition_probability=0.60,
        target_error_fractions=[0.8],
        named_release_error_shares=[0.9],
    )

    assert plan.total_cost <= 6_000.0
    assert plan.mode == "direct"
    assert plan.target_records == 1_000
    assert plan.releases == 500
    assert plan.projected_reliability_lower_bound > 0.8


def test_cost_plan_projection_detects_pilot_misspecification() -> None:
    plan = recommend_cost_normalized_audit(
        total_budget=6_000.0,
        target_record_cost=5.0,
        release_cost=1.0,
        candidate_target_records=[500, 1_000],
        candidate_releases=[100, 500],
        tolerances=[0.3],
        candidate_slacks=[0.04, 0.06],
        expected_direct_score_probability=0.97,
        expected_named_recognition_probability=0.60,
        target_error_fractions=[0.8],
        named_release_error_shares=[0.9],
    )
    realized = project_cost_normalized_plan(
        plan,
        tolerances=[0.3],
        direct_score_probability=0.87,
        named_recognition_probability=0.60,
    )

    assert plan.mode == "direct"
    assert realized < plan.projected_reliability_lower_bound
    assert 0.0 <= realized <= 1.0


def test_equal_weight_strata_retain_bernoulli_kl_ceiling() -> None:
    first = np.zeros((100, 400, 1), dtype=float)
    second = np.zeros((100, 400, 1), dtype=float)
    result = stratified_shared_target_conditional_witness_lower_bound(
        [first, second],
        stratum_weights=[0.5, 0.5],
        tolerances=[0.4],
        slacks=[0.1],
        error_rate=0.05,
        target_error_fraction=0.8,
    )
    expected = np.exp(-800 * (0.3 * np.log(0.3 / 0.4) + 0.7 * np.log(0.7 / 0.6)))

    assert result.target_records == 800
    assert result.invalid_release_score_ceiling == pytest.approx(expected)
    assert result.reliability_lower_bound > 0.8


def test_unequal_stratum_weights_use_weighted_hoeffding() -> None:
    first = np.full((4, 100, 1), 0.1, dtype=float)
    second = np.full((4, 50, 1), 0.3, dtype=float)
    evidence = stratified_release_evidence(
        [first, second],
        stratum_weights=[0.8, 0.2],
        tolerances=[0.25],
    )
    weighted_mean = 0.8 * 0.1 + 0.2 * 0.3
    variance_proxy = 0.8**2 / 100 + 0.2**2 / 50
    expected_pvalue = np.exp(-2.0 * (0.25 - weighted_mean) ** 2 / variance_proxy)

    assert evidence.concentration_method == "hoeffding"
    assert evidence.weighted_means[:, 0] == pytest.approx(weighted_mean)
    assert evidence.validation_pvalues[:, 0] == pytest.approx(expected_pvalue)
    assert evidence.violation_pvalues[:, 0] == pytest.approx(1.0)


def test_bernstein_restriction_matches_original_polynomial() -> None:
    coefficients = np.array([-0.4, 0.2, 0.8, 0.3, -0.1])
    restricted = bernstein_restriction_matrix(4, 0.2, 0.7) @ coefficients
    original = BPoly(coefficients[:, None], [0.0, 1.0])
    interval = BPoly(restricted[:, None], [0.0, 1.0])
    grid = np.linspace(0.0, 1.0, 1001)

    assert np.allclose(
        np.asarray(interval(grid)).reshape(-1),
        np.asarray(original(0.2 + 0.5 * grid)).reshape(-1),
        atol=1e-12,
    )


def test_bernstein_step_minorant_is_globally_valid() -> None:
    coefficients, valid_floor = bernstein_step_minorant(
        threshold=0.2,
        margin=0.12,
        degree=16,
        coefficient_floor=1.0,
    )
    polynomial = BPoly(coefficients[:, None], [0.0, 1.0])
    grid = np.linspace(0.0, 1.0, 200_001)
    values = np.asarray(polynomial(grid)).reshape(-1)
    indicator = (grid < 0.2).astype(float)

    assert np.all(values <= indicator + 1e-10)
    assert values[grid <= 0.08].min() >= valid_floor - 1e-10
    assert valid_floor > 0.0


def test_shared_target_polynomial_certificate_uses_one_target() -> None:
    losses = np.zeros((50, 3000, 3), dtype=float)

    result = shared_target_polynomial_lower_bound(
        losses,
        tolerances=0.2,
        margins=0.12,
        degree=16,
        coefficient_floor=1.0,
        error_rate=0.05,
        block_seed=31,
    )

    assert result.target_blocks == 187
    assert result.target_records_used == 2992
    assert result.effective_sample_size == 50
    assert min(result.valid_region_floors) > 0.0
    assert result.reliability_lower_bound > 0.0


def test_tensor_bernstein_minorant_is_valid_on_joint_domain() -> None:
    coefficients, valid_floor = tensor_bernstein_validity_minorant(
        thresholds=[0.3, 0.4],
        margins=[0.15, 0.2],
        degree=5,
        coefficient_floor=0.25,
    )
    grid = np.linspace(0.0, 1.0, 101)
    basis = np.asarray(
        [
            binom.pmf(np.arange(6), 5, value)
            for value in grid
        ]
    )
    values = np.einsum(
        "ai,bj,ij->ab",
        basis,
        basis,
        coefficients,
    )
    indicator = (
        (grid[:, None] < 0.3)
        & (grid[None, :] < 0.4)
    ).astype(float)

    assert np.all(values <= indicator + 1e-9)
    valid_box = values[
        grid <= 0.15,
        :,
    ][:, grid <= 0.2]
    assert valid_box.min() >= valid_floor - 1e-9
    assert valid_floor > 0.0


def test_shared_target_tensor_polynomial_can_clear_reliability_target() -> None:
    losses = np.zeros((50, 3000, 3), dtype=float)

    result = shared_target_tensor_polynomial_lower_bound(
        losses,
        tolerances=0.2,
        margins=0.15,
        degree=8,
        coefficient_floor=0.25,
        error_rate=0.05,
        block_seed=41,
    )

    assert result.target_blocks == 125
    assert result.target_records_used == 3000
    assert result.valid_region_floor > 0.0
    assert result.reliability_lower_bound > 0.8


def test_design_optimized_tensor_minorant_remains_globally_valid() -> None:
    design_points = np.array(
        [
            [0.02, 0.03],
            [0.04, 0.01],
            [0.03, 0.04],
        ]
    )
    coefficients = tensor_bernstein_design_minorant(
        thresholds=[0.2, 0.2],
        degree=6,
        design_points=design_points,
        coefficient_floor=0.25,
    )
    grid = np.linspace(0.0, 1.0, 101)
    basis = np.asarray(
        [
            binom.pmf(np.arange(7), 6, value)
            for value in grid
        ]
    )
    values = np.einsum(
        "ai,bj,ij->ab",
        basis,
        basis,
        coefficients,
    )
    indicator = (
        (grid[:, None] < 0.2)
        & (grid[None, :] < 0.2)
    ).astype(float)

    assert np.all(values <= indicator + 1e-9)
    assert values[2, 3] > 0.4


def test_bernstein_overshoot_covers_dense_domain() -> None:
    threshold = 0.6
    margin = 0.2
    degree = 12
    grid = np.linspace(0.0, 1.0, degree + 1)
    soft_grid = np.clip((threshold - grid) / margin, 0.0, 1.0)
    polynomial = BPoly(soft_grid[:, None], [0.0, 1.0])
    evaluation_grid = np.linspace(0.0, 1.0, 200_001)
    soft_values = np.clip(
        (threshold - evaluation_grid) / margin,
        0.0,
        1.0,
    )
    observed_overshoot = float(
        (np.asarray(polynomial(evaluation_grid)).reshape(-1) - soft_values).max()
    )

    certified = bernstein_soft_step_overshoot(
        threshold=threshold,
        margin=margin,
        degree=degree,
    )

    assert certified >= observed_overshoot
    assert certified < 0.3


def test_bounded_kl_lower_bound_handles_endpoints_and_sample_size() -> None:
    assert bounded_kl_lower_bound(0.0, 20, 0.05) == 0.0
    assert bounded_kl_lower_bound(1.0, 20, 0.05) == pytest.approx(
        0.05 ** (1.0 / 20)
    )
    assert bounded_kl_lower_bound(0.9, 100, 0.05) > bounded_kl_lower_bound(
        0.9,
        20,
        0.05,
    )


def test_shared_target_bernstein_certificate_uses_one_target() -> None:
    losses = np.zeros((100, 3000, 3), dtype=float)

    result = shared_target_reliability_lower_bound(
        losses,
        tolerances=[0.5, 0.5, 0.5],
        margins=[0.2, 0.2, 0.2],
        degree=10,
        error_rate=0.05,
        block_seed=11,
    )

    assert result.certificate_mean == pytest.approx(1.0)
    assert result.target_blocks == 100
    assert result.target_records_used == 3000
    assert result.effective_sample_size == 100
    assert result.soft_reliability_lower_bound > 0.0


def test_shared_target_certificate_rejects_uninformative_invalid_losses() -> None:
    losses = np.ones((30, 900, 3), dtype=float)

    result = shared_target_reliability_lower_bound(
        losses,
        tolerances=0.5,
        margins=0.2,
        degree=10,
        error_rate=0.05,
        block_seed=7,
    )

    assert result.certificate_mean == pytest.approx(0.0)
    assert result.soft_reliability_lower_bound == 0.0


def test_shared_target_witness_resolves_deep_validity() -> None:
    losses = np.zeros((30, 3000, 3), dtype=float)

    result = shared_target_witness_lower_bound(
        losses,
        tolerances=[0.1, 0.1, 0.1],
        slacks=[0.1, 0.1, 0.1],
        block_size=100,
        error_rate=0.05,
        block_seed=19,
    )

    assert result.witness_mean == pytest.approx(1.0)
    assert result.target_blocks == 30
    assert result.effective_sample_size == 30
    assert result.invalid_release_witness_ceiling == pytest.approx(0.9**100)
    assert result.reliability_lower_bound > 0.8


def test_shared_target_witness_handles_continuous_scores() -> None:
    losses = np.full((40, 4000, 2), 0.01, dtype=float)

    result = shared_target_witness_lower_bound(
        losses,
        tolerances=[0.2, 0.2],
        slacks=[0.1, 0.1],
        ramp_widths=[0.05, 0.05],
        block_size=100,
        error_rate=0.05,
        block_seed=23,
    )

    assert result.witness_mean == pytest.approx(1.0)
    assert result.reliability_lower_bound > 0.85


def test_conditional_witness_reuses_full_target_and_clears_target() -> None:
    losses = np.zeros((50, 5000, 3), dtype=float)

    result = shared_target_conditional_witness_lower_bound(
        losses,
        tolerances=0.2,
        slacks=0.1,
        ramp_widths=0.02,
        error_rate=0.05,
        mechanisms=3,
    )

    assert result.conditional_score_mean == pytest.approx(1.0)
    assert result.releases == 50
    assert result.target_records == 5000
    assert result.target_contamination_allowance < 1e-20
    assert result.reliability_lower_bound > 0.8


def test_conditional_mean_api_matches_full_loss_api() -> None:
    losses = np.linspace(0.01, 0.08, 40 * 200 * 2).reshape(40, 200, 2)
    arguments = {
        "tolerances": [0.2, 0.3],
        "slacks": [0.08, 0.1],
        "error_rate": 0.05,
        "target_error_fraction": 0.7,
    }
    full = shared_target_conditional_witness_lower_bound(losses, **arguments)
    means = shared_target_conditional_mean_lower_bound(
        losses.mean(axis=1),
        target_records=losses.shape[1],
        **arguments,
    )

    assert means == full


def test_conditional_witness_returns_zero_for_invalid_releases() -> None:
    losses = np.ones((30, 1000, 2), dtype=float)

    result = shared_target_conditional_witness_lower_bound(
        losses,
        tolerances=0.2,
        slacks=0.1,
        error_rate=0.05,
    )

    assert result.conditional_score_mean == pytest.approx(0.0)
    assert result.reliability_lower_bound == 0.0


def test_conditional_witness_target_penalty_decreases_with_target_size() -> None:
    small = shared_target_conditional_witness_lower_bound(
        np.zeros((50, 100, 1)),
        tolerances=0.2,
        slacks=0.05,
        error_rate=0.05,
    )
    large = shared_target_conditional_witness_lower_bound(
        np.zeros((50, 1000, 1)),
        tolerances=0.2,
        slacks=0.05,
        error_rate=0.05,
    )

    assert (
        large.target_contamination_allowance
        < small.target_contamination_allowance
    )


def test_conditional_planning_matches_best_case_resolution() -> None:
    plan = plan_conditional_shared_target(
        target_records=1000,
        minimum_reliability=0.8,
        tolerances=0.2,
        slacks=0.042,
        error_rate=0.05,
        target_error_fraction=0.9,
    )

    assert plan.minimum_target_records < 1000
    assert plan.minimum_releases is not None
    assert plan.minimum_releases > 1
    assert plan.target_contamination_allowance == pytest.approx(
        plan.invalid_release_score_ceiling / plan.target_error_rate
    )


def test_conditional_planning_detects_vacuous_target_size() -> None:
    plan = plan_conditional_shared_target(
        target_records=10,
        minimum_reliability=0.8,
        tolerances=0.2,
        slacks=0.01,
        error_rate=0.05,
        target_error_fraction=0.5,
    )

    assert plan.minimum_target_records > 10
    assert plan.minimum_releases is None


def test_conditional_witness_requires_strict_interior_cutoffs() -> None:
    with pytest.raises(ValueError, match="lower < tolerance - slack"):
        shared_target_conditional_witness_lower_bound(
            np.zeros((20, 100, 1)),
            tolerances=0.2,
            slacks=0.2,
        )


def test_shared_target_certificate_accepts_nonunit_loss_ranges() -> None:
    losses = np.full((80, 1200, 2), -0.8, dtype=float)

    result = shared_target_reliability_lower_bound(
        losses,
        tolerances=[0.0, 0.0],
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        margins=[0.5, 0.5],
        degree=6,
        error_rate=0.05,
        block_seed=17,
    )

    assert result.target_blocks == 100
    assert result.soft_reliability_lower_bound > 0.0


def test_hybrid_bound_is_deterministically_no_smaller() -> None:
    arguments = {
        "identified_error_rate": 0.025,
        "shared_target_error_rate": 0.025,
        "total_error_rate": 0.05,
    }
    assert hybrid_reliability_lower_bound(0.62, 0.79, **arguments) == pytest.approx(
        0.79
    )
    assert hybrid_reliability_lower_bound(0.83, 0.77, **arguments) == pytest.approx(
        0.83
    )


def test_hybrid_bound_rejects_post_hoc_full_level_maximum() -> None:
    with pytest.raises(ValueError, match="sum to at most"):
        hybrid_reliability_lower_bound(
            0.81,
            0.84,
            identified_error_rate=0.05,
            shared_target_error_rate=0.05,
            total_error_rate=0.05,
        )


def test_shared_target_certificate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        shared_target_reliability_lower_bound(
            np.zeros((10, 20)),
            tolerances=0.5,
        )
    with pytest.raises(ValueError, match="Not enough target records"):
        shared_target_reliability_lower_bound(
            np.zeros((10, 10, 3)),
            tolerances=0.5,
            degree=4,
        )
