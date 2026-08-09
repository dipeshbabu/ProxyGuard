import numpy as np
import pandas as pd
import pytest

from proxyguard.core import (
    RiskRequirement,
    audit_adaptive_candidate_stream,
    audit_proxy_candidates,
    audit_proxy_mechanisms,
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
    empirical_bernstein_badness_pvalue,
    empirical_bernstein_lower_bound,
    empirical_bernstein_upper_bound,
    hoeffding_badness_pvalue,
    holm_adjust,
    mechanism_release_lower_count,
    mechanism_validity_pvalue,
    mechanism_violation_pvalue,
    paired_prediction_losses,
    quadratic_alpha_schedule,
)


def test_badness_pvalue_rewards_evidence_below_tolerance() -> None:
    strong = hoeffding_badness_pvalue(np.full(500, -0.2), tolerance=0.0)
    weak = hoeffding_badness_pvalue(np.full(500, -0.05), tolerance=0.0)

    assert strong < weak < 1.0
    assert hoeffding_badness_pvalue(np.full(20, 0.1), tolerance=0.0) == 1.0


def test_empirical_bernstein_uses_low_observed_variance() -> None:
    values = np.full(2000, -0.05)

    empirical = empirical_bernstein_badness_pvalue(values, tolerance=0.0)
    hoeffding = hoeffding_badness_pvalue(values, tolerance=0.0)

    assert empirical < hoeffding


def test_holm_adjustment_is_step_down_and_order_preserving() -> None:
    adjusted = holm_adjust({"b": 0.03, "a": 0.01, "c": 0.2})

    assert list(adjusted) == ["b", "a", "c"]
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.2)


def test_proxy_audit_requires_every_declared_requirement() -> None:
    requirements = [
        RiskRequirement("brier", tolerance=0.0),
        RiskRequirement("cost5x", tolerance=0.0),
    ]
    candidates = {
        "safe": {
            "brier": np.full(2000, -0.2),
            "cost5x": np.full(2000, -0.2),
        },
        "mixed": {
            "brier": np.full(2000, -0.2),
            "cost5x": np.full(2000, 0.2),
        },
    }

    result = audit_proxy_candidates(candidates, requirements, alpha=0.05)
    summary = result.candidate_summary.set_index("Candidate")

    assert summary.loc["safe", "Status"] == "Validated"
    assert summary.loc["mixed", "Status"] == "Violation detected"
    mixed_detail = result.requirement_detail[
        result.requirement_detail["Candidate"].eq("mixed")
    ].set_index("Requirement")
    assert mixed_detail.loc["cost5x", "ViolationDetected"]


def test_proxy_audit_combines_relative_and_absolute_risk_requirements() -> None:
    requirements = [
        RiskRequirement(
            "brier regret",
            tolerance=0.02,
            lower=-1.0,
            upper=1.0,
            estimand="relative_regret",
        ),
        RiskRequirement(
            "proxy brier risk",
            tolerance=0.20,
            lower=0.0,
            upper=1.0,
            estimand="absolute_risk",
        ),
    ]
    result = audit_proxy_candidates(
        {
            "adequate": {
                "brier regret": np.full(4000, 0.0),
                "proxy brier risk": np.full(4000, 0.10),
            },
            "poor source and proxy": {
                "brier regret": np.full(4000, 0.0),
                "proxy brier risk": np.full(4000, 0.40),
            },
        },
        requirements,
        alpha=0.05,
    )
    summary = result.candidate_summary.set_index("Candidate")
    detail = result.requirement_detail.set_index(["Candidate", "Requirement"])

    assert summary.loc["adequate", "Status"] == "Validated"
    assert summary.loc["poor source and proxy", "Status"] == "Violation detected"
    assert (
        detail.loc[("poor source and proxy", "proxy brier risk"), "Estimand"]
        == "absolute_risk"
    )
    assert detail.loc[("adequate", "proxy brier risk"), "MeanValue"] == pytest.approx(
        0.10
    )


def test_reported_two_sided_bounds_split_alpha_across_both_tails() -> None:
    values = np.linspace(-0.2, 0.1, 200)
    result = audit_proxy_candidates(
        {"candidate": {"brier": values}},
        [RiskRequirement("brier", tolerance=0.0)],
        alpha=0.05,
    )
    detail = result.requirement_detail.iloc[0]

    assert detail["SimultaneousLCB"] == pytest.approx(
        empirical_bernstein_lower_bound(values, 0.025)
    )
    assert detail["SimultaneousUCB"] == pytest.approx(
        empirical_bernstein_upper_bound(values, 0.025)
    )


def test_paired_prediction_losses_are_bounded_and_oriented() -> None:
    frame = paired_prediction_losses(
        y_true=[0, 1],
        source_probability=[0.1, 0.9],
        proxy_probability=[0.9, 0.1],
        source_thresholds={5.0: 0.5},
        proxy_thresholds={5.0: 0.5},
        record_ids=["a", "b"],
    )

    assert frame["record_id"].tolist() == ["a", "b"]
    assert (frame["brier_regret"] > 0.0).all()
    assert (frame["cost5x_regret"] > 0.0).all()
    assert frame["brier_regret"].between(-1.0, 1.0).all()
    assert frame["logloss_regret"].between(-1.0, 1.0).all()


def test_requirement_bounds_apply_to_paired_regret_not_component_loss() -> None:
    regrets = {"proxy": {"loss": np.array([-1.0, 1.0])}}

    result = audit_proxy_candidates(
        regrets,
        [RiskRequirement("loss", tolerance=0.5, lower=-1.0, upper=1.0)],
    )

    detail = result.requirement_detail.iloc[0]
    assert detail["LowerBound"] == -1.0
    assert detail["UpperBound"] == 1.0
    with pytest.raises(ValueError, match="must lie in"):
        audit_proxy_candidates(
            regrets,
            [RiskRequirement("loss", tolerance=0.5, lower=0.0, upper=1.0)],
        )


def test_exact_binomial_bounds_handle_endpoints_and_tighten_with_successes() -> None:
    assert clopper_pearson_lower_bound(0, 10, 0.05) == 0.0
    assert clopper_pearson_upper_bound(10, 10, 0.05) == 1.0
    assert clopper_pearson_lower_bound(9, 10, 0.05) > clopper_pearson_lower_bound(
        8,
        10,
        0.05,
    )
    assert clopper_pearson_upper_bound(1, 10, 0.05) < clopper_pearson_upper_bound(
        2,
        10,
        0.05,
    )


def test_mechanism_tail_pvalues_reward_consistent_release_evidence() -> None:
    validity_weak = mechanism_validity_pvalue(20, 30, minimum_reliability=0.8)
    validity_strong = mechanism_validity_pvalue(30, 30, minimum_reliability=0.8)
    violation_weak = mechanism_violation_pvalue(5, 30, minimum_reliability=0.8)
    violation_strong = mechanism_violation_pvalue(30, 30, minimum_reliability=0.8)

    assert validity_strong < validity_weak
    assert violation_strong < violation_weak


def test_mechanism_release_lower_count_modes_are_valid() -> None:
    frame = pd.DataFrame(
        {
            "CandidatePValue": [0.002, 0.02, 0.02, 0.02],
            "Validated": [True, False, False, False],
        }
    )
    holm_count = mechanism_release_lower_count(
        frame,
        mode="holm",
        release_alpha=0.05,
    )
    simes_count = mechanism_release_lower_count(
        frame,
        mode="simes",
        release_alpha=0.05,
    )

    assert holm_count == 1
    assert simes_count == 4
    assert simes_count >= holm_count


def test_mechanism_release_lower_count_rejects_invalid_mode() -> None:
    frame = pd.DataFrame(
        {
            "CandidatePValue": [0.02],
            "Validated": [False],
        }
    )
    with pytest.raises(ValueError, match="must be 'holm' or 'simes'"):
        mechanism_release_lower_count(
            frame,
            mode="invalid",
            release_alpha=0.05,
        )


def test_mechanism_audit_separates_validation_violation_and_low_release_count() -> None:
    requirement = [RiskRequirement("loss", tolerance=0.0)]
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    release_to_mechanism: dict[str, str] = {}
    for release in range(30):
        valid_name = f"reliable::{release}"
        invalid_name = f"unreliable::{release}"
        candidate_regrets[valid_name] = {"loss": np.full(1000, -0.4)}
        candidate_regrets[invalid_name] = {"loss": np.full(1000, 0.4)}
        release_to_mechanism[valid_name] = "reliable"
        release_to_mechanism[invalid_name] = "unreliable"
    for release in range(3):
        name = f"too_few::{release}"
        candidate_regrets[name] = {"loss": np.full(1000, -0.4)}
        release_to_mechanism[name] = "too_few"

    result = audit_proxy_mechanisms(
        candidate_regrets,
        release_to_mechanism,
        requirements=requirement,
        minimum_reliability=0.8,
        total_alpha=0.05,
    )
    summary = result.mechanism_summary.set_index("Mechanism")

    assert summary.loc["reliable", "Status"] == "Mechanism validated"
    assert summary.loc["unreliable", "Status"] == "Reliability violation detected"
    assert summary.loc["too_few", "Status"] == "Unresolved"
    assert summary.loc["reliable", "ReliabilityLCB"] > 0.8
    assert summary.loc["unreliable", "ReliabilityUCB"] < 0.8


def test_mechanism_audit_requires_a_complete_release_mapping() -> None:
    with pytest.raises(ValueError, match="name every candidate"):
        audit_proxy_mechanisms(
            {"release": {"loss": np.full(50, -0.2)}},
            {},
            requirements=[RiskRequirement("loss", tolerance=0.0)],
        )


def test_mechanism_audit_tracks_separate_directional_error_budgets() -> None:
    result = audit_proxy_mechanisms(
        {"release": {"loss": np.full(200, -0.2)}},
        {"release": "mechanism"},
        requirements=[RiskRequirement("loss", tolerance=0.0)],
        total_alpha=0.04,
        release_error_share=0.25,
        violation_total_alpha=0.06,
        violation_release_error_share=0.5,
    )
    row = result.mechanism_summary.iloc[0]

    assert row["ReleaseErrorRate"] == pytest.approx(0.01)
    assert row["MechanismReleaseErrorAllocation"] == pytest.approx(0.01)
    assert row["MechanismErrorRate"] == pytest.approx(0.03)
    assert row["ViolationReleaseErrorRate"] == pytest.approx(0.03)
    assert row["ViolationMechanismErrorRate"] == pytest.approx(0.03)
    assert row["ViolationTotalErrorRate"] == pytest.approx(0.06)
    assert row["LowerBoundSimultaneousCoverage"] == pytest.approx(0.96)
    assert row["UpperBoundSimultaneousCoverage"] == pytest.approx(0.94)
    assert row["DisplayedPairJointCoverageLowerBound"] == pytest.approx(0.90)


def test_mechanism_count_mode_is_recorded_and_returns_release_level_counts() -> None:
    requirements = [RiskRequirement("loss", tolerance=0.0)]
    release_count = 20
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    mapping: dict[str, str] = {}
    for index in range(release_count):
        candidate_regrets[f"mA::{index}"] = {"loss": np.full(500, -0.2)}
        candidate_regrets[f"mB::{index}"] = {"loss": np.full(500, 0.2)}
        mapping[f"mA::{index}"] = "mechanism_a"
        mapping[f"mB::{index}"] = "mechanism_b"

    result = audit_proxy_mechanisms(
        candidate_regrets,
        mapping,
        requirements=requirements,
        minimum_reliability=0.8,
        mechanism_count_mode="simes",
        collective_dependence_verified=True,
        total_alpha=0.05,
    )
    summary = result.mechanism_summary.set_index("Mechanism")

    assert summary.loc["mechanism_a", "Status"] == "Mechanism validated"
    assert summary.loc["mechanism_a", "MechanismCountMode"] == "simes"
    assert bool(summary.loc["mechanism_a", "CollectiveDependenceVerified"])
    assert summary.loc[
        "mechanism_a", "MechanismReleaseErrorAllocation"
    ] == pytest.approx(0.0125)
    assert (
        int(summary.loc["mechanism_a", "IndividuallyValidatedReleases"])
        == release_count
    )
    assert summary.loc["mechanism_b", "Status"] == "Reliability violation detected"


def test_mechanism_audit_requires_explicit_simes_dependence_assertion() -> None:
    with pytest.raises(ValueError, match="collective_dependence_verified"):
        audit_proxy_mechanisms(
            {"release": {"loss": np.full(500, -0.2)}},
            {"release": "mechanism"},
            requirements=[RiskRequirement("loss", tolerance=0.0)],
            mechanism_count_mode="simes",
        )


def test_quadratic_spending_controls_an_unbounded_sequence() -> None:
    spending = quadratic_alpha_schedule(0.05, 100_000)

    assert spending[0] > spending[1] > spending[-1]
    assert spending.sum() < 0.05
    assert spending.sum() == pytest.approx(0.05, rel=1e-5)


def test_adaptive_stream_records_round_specific_error_spending() -> None:
    candidates = {
        "first": {"loss": np.full(2000, -0.4)},
        "second": {"loss": np.full(2000, 0.4)},
    }
    result = audit_adaptive_candidate_stream(
        candidates,
        requirements=[RiskRequirement("loss", tolerance=0.0)],
        total_alpha=0.05,
    )
    summary = result.round_summary.set_index("Candidate")

    assert summary.loc["first", "Status"] == "Validated"
    assert summary.loc["second", "Status"] == "Violation detected"
    assert summary.loc["first", "RoundAlpha"] > summary.loc["second", "RoundAlpha"]
    assert summary.loc["second", "CumulativeAlpha"] < 0.05
