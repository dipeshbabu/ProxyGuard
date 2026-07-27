import pandas as pd
import pytest

from scripts.proxyguard.build_audit_validation_assets import (
    all_hidden_failure,
    assign_pareto_membership,
    classify_state,
    family_failure,
    held_out_failure,
    held_out_failures_by_family,
    matched_retention_summary,
    utility_pass,
    validation_selected,
    visible_order_scores,
)
from scripts.proxyguard.run_temporal_proxy_audit import campaign_boundaries


def test_validation_selected_uses_validation_cost_only() -> None:
    frame = pd.DataFrame(
        {
            "split_seed": [1, 1, 2, 2],
            "Model": ["a", "b", "a", "b"],
            "val_cost_5x_best": [0.4, 0.2, 0.1, 0.3],
            "AUC": [0.9, 0.5, 0.5, 0.9],
        }
    )

    selected = validation_selected(frame)

    assert selected.loc[1, "Model"] == "b"
    assert selected.loc[2, "Model"] == "a"


@pytest.mark.parametrize(
    ("auc", "ece", "cost", "expected"),
    [
        (-0.010, 0.005, 0.010, True),
        (-0.011, 0.000, 0.000, False),
        (0.000, 0.006, 0.000, False),
        (0.000, 0.000, 0.011, False),
    ],
)
def test_utility_pass_includes_tolerance_boundaries(
    auc: float,
    ece: float,
    cost: float,
    expected: bool,
) -> None:
    delta = pd.DataFrame(
        {
            "AUC": [auc],
            "ECE (10-bin)": [ece],
            "DecisionCost5x": [cost],
        }
    )

    assert bool(utility_pass(delta).iloc[0]) is expected


@pytest.mark.parametrize(
    ("auc", "ece", "cost", "gain", "regression", "expected"),
    [
        (-0.011, 0.000, 0.000, True, False, "Ranking-utility failure"),
        (0.000, 0.006, 0.000, True, False, "Decision-utility failure"),
        (0.000, 0.000, 0.011, True, False, "Decision-utility failure"),
        (0.000, 0.000, 0.000, True, False, "Requirements met"),
        (0.000, 0.000, 0.000, True, True, "Conflicting exposure"),
    ],
)
def test_classify_state_uses_non_compensatory_order(
    auc: float,
    ece: float,
    cost: float,
    gain: bool,
    regression: bool,
    expected: str,
) -> None:
    row = pd.Series(
        {
            "AUC": auc,
            "ECE (10-bin)": ece,
            "DecisionCost5x": cost,
        }
    )

    assert classify_state(row, gain=gain, regression=regression) == expected


def test_held_out_failure_checks_every_omitted_dimension() -> None:
    passing = {
        "AUPRC": -0.010,
        "Brier": 0.005,
        "ECE (15-bin)": 0.005,
        "ECE (20-bin)": 0.005,
        "AdaptiveECE (10-bin)": 0.005,
        "DecisionCost10x": 0.010,
        "DecisionCost20x": 0.010,
    }
    frame = pd.DataFrame([passing, {**passing, "DecisionCost20x": 0.011}])

    result = held_out_failure(frame)

    assert result.tolist() == [False, True]


def test_held_out_failures_are_grouped_by_metric_family() -> None:
    passing = {
        "AUPRC": -0.010,
        "Brier": 0.005,
        "ECE (15-bin)": 0.005,
        "ECE (20-bin)": 0.005,
        "AdaptiveECE (10-bin)": 0.005,
        "DecisionCost10x": 0.010,
        "DecisionCost20x": 0.010,
    }
    frame = pd.DataFrame(
        [
            passing,
            {
                **passing,
                "ECE (20-bin)": 0.006,
                "DecisionCost20x": 0.011,
            },
        ]
    )

    result = held_out_failures_by_family(frame)

    assert result.loc[0].tolist() == [False, False, False, False, False]
    assert result.loc[1].tolist() == [False, False, True, True, True]


def test_visible_order_scores_use_oriented_tolerance_margins() -> None:
    detail = pd.DataFrame(
        {
            "AUCDelta": [-0.010, 0.000],
            "ECEDelta": [0.005, 0.000],
            "CostDelta": [0.010, 0.000],
        }
    )

    scores = visible_order_scores(detail)

    assert scores.loc[0, "ScalarScore"] == pytest.approx(0.0)
    assert scores.loc[0, "NonCompensatoryScore"] == pytest.approx(0.0)
    assert scores.loc[1, "ScalarScore"] == pytest.approx(1.0)
    assert scores.loc[1, "NonCompensatoryScore"] == pytest.approx(1.0)


def test_family_failure_hides_the_entire_calibration_family() -> None:
    detail = pd.DataFrame(
        {
            "ECEDelta": [0.000, 0.006],
            "ECE (15-bin)Delta": [0.000, 0.000],
            "ECE (20-bin)Delta": [0.000, 0.000],
            "AdaptiveECE (10-bin)Delta": [0.000, 0.000],
        }
    )

    result = family_failure(detail, "Calibration")

    assert result.tolist() == [False, True]


def test_all_hidden_failure_respects_tolerance_scale() -> None:
    passing = {
        "AUPRCDelta": -0.015,
        "BrierDelta": 0.000,
        "ECE (15-bin)Delta": 0.000,
        "ECE (20-bin)Delta": 0.000,
        "AdaptiveECE (10-bin)Delta": 0.000,
        "DecisionCost10xDelta": 0.000,
        "DecisionCost20xDelta": 0.000,
    }
    detail = pd.DataFrame([passing])

    assert bool(all_hidden_failure(detail, tolerance_scale=1.0).iloc[0])
    assert not bool(all_hidden_failure(detail, tolerance_scale=2.0).iloc[0])


def test_matched_retention_uses_equal_case_counts() -> None:
    detail = pd.DataFrame(
        {
            "Dataset": ["d"] * 4,
            "Variant": ["a", "b", "c", "d"],
            "split_seed": [1, 1, 1, 1],
            "HeldOutFailure": [False, True, False, True],
            "AUCOnly": [True, True, False, False],
            "ScalarMean": [True, False, True, False],
            "NonCompensatory": [True, False, False, False],
            "AUCScore": [4.0, 3.0, 2.0, 1.0],
            "ScalarScore": [1.0, 4.0, 3.0, 2.0],
            "NonCompensatoryScore": [2.0, 1.0, 4.0, 3.0],
            "AUCDelta": [0.04, 0.03, 0.02, 0.01],
            "ECEDelta": [0.00, 0.01, 0.02, 0.03],
            "CostDelta": [0.00, 0.01, 0.02, 0.03],
        }
    )
    detail = assign_pareto_membership(detail)

    summary = matched_retention_summary(detail, retained_cases=2)

    assert summary["SelectedCases"].tolist() == [2, 2, 2, 2]
    assert summary["TotalCases"].tolist() == [4, 4, 4, 4]


def test_campaign_boundaries_find_year_transitions() -> None:
    frame = pd.DataFrame(
        {
            "month": [
                "nov",
                "dec",
                "jan",
                "jan",
                "feb",
                "dec",
                "jan",
                "jan",
                "feb",
            ]
        }
    )

    assert campaign_boundaries(frame) == (2, 6)
