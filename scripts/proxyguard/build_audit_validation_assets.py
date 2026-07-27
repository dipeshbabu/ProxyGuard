from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ruap_audit import build_exposure_table  # noqa: E402

MAIN_VARIANTS = {
    "australian_credit": [
        "numeric_noise_10",
        "numeric_noise_20",
        "coarsen_quartile",
        "feature_mask_20",
        "sensitive_mask",
    ],
    "german_credit": ["feature_mask_20", "sensitive_mask"],
    "compas_recidivism": ["sensitive_mask"],
}

DISPLAY_DATASETS = {
    "australian_credit": "Australian",
    "german_credit": "German",
    "compas_recidivism": "COMPAS",
}

DISPLAY_VARIANTS = {
    "numeric_noise_10": "10\\% noise",
    "numeric_noise_20": "20\\% noise",
    "coarsen_quartile": "Quartile",
    "feature_mask_20": "20\\% mask",
    "sensitive_mask": "Sensitive mask",
}

PRIMARY_METRICS = ["AUC", "ECE (10-bin)", "DecisionCost5x"]
HELD_OUT_CHECKS = {
    "AUPRC": ("ge", -0.010),
    "Brier": ("le", 0.005),
    "ECE (15-bin)": ("le", 0.005),
    "ECE (20-bin)": ("le", 0.005),
    "AdaptiveECE (10-bin)": ("le", 0.005),
    "DecisionCost10x": ("le", 0.010),
    "DecisionCost20x": ("le", 0.010),
}
HELD_OUT_FAMILIES = {
    "Ranking": ["AUPRC"],
    "Probability": ["Brier"],
    "Calibration": [
        "ECE (15-bin)",
        "ECE (20-bin)",
        "AdaptiveECE (10-bin)",
    ],
    "Cost": ["DecisionCost10x", "DecisionCost20x"],
}
RULES = {
    "AUCOnly": "AUC gate",
    "ScalarMean": "Scalar mean",
    "Pareto": "Pareto + AUC tie-break",
    "NonCompensatory": "Non-compensatory gates",
}
RULE_SCORES = {
    "AUCOnly": "AUCScore",
    "ScalarMean": "ScalarScore",
    "NonCompensatory": "NonCompensatoryScore",
}
VISIBLE_TOLERANCES = {
    "AUC": 0.010,
    "ECE": 0.005,
    "Cost": 0.010,
}
VISIBLE_DELTA_COLUMNS = {
    "AUC": "AUCDelta",
    "ECE": "ECEDelta",
    "Cost": "CostDelta",
}
FULL_METRIC_FAMILIES = {
    "Ranking": [
        ("AUCDelta", "ge", 0.010),
        ("AUPRCDelta", "ge", 0.010),
    ],
    "Probability quality": [
        ("BrierDelta", "le", 0.005),
    ],
    "Calibration": [
        ("ECEDelta", "le", 0.005),
        ("ECE (15-bin)Delta", "le", 0.005),
        ("ECE (20-bin)Delta", "le", 0.005),
        ("AdaptiveECE (10-bin)Delta", "le", 0.005),
    ],
    "Asymmetric cost": [
        ("CostDelta", "le", 0.010),
        ("DecisionCost10xDelta", "le", 0.010),
        ("DecisionCost20xDelta", "le", 0.010),
    ],
}
VISIBLE_FAMILY = {
    "AUC": "Ranking",
    "ECE": "Calibration",
    "Cost": "Asymmetric cost",
}
SENSITIVITY_MULTIPLIERS = (0.50, 0.75, 1.00, 1.50, 2.00)


def read_split(root: Path, dataset: str, variant: str) -> pd.DataFrame:
    path = root / "proxy_transform" / variant / dataset / "split_metrics.csv"
    frame = pd.read_csv(path)
    required = {
        "Model",
        "split_seed",
        "val_cost_5x_best",
        *PRIMARY_METRICS,
        *HELD_OUT_CHECKS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def validation_selected(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one learner per split using validation Cost5x only."""
    indices = frame.groupby("split_seed")["val_cost_5x_best"].idxmin()
    return frame.loc[indices].set_index("split_seed").sort_index()


def paired_delta(current: pd.DataFrame, baseline: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    common = current.index.intersection(baseline.index)
    if len(common) == 0:
        raise ValueError("No common split seeds between original and proxy outputs.")
    return current.loc[common, columns] - baseline.loc[common, columns]


def utility_pass(delta: pd.DataFrame) -> pd.Series:
    return (
        (delta["AUC"] >= -0.010)
        & (delta["ECE (10-bin)"] <= 0.005)
        & (delta["DecisionCost5x"] <= 0.010)
    )


def exposure_state(exposure_row: pd.Series) -> tuple[bool, bool]:
    values = exposure_row[
        [
            "UniquenessDelta",
            "NearestNeighborRiskDelta",
            "SensitivePredictabilityDelta",
        ]
    ].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    gain = bool(len(finite) and np.min(finite) < -0.010)
    regression = bool(len(finite) and np.max(finite) > 0.010)
    return gain, regression


def classify_state(delta_row: pd.Series, gain: bool, regression: bool) -> str:
    if delta_row["AUC"] < -0.010:
        return "Ranking-utility failure"
    if delta_row["ECE (10-bin)"] > 0.005 or delta_row["DecisionCost5x"] > 0.010:
        return "Decision-utility failure"
    if gain and not regression:
        return "Requirements met"
    return "Conflicting exposure"


def fixed_model_summary(
    root: Path,
    report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []

    for dataset, variants in MAIN_VARIANTS.items():
        baseline_raw = read_split(root, dataset, "baseline")
        baseline_model = baseline_raw.groupby("Model")[PRIMARY_METRICS].mean()
        baseline_selected = validation_selected(baseline_raw)

        for variant in variants:
            current_raw = read_split(root, dataset, variant)
            current_model = current_raw.groupby("Model")[PRIMARY_METRICS].mean()
            model_delta = current_model - baseline_model
            model_pass = utility_pass(model_delta)

            fixed_rows.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "ModelsPassing": int(model_pass.sum()),
                    "ModelsTotal": len(model_pass),
                    "AUCDeltaMin": float(model_delta["AUC"].min()),
                    "AUCDeltaMax": float(model_delta["AUC"].max()),
                    "ECEDeltaMin": float(model_delta["ECE (10-bin)"].min()),
                    "ECEDeltaMax": float(model_delta["ECE (10-bin)"].max()),
                    "CostDeltaMin": float(model_delta["DecisionCost5x"].min()),
                    "CostDeltaMax": float(model_delta["DecisionCost5x"].max()),
                }
            )

            current_selected = validation_selected(current_raw)
            selected_delta = paired_delta(
                current_selected,
                baseline_selected,
                [*PRIMARY_METRICS, *HELD_OUT_CHECKS],
            )
            same_model = (
                current_selected.loc[selected_delta.index, "Model"]
                == baseline_selected.loc[selected_delta.index, "Model"]
            )
            selected_rows.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "AUCDelta": float(selected_delta["AUC"].mean()),
                    "ECEDelta": float(selected_delta["ECE (10-bin)"].mean()),
                    "CostDelta": float(selected_delta["DecisionCost5x"].mean()),
                    "SameModelRate": float(same_model.mean()),
                    "Splits": len(selected_delta),
                }
            )

            exposure_row = report[
                (report["Dataset"] == dataset) & (report["Variant"] == variant)
            ]
            if len(exposure_row) != 1:
                raise ValueError(f"Expected one exposure row for {dataset}/{variant}.")
            gain, regression = exposure_state(exposure_row.iloc[0])
            states = selected_delta.apply(
                classify_state,
                axis=1,
                gain=gain,
                regression=regression,
            )
            frequencies = states.value_counts(normalize=True)
            state_rows.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "RequirementsMet": float(frequencies.get("Requirements met", 0.0)),
                    "ConflictingExposure": float(frequencies.get("Conflicting exposure", 0.0)),
                    "DecisionUtilityFailure": float(
                        frequencies.get("Decision-utility failure", 0.0)
                    ),
                    "RankingUtilityFailure": float(
                        frequencies.get("Ranking-utility failure", 0.0)
                    ),
                    "Splits": len(states),
                }
            )

    return (
        pd.DataFrame(fixed_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(state_rows),
    )


def repeated_core_exposure(
    seeds: list[int],
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the three core exposure probes and retain paired seed-level deltas."""
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        for dataset, variants in MAIN_VARIANTS.items():
            frame = build_exposure_table(
                [dataset],
                ["baseline", *variants],
                seed=seed,
                max_rows=max_rows,
            )
            frame["ExposureSeed"] = seed
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, object]] = []
    for (dataset, seed), group in raw.groupby(["Dataset", "ExposureSeed"]):
        baseline = group[group["Variant"] == "baseline"]
        if len(baseline) != 1:
            raise ValueError(f"Expected one baseline for {dataset}, seed {seed}.")
        baseline_row = baseline.iloc[0]
        for _, row in group[group["Variant"] != "baseline"].iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": row["Variant"],
                    "ExposureSeed": int(seed),
                    "UniquenessDelta": float(
                        row["UniquenessRate"]
                        - baseline_row["UniquenessRate"]
                    ),
                    "NearestNeighborRiskDelta": float(
                        row["NearestNeighborRisk"]
                        - baseline_row["NearestNeighborRisk"]
                    ),
                    "SensitivePredictabilityDelta": float(
                        row["SensitivePredictability"]
                        - baseline_row["SensitivePredictability"]
                    ),
                    "SensitiveTarget": row["SensitiveTarget"],
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Dataset", "Variant"])
        .agg(
            ExposureSeeds=("ExposureSeed", "nunique"),
            UniquenessDelta=("UniquenessDelta", "mean"),
            UniquenessDeltaSD=("UniquenessDelta", "std"),
            NearestNeighborRiskDelta=("NearestNeighborRiskDelta", "mean"),
            NearestNeighborRiskDeltaSD=("NearestNeighborRiskDelta", "std"),
            SensitivePredictabilityDelta=(
                "SensitivePredictabilityDelta",
                "mean",
            ),
            SensitivePredictabilityDeltaSD=(
                "SensitivePredictabilityDelta",
                "std",
            ),
        )
        .reset_index()
    )
    return detail, summary


def joint_state_stability(
    root: Path,
    exposure_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Cross utility splits with repeated exposure estimates for each intervention."""
    rows: list[dict[str, object]] = []
    for dataset, variants in MAIN_VARIANTS.items():
        baseline = validation_selected(read_split(root, dataset, "baseline"))
        for variant in variants:
            current = validation_selected(read_split(root, dataset, variant))
            utility_delta = paired_delta(
                current,
                baseline,
                PRIMARY_METRICS,
            )
            exposure = exposure_detail[
                (exposure_detail["Dataset"] == dataset)
                & (exposure_detail["Variant"] == variant)
            ]
            states: list[str] = []
            for _, exposure_row in exposure.iterrows():
                gain, regression = exposure_state(exposure_row)
                states.extend(
                    utility_delta.apply(
                        classify_state,
                        axis=1,
                        gain=gain,
                        regression=regression,
                    ).tolist()
                )
            frequencies = pd.Series(states).value_counts(normalize=True)
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "RequirementsMet": float(
                        frequencies.get("Requirements met", 0.0)
                    ),
                    "ConflictingExposure": float(
                        frequencies.get("Conflicting exposure", 0.0)
                    ),
                    "DecisionUtilityFailure": float(
                        frequencies.get("Decision-utility failure", 0.0)
                    ),
                    "RankingUtilityFailure": float(
                        frequencies.get("Ranking-utility failure", 0.0)
                    ),
                    "UtilitySplits": len(utility_delta),
                    "ExposureSeeds": len(exposure),
                    "StateEvaluations": len(states),
                }
            )
    return pd.DataFrame(rows)


def report_with_repeated_exposure(
    report: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Replace single-seed exposure deltas with repeated-seed means."""
    repeated_columns = [
        "Dataset",
        "Variant",
        "UniquenessDelta",
        "NearestNeighborRiskDelta",
        "SensitivePredictabilityDelta",
    ]
    merged = report.drop(
        columns=[
            "UniquenessDelta",
            "NearestNeighborRiskDelta",
            "SensitivePredictabilityDelta",
        ]
    ).merge(
        exposure_summary[repeated_columns],
        on=["Dataset", "Variant"],
        how="left",
    )
    return merged


def metric_failure(delta: pd.DataFrame, metric: str) -> pd.Series:
    operator, tolerance = HELD_OUT_CHECKS[metric]
    if operator == "ge":
        return delta[metric] < tolerance
    return delta[metric] > tolerance


def held_out_failures_by_family(delta: pd.DataFrame) -> pd.DataFrame:
    failures = pd.DataFrame(index=delta.index)
    for family, metrics in HELD_OUT_FAMILIES.items():
        family_failure = pd.Series(False, index=delta.index)
        for metric in metrics:
            family_failure |= metric_failure(delta, metric)
        failures[family] = family_failure
    failures["Any"] = failures.any(axis=1)
    return failures


def held_out_failure(delta: pd.DataFrame) -> pd.Series:
    return held_out_failures_by_family(delta)["Any"]


def visible_order_scores(
    detail: pd.DataFrame,
    tolerances: dict[str, float] | None = None,
    visible_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute scalar and non-compensatory scores from declared utility metrics."""
    tolerances = tolerances or VISIBLE_TOLERANCES
    visible_metrics = visible_metrics or list(VISIBLE_TOLERANCES)
    margins: dict[str, pd.Series] = {}
    for metric in visible_metrics:
        delta = detail[VISIBLE_DELTA_COLUMNS[metric]]
        tolerance = tolerances[metric]
        if metric == "AUC":
            margins[metric] = (delta + tolerance) / tolerance
        else:
            margins[metric] = (tolerance - delta) / tolerance
    margin_frame = pd.DataFrame(margins, index=detail.index)
    return pd.DataFrame(
        {
            "ScalarScore": margin_frame.mean(axis=1),
            "NonCompensatoryScore": margin_frame.min(axis=1),
        },
        index=detail.index,
    )


def family_failure(
    detail: pd.DataFrame,
    family: str,
    tolerance_scale: float = 1.0,
) -> pd.Series:
    """Return failures for every measurement in one withheld metric family."""
    failure = pd.Series(False, index=detail.index)
    for column, direction, base_tolerance in FULL_METRIC_FAMILIES[family]:
        tolerance = base_tolerance * tolerance_scale
        if direction == "ge":
            failure |= detail[column] < -tolerance
        else:
            failure |= detail[column] > tolerance
    return failure


def all_hidden_failure(
    detail: pd.DataFrame,
    tolerance_scale: float = 1.0,
) -> pd.Series:
    """Recompute the original seven-check endpoint at a scaled tolerance."""
    failure = pd.Series(False, index=detail.index)
    for family in HELD_OUT_FAMILIES:
        full_family = (
            "Probability quality" if family == "Probability" else
            "Asymmetric cost" if family == "Cost" else
            family
        )
        family_metrics = FULL_METRIC_FAMILIES[full_family]
        if family == "Ranking":
            family_metrics = [metric for metric in family_metrics if metric[0] == "AUPRCDelta"]
        elif family == "Calibration":
            family_metrics = [metric for metric in family_metrics if metric[0] != "ECEDelta"]
        elif family == "Cost":
            family_metrics = [metric for metric in family_metrics if metric[0] != "CostDelta"]
        for column, direction, base_tolerance in family_metrics:
            tolerance = base_tolerance * tolerance_scale
            if direction == "ge":
                failure |= detail[column] < -tolerance
            else:
                failure |= detail[column] > tolerance
    return failure


def ordered_by_score(detail: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return detail.sort_values(
        [score_column, "Dataset", "Variant", "split_seed"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )


def omission_tolerance_sensitivity(
    detail: pd.DataFrame,
    retained_cases: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vary one visible tolerance and all hidden tolerances over a fixed grid."""
    rows: list[dict[str, object]] = []
    for varied_metric in VISIBLE_TOLERANCES:
        for visible_scale in SENSITIVITY_MULTIPLIERS:
            tolerances = dict(VISIBLE_TOLERANCES)
            tolerances[varied_metric] *= visible_scale
            scores = visible_order_scores(detail, tolerances=tolerances)
            scored = detail.copy()
            scored[["ScalarScore", "NonCompensatoryScore"]] = scores
            for hidden_scale in SENSITIVITY_MULTIPLIERS:
                scored["SensitivityFailure"] = all_hidden_failure(
                    scored,
                    tolerance_scale=hidden_scale,
                )
                result: dict[str, object] = {
                    "VariedVisibleTolerance": varied_metric,
                    "VisibleScale": visible_scale,
                    "HiddenScale": hidden_scale,
                    "RetainedCases": retained_cases,
                }
                for rule, score_column in [
                    ("Scalar", "ScalarScore"),
                    ("NonCompensatory", "NonCompensatoryScore"),
                ]:
                    ordered = ordered_by_score(scored, score_column)
                    selected = ordered.head(retained_cases)
                    risks = (
                        ordered["SensitivityFailure"].astype(int).cumsum()
                        / np.arange(1, len(ordered) + 1)
                    )
                    result[f"{rule}Failures"] = int(
                        selected["SensitivityFailure"].sum()
                    )
                    result[f"{rule}FailureRate"] = float(
                        selected["SensitivityFailure"].mean()
                    )
                    result[f"{rule}AURC"] = float(risks.mean())
                result["MatchedRiskDifference"] = float(
                    result["NonCompensatoryFailureRate"]
                    - result["ScalarFailureRate"]
                )
                result["AURCDifference"] = float(
                    result["NonCompensatoryAURC"] - result["ScalarAURC"]
                )
                rows.append(result)
    detail_out = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    for varied_metric, group in detail_out.groupby("VariedVisibleTolerance"):
        matched_difference = group["MatchedRiskDifference"]
        aurc_difference = group["AURCDifference"]
        summary_rows.append(
            {
                "VariedVisibleTolerance": varied_metric,
                "Settings": len(group),
                "ScalarMatchedWins": int((matched_difference > 1e-12).sum()),
                "TiesAtMatchedRetention": int(
                    (matched_difference.abs() <= 1e-12).sum()
                ),
                "NonCompensatoryMatchedWins": int(
                    (matched_difference < -1e-12).sum()
                ),
                "MatchedRiskDifferenceMin": float(matched_difference.min()),
                "MatchedRiskDifferenceMax": float(matched_difference.max()),
                "ScalarAURCWins": int((aurc_difference > 1e-12).sum()),
                "TiesOnAURC": int((aurc_difference.abs() <= 1e-12).sum()),
                "NonCompensatoryAURCWins": int(
                    (aurc_difference < -1e-12).sum()
                ),
                "AURCDifferenceMin": float(aurc_difference.min()),
                "AURCDifferenceMax": float(aurc_difference.max()),
            }
        )
    return detail_out, pd.DataFrame(summary_rows)


def leave_one_family_out(
    detail: pd.DataFrame,
    retained_cases: int,
) -> pd.DataFrame:
    """Hide one metric family and rank using representatives of the others."""
    rows: list[dict[str, object]] = []
    for hidden_family in FULL_METRIC_FAMILIES:
        visible_metrics = [
            metric
            for metric, family in VISIBLE_FAMILY.items()
            if family != hidden_family
        ]
        scores = visible_order_scores(
            detail,
            visible_metrics=visible_metrics,
        )
        scored = detail.copy()
        scored[["ScalarScore", "NonCompensatoryScore"]] = scores
        scored["FamilyFailure"] = family_failure(scored, hidden_family)
        for rule, score_column in [
            ("Scalar mean", "ScalarScore"),
            ("Non-compensatory gates", "NonCompensatoryScore"),
        ]:
            selected = ordered_by_score(scored, score_column).head(retained_cases)
            failures = int(selected["FamilyFailure"].sum())
            rows.append(
                {
                    "HiddenFamily": hidden_family,
                    "VisibleMetrics": "+".join(visible_metrics),
                    "Rule": rule,
                    "SelectedCases": len(selected),
                    "HiddenFamilyFailures": failures,
                    "HiddenFamilyFailureRate": float(
                        selected["FamilyFailure"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def ece_only_failure_summary(
    detail: pd.DataFrame,
    retained_cases: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule, label in RULES.items():
        selected = order_cases(detail, rule).head(retained_cases)
        no_brier_or_cost = (
            selected["CalibrationFailure"]
            & ~selected["ProbabilityFailure"]
            & ~selected["CostFailure"]
        )
        ece_only = no_brier_or_cost & ~selected["RankingFailure"]
        rows.append(
            {
                "Rule": label,
                "SelectedCases": len(selected),
                "AlternativeECEWithoutBrierOrCost": int(
                    no_brier_or_cost.sum()
                ),
                "AlternativeECEOnly": int(ece_only.sum()),
            }
        )
    return pd.DataFrame(rows)


def assign_pareto_membership(detail: pd.DataFrame) -> pd.DataFrame:
    detail = detail.copy()
    detail["Pareto"] = False
    for (_, _), group in detail.groupby(["Dataset", "split_seed"]):
        feasible = group[group["AUCOnly"]]
        for index, candidate in feasible.iterrows():
            dominated = (
                (feasible["AUCDelta"] >= candidate["AUCDelta"])
                & (feasible["ECEDelta"] <= candidate["ECEDelta"])
                & (feasible["CostDelta"] <= candidate["CostDelta"])
                & (
                    (feasible["AUCDelta"] > candidate["AUCDelta"])
                    | (feasible["ECEDelta"] < candidate["ECEDelta"])
                    | (feasible["CostDelta"] < candidate["CostDelta"])
                )
            ).any()
            detail.loc[index, "Pareto"] = not bool(dominated)
    return detail


def omission_stress_test(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for dataset, variants in MAIN_VARIANTS.items():
        baseline = validation_selected(read_split(root, dataset, "baseline"))
        for variant in variants:
            current = validation_selected(read_split(root, dataset, variant))
            columns = [*PRIMARY_METRICS, *HELD_OUT_CHECKS]
            delta = paired_delta(current, baseline, columns)
            held_failures = held_out_failures_by_family(delta)
            auc_margin = (delta["AUC"] + 0.010) / 0.010
            ece_margin = (0.005 - delta["ECE (10-bin)"]) / 0.005
            cost_margin = (0.010 - delta["DecisionCost5x"]) / 0.010
            for split_seed in delta.index:
                scalar_score = float(
                    np.mean(
                        [
                            auc_margin.loc[split_seed],
                            ece_margin.loc[split_seed],
                            cost_margin.loc[split_seed],
                        ]
                    )
                )
                non_compensatory_score = float(
                    min(
                        auc_margin.loc[split_seed],
                        ece_margin.loc[split_seed],
                        cost_margin.loc[split_seed],
                    )
                )
                rows.append(
                    {
                        "Dataset": dataset,
                        "Variant": variant,
                        "split_seed": int(split_seed),
                        "HeldOutFailure": bool(
                            held_failures.loc[split_seed, "Any"]
                        ),
                        **{
                            f"{family}Failure": bool(
                                held_failures.loc[split_seed, family]
                            )
                            for family in HELD_OUT_FAMILIES
                        },
                        "AUCOnly": bool(auc_margin.loc[split_seed] >= 0),
                        "ScalarMean": bool(scalar_score >= 0),
                        "NonCompensatory": bool(non_compensatory_score >= 0),
                        "AUCScore": float(auc_margin.loc[split_seed]),
                        "ScalarScore": scalar_score,
                        "NonCompensatoryScore": non_compensatory_score,
                        "AUCDelta": float(delta.loc[split_seed, "AUC"]),
                        "ECEDelta": float(delta.loc[split_seed, "ECE (10-bin)"]),
                        "CostDelta": float(delta.loc[split_seed, "DecisionCost5x"]),
                        **{
                            f"{metric}Delta": float(delta.loc[split_seed, metric])
                            for metric in HELD_OUT_CHECKS
                        },
                    }
                )

    detail = assign_pareto_membership(pd.DataFrame(rows))

    summary_rows = []
    for key, label in [
        ("AUCOnly", "AUC gate"),
        ("ScalarMean", "Scalar mean"),
        ("Pareto", "AUC-feasible Pareto set"),
        ("NonCompensatory", "Non-compensatory gates"),
    ]:
        selected = detail[detail[key]]
        summary_rows.append(
            {
                "Rule": label,
                "SelectedCases": len(selected),
                "HeldOutFailures": int(selected["HeldOutFailure"].sum()),
                "HeldOutFailureRate": float(selected["HeldOutFailure"].mean()),
                "TotalCases": len(detail),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def order_cases(detail: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return a deterministic ordering for matched-retention comparisons."""
    tie_breakers = ["Dataset", "Variant", "split_seed"]
    if rule == "Pareto":
        return detail.sort_values(
            ["Pareto", "AUCScore", *tie_breakers],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
    score = RULE_SCORES[rule]
    return detail.sort_values(
        [score, *tie_breakers],
        ascending=[False, True, True, True],
        kind="mergesort",
    )


def matched_retention_summary(
    detail: pd.DataFrame,
    retained_cases: int,
) -> pd.DataFrame:
    rows = []
    for rule, label in RULES.items():
        selected = order_cases(detail, rule).head(retained_cases)
        failures = int(selected["HeldOutFailure"].sum())
        rows.append(
            {
                "Rule": label,
                "SelectedCases": len(selected),
                "HeldOutFailures": failures,
                "HiddenSafeRetained": int(len(selected) - failures),
                "HeldOutFailureRate": float(selected["HeldOutFailure"].mean()),
                "TotalCases": len(detail),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage_curve(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total = len(detail)
    for rule, label in RULES.items():
        ordered = order_cases(detail, rule)
        cumulative_failures = ordered["HeldOutFailure"].astype(int).cumsum()
        risks = cumulative_failures / np.arange(1, total + 1)
        for retained, risk in enumerate(risks, start=1):
            curve_rows.append(
                {
                    "Rule": label,
                    "RetainedCases": retained,
                    "Coverage": retained / total,
                    "HeldOutFailureRate": float(risk),
                }
            )
        summary_rows.append(
            {
                "Rule": label,
                "AURC": float(risks.mean()),
                "TotalCases": total,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def family_failure_summary(
    detail: pd.DataFrame,
    retained_cases: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selections = {
        "Default": {
            rule: detail[detail[rule]]
            for rule in RULES
        },
        f"Matched {retained_cases}": {
            rule: order_cases(detail, rule).head(retained_cases)
            for rule in RULES
        },
    }
    for comparison, by_rule in selections.items():
        for rule, selected in by_rule.items():
            rows.append(
                {
                    "Comparison": comparison,
                    "Rule": RULES[rule],
                    "SelectedCases": len(selected),
                    **{
                        f"{family}Failures": int(
                            selected[f"{family}Failure"].sum()
                        )
                        for family in HELD_OUT_FAMILIES
                    },
                    "AnyFailures": int(selected["HeldOutFailure"].sum()),
                }
            )
    return pd.DataFrame(rows)


def intervention_robustness(
    detail: pd.DataFrame,
    retained_cases: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, object]] = []
    for rule, label in RULES.items():
        selected_mask = detail[rule]
        for (dataset, variant), group in detail.groupby(["Dataset", "Variant"]):
            selected = group[selected_mask.loc[group.index]]
            group_rows.append(
                {
                    "Rule": label,
                    "Dataset": dataset,
                    "Variant": variant,
                    "SelectedCases": len(selected),
                    "HeldOutFailures": int(selected["HeldOutFailure"].sum()),
                    "HeldOutFailureRate": (
                        float(selected["HeldOutFailure"].mean())
                        if len(selected)
                        else np.nan
                    ),
                    "TotalCases": len(group),
                }
            )
    group_detail = pd.DataFrame(group_rows)
    macro_rows = []
    for label, group in group_detail.groupby("Rule"):
        observed = group.dropna(subset=["HeldOutFailureRate"])
        macro_rows.append(
            {
                "Rule": label,
                "InterventionsRepresented": len(observed),
                "InterventionsTotal": len(group),
                "MacroFailureRate": float(observed["HeldOutFailureRate"].mean()),
                "MinInterventionFailureRate": float(
                    observed["HeldOutFailureRate"].min()
                ),
                "MaxInterventionFailureRate": float(
                    observed["HeldOutFailureRate"].max()
                ),
            }
        )

    target_coverage = retained_cases / len(detail)
    robustness_rows: list[dict[str, object]] = []
    intervention_groups = list(
        detail[["Dataset", "Variant"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    datasets = sorted(detail["Dataset"].unique())
    for omission_type, omissions in [
        ("Intervention", intervention_groups),
        ("Dataset", datasets),
    ]:
        for omitted in omissions:
            if omission_type == "Intervention":
                dataset, variant = omitted
                remaining = detail[
                    ~(
                        (detail["Dataset"] == dataset)
                        & (detail["Variant"] == variant)
                    )
                ]
                omitted_label = f"{dataset}/{variant}"
                remaining = assign_pareto_membership(remaining)
            else:
                remaining = detail[detail["Dataset"] != omitted]
                omitted_label = str(omitted)
            local_retained = max(1, round(target_coverage * len(remaining)))
            for rule, label in RULES.items():
                selected = order_cases(remaining, rule).head(local_retained)
                robustness_rows.append(
                    {
                        "OmissionType": omission_type,
                        "Omitted": omitted_label,
                        "Rule": label,
                        "SelectedCases": len(selected),
                        "HeldOutFailureRate": float(
                            selected["HeldOutFailure"].mean()
                        ),
                    }
                )
    robustness = pd.DataFrame(robustness_rows)
    robustness_summary = (
        robustness.groupby(["OmissionType", "Rule"])["HeldOutFailureRate"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "MeanFailureRate",
                "min": "MinFailureRate",
                "max": "MaxFailureRate",
            }
        )
    )
    return group_detail, pd.DataFrame(macro_rows), robustness_summary


def exposure_omission_test(
    exposure_csv: Path,
    adaptive_attack_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure = pd.read_csv(exposure_csv)
    exposure = exposure[exposure["Variant"] != "baseline"].copy()
    core_columns = [
        "UniquenessRateDelta",
        "NearestNeighborRiskDelta",
        "SensitivePredictabilityDelta",
    ]
    core = exposure[core_columns].astype(float)
    finite_count = core.notna().sum(axis=1)
    core_gain = (core < -0.010).any(axis=1)
    core_regression = (core > 0.010).any(axis=1)
    exposure["UniquenessGain"] = core["UniquenessRateDelta"] < -0.010
    exposure["ScalarCoreGain"] = (
        (-core.fillna(0.0) / 0.010).sum(axis=1) / finite_count
    ) > 1.0
    exposure["NonCompensatoryCore"] = core_gain & ~core_regression
    exposure["HiddenRegression"] = (
        (exposure["AuxLinkAUCDelta"] > 0.010)
        | (exposure["MemberAUCDelta"] > 0.010)
    )
    exposure["HiddenJointGainMissing"] = (
        (exposure["AuxLinkAUCDelta"] >= -0.010)
        | (exposure["MemberAUCDelta"] >= -0.010)
    )

    adaptive = pd.read_csv(adaptive_attack_csv)
    adaptive = adaptive[adaptive["Variant"] != "baseline"][
        ["Dataset", "Variant", "AdaptiveAttackDelta"]
    ]
    exposure = exposure.merge(adaptive, on=["Dataset", "Variant"], how="left")
    exposure["AdaptiveRegression"] = exposure["AdaptiveAttackDelta"] > 0.010
    exposure["AdaptiveGainMissing"] = exposure["AdaptiveAttackDelta"] >= -0.010

    rows = []
    for rule, label in [
        ("UniquenessGain", "Uniqueness gain"),
        ("ScalarCoreGain", "Scalar core score"),
        ("NonCompensatoryCore", "Non-compensatory core"),
    ]:
        selected = exposure[exposure[rule]]
        adaptive_selected = selected.dropna(subset=["AdaptiveAttackDelta"])
        rows.append(
            {
                "Rule": label,
                "SelectedCases": len(selected),
                "HiddenRegressions": int(selected["HiddenRegression"].sum()),
                "HiddenJointGainMissing": int(
                    selected["HiddenJointGainMissing"].sum()
                ),
                "AdaptiveCases": len(adaptive_selected),
                "AdaptiveRegressions": int(
                    adaptive_selected["AdaptiveRegression"].sum()
                ),
                "AdaptiveGainMissing": int(
                    adaptive_selected["AdaptiveGainMissing"].sum()
                ),
                "TotalCases": len(exposure),
            }
        )
    return exposure, pd.DataFrame(rows)


def fmt_delta(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}"


def fmt_percent(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}%}".replace("%", "\\%")


def display_dataset(value: str) -> str:
    return DISPLAY_DATASETS.get(value, value.replace("_", "\\_"))


def display_variant(value: str) -> str:
    return DISPLAY_VARIANTS.get(value, value.replace("_", "\\_"))


def write_main_audit_table(
    fixed: pd.DataFrame,
    selected: pd.DataFrame,
    report: pd.DataFrame,
    output_dir: Path,
) -> None:
    table = selected.merge(
        fixed[["Dataset", "Variant", "ModelsPassing", "ModelsTotal"]],
        on=["Dataset", "Variant"],
    ).merge(
        report[
            [
                "Dataset",
                "Variant",
                "UniquenessDelta",
                "NearestNeighborRiskDelta",
                "SensitivePredictabilityDelta",
            ]
        ],
        on=["Dataset", "Variant"],
    )
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\resizebox{0.98\\textwidth}{!}{%",
        "\\begin{tabular}{llcrrrrrr}",
        "\\toprule",
        "Dataset & Proxy & Fixed learners & $\\Delta\\mathrm{AUC}_{\\pi}$ & $\\Delta\\mathrm{ECE}_{\\pi}$ & $\\Delta\\mathrm{Cost}_{\\pi}$ & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{display_dataset(row['Dataset'])} & {display_variant(row['Variant'])} & "
            f"{int(row['ModelsPassing'])}/{int(row['ModelsTotal'])} & "
            f"{fmt_delta(row['AUCDelta'])} & {fmt_delta(row['ECEDelta'])} & "
            f"{fmt_delta(row['CostDelta'])} & {fmt_delta(row['UniquenessDelta'])} & "
            f"{fmt_delta(row['NearestNeighborRiskDelta'])} & "
            f"{fmt_delta(row['SensitivePredictabilityDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Paired proxy audit under two utility estimands. Fixed pass counts the six learners that meet every mean utility tolerance. Procedure $\\pi$ selects by validation Cost5x on each split and evaluates on the test fold. Exposure deltas are means over five seeds. All deltas are relative to the original table; lower ECE, cost, and exposure values are better.}\\label{tab:ruap_report}",
            "\\end{table}",
        ]
    )
    (output_dir / "audit_estimands_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_state_table(states: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.04}",
        "\\resizebox{0.9\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Proxy & Requirements met & Conflicting exposure & Decision-utility failure & Ranking-utility failure \\\\",
        "\\midrule",
    ]
    for _, row in states.iterrows():
        lines.append(
            f"{display_dataset(row['Dataset'])} & {display_variant(row['Variant'])} & "
            f"{fmt_percent(row['RequirementsMet'])} & "
            f"{fmt_percent(row['ConflictingExposure'])} & "
            f"{fmt_percent(row['DecisionUtilityFailure'])} & "
            f"{fmt_percent(row['RankingUtilityFailure'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Decision-state frequencies from the cross-product of 20 utility splits and five exposure seeds for the validation-selected procedure $\\pi$. Exposure seeds repeat transformation randomness, sampling, and cross-validation in the core probe screen. These descriptive frequencies are not posterior probabilities.}\\label{tab:state_stability_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "decision_state_stability.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_exposure_seed_table(
    exposure_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Proxy & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak \\\\",
        "\\midrule",
    ]
    for _, row in exposure_summary.iterrows():
        lines.append(
            f"{display_dataset(row['Dataset'])} & "
            f"{display_variant(row['Variant'])} & "
            f"{fmt_delta(row['UniquenessDelta'])} "
            f"$\\pm$ {row['UniquenessDeltaSD']:.3f} & "
            f"{fmt_delta(row['NearestNeighborRiskDelta'])} "
            f"$\\pm$ {row['NearestNeighborRiskDeltaSD']:.3f} & "
            f"{fmt_delta(row['SensitivePredictabilityDelta'])} "
            f"$\\pm$ {row['SensitivePredictabilityDeltaSD']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            (
                "\\caption{Mean $\\pm$ standard deviation of paired core exposure "
                "deltas over five seeds. Seeds repeat transformation randomness, "
                "row sampling, and cross-validation where applicable. "
                "Deterministic transforms can still vary through sampling or "
                "probe fitting.}\\label{tab:exposure_seed_uncertainty}"
            ),
            "\\end{table}",
        ]
    )
    (output_dir / "exposure_seed_uncertainty.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_omission_table(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Rule & Cases retained & Held-out failures & Failure rate \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Rule']} & {int(row['SelectedCases'])}/{int(row['TotalCases'])} & "
            f"{int(row['HeldOutFailures'])} & "
            f"{fmt_percent(row['HeldOutFailureRate'], digits=1)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Default-threshold omission test over 160 dataset--transformation--split cases. Rules see AUC, ECE$_{10}$, and Cost5x. A failure means that held-out AUPRC, Brier, another ECE estimate, Cost10x, or Cost20x crosses its corresponding tolerance.}\\label{tab:omission_default}",
            "\\end{table}",
        ]
    )
    (output_dir / "omission_stress_test.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_matched_retention_table(
    matched: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Ordering & Retained & Hidden failures & Hidden-safe & Failure rate \\\\",
        "\\midrule",
    ]
    for _, row in matched.iterrows():
        lines.append(
            f"{row['Rule']} & {int(row['SelectedCases'])} & "
            f"{int(row['HeldOutFailures'])} & "
            f"{int(row['HiddenSafeRetained'])} & "
            f"{fmt_percent(row['HeldOutFailureRate'], digits=1)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Matched-retention omission test. Each ordering retains 39 of 160 cases. The Pareto ordering ranks members of the default AUC-feasible non-dominated set first and uses AUC margin to break ties.}\\label{tab:omission_matched}",
            "\\end{table}",
        ]
    )
    (output_dir / "omission_matched_retention.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_family_failure_table(
    family: pd.DataFrame,
    retained_cases: int,
    output_dir: Path,
) -> None:
    table = family[family["Comparison"] == f"Matched {retained_cases}"]
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Ordering & AUPRC & Brier & Calibration & Cost & Any \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Rule']} & {int(row['RankingFailures'])} & "
            f"{int(row['ProbabilityFailures'])} & "
            f"{int(row['CalibrationFailures'])} & "
            f"{int(row['CostFailures'])} & {int(row['AnyFailures'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            f"\\caption{{Held-out failures by metric family at matched retention ({retained_cases} cases per ordering). A case can fail more than one family. The calibration column combines three ECE estimates; cost combines Cost10x and Cost20x.}}\\label{{tab:omission_families}}",
            "\\end{table}",
        ]
    )
    (output_dir / "omission_family_failures.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_leave_one_family_out_table(
    family_out: pd.DataFrame,
    output_dir: Path,
) -> None:
    scalar = family_out[family_out["Rule"] == "Scalar mean"].set_index(
        "HiddenFamily"
    )
    non_compensatory = family_out[
        family_out["Rule"] == "Non-compensatory gates"
    ].set_index("HiddenFamily")
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Hidden family & Visible representatives & Scalar & Non-compensatory \\\\",
        "\\midrule",
    ]
    for hidden_family in FULL_METRIC_FAMILIES:
        scalar_row = scalar.loc[hidden_family]
        non_compensatory_row = non_compensatory.loc[hidden_family]
        visible = str(scalar_row["VisibleMetrics"]).replace("+", ", ")
        lines.append(
            f"{hidden_family} & {visible} & "
            f"{int(scalar_row['HiddenFamilyFailures'])}/"
            f"{int(scalar_row['SelectedCases'])} & "
            f"{int(non_compensatory_row['HiddenFamilyFailures'])}/"
            f"{int(non_compensatory_row['SelectedCases'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            (
                "\\caption{Leave-one-family-out test at 39 retained cases. The "
                "ordering sees the listed representative metrics and is evaluated "
                "against every measurement in the hidden family. Probability "
                "quality contains Brier score; it has no visible representative "
                "in the original three-metric rule.}\\label{tab:omission_family_out}"
            ),
            "\\end{table}",
        ]
    )
    (output_dir / "omission_leave_one_family_out.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_tolerance_sensitivity_table(
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        (
            "Visible tolerance varied & Scalar wins at 39 & Ties & "
            "Non-comp. wins & Scalar lower AURC \\\\"
        ),
        "\\midrule",
    ]
    for _, row in sensitivity.iterrows():
        lines.append(
            f"{row['VariedVisibleTolerance']} & "
            f"{int(row['ScalarMatchedWins'])}/{int(row['Settings'])} & "
            f"{int(row['TiesAtMatchedRetention'])}/{int(row['Settings'])} & "
            f"{int(row['NonCompensatoryMatchedWins'])}/{int(row['Settings'])} & "
            f"{int(row['ScalarAURCWins'])}/{int(row['Settings'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            (
                "\\caption{Tolerance sensitivity over 25 settings per row. One "
                "visible tolerance and all seven hidden-check tolerances vary over "
                "$\\{0.5,0.75,1,1.5,2\\}$ times their reference values. Wins denote "
                "strictly lower hidden-failure risk; no rule dominates across the "
                "grid.}\\label{tab:omission_tolerance_sensitivity}"
            ),
            "\\end{table}",
        ]
    )
    (output_dir / "omission_tolerance_sensitivity.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_ece_only_table(
    ece_only: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Ordering & Alt. ECE, no Brier/cost & Alt. ECE only \\\\",
        "\\midrule",
    ]
    for _, row in ece_only.iterrows():
        lines.append(
            f"{row['Rule']} & "
            f"{int(row['AlternativeECEWithoutBrierOrCost'])} & "
            f"{int(row['AlternativeECEOnly'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            (
                "\\caption{Calibration-only diagnoses at matched retention. The "
                "first column counts failures of an alternative ECE estimate "
                "without a Brier or higher-cost failure; the second also excludes "
                "AUPRC failure.}\\label{tab:omission_ece_only}"
            ),
            "\\end{table}",
        ]
    )
    (output_dir / "omission_ece_only.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_intervention_robustness_table(
    macro: pd.DataFrame,
    robustness: pd.DataFrame,
    output_dir: Path,
) -> None:
    intervention = robustness[robustness["OmissionType"] == "Intervention"][
        ["Rule", "MinFailureRate", "MaxFailureRate"]
    ].rename(
        columns={
            "MinFailureRate": "InterventionMin",
            "MaxFailureRate": "InterventionMax",
        }
    )
    dataset = robustness[robustness["OmissionType"] == "Dataset"][
        ["Rule", "MinFailureRate", "MaxFailureRate"]
    ].rename(
        columns={
            "MinFailureRate": "DatasetMin",
            "MaxFailureRate": "DatasetMax",
        }
    )
    table = macro.merge(intervention, on="Rule").merge(dataset, on="Rule")
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Ordering & Groups & Macro risk & Leave-one group & Leave-one dataset \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Rule']} & "
            f"{int(row['InterventionsRepresented'])}/{int(row['InterventionsTotal'])} & "
            f"{fmt_percent(row['MacroFailureRate'], digits=1)} & "
            f"{fmt_percent(row['InterventionMin'], digits=1)}--"
            f"{fmt_percent(row['InterventionMax'], digits=1)} & "
            f"{fmt_percent(row['DatasetMin'], digits=1)}--"
            f"{fmt_percent(row['DatasetMax'], digits=1)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Intervention-level omission checks. Group-macro risk gives each of the eight dataset--transformation interventions with retained cases equal weight under the default rules. Leave-one-out ranges use matched 24.4\\% coverage and recompute the ordering after dropping one intervention or dataset.}\\label{tab:omission_robustness}",
            "\\end{table}",
        ]
    )
    (output_dir / "omission_intervention_robustness.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_exposure_omission_table(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Visible rule & Retained & Link/member regress. & No joint gain & Learned no gain \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Rule']} & {int(row['SelectedCases'])} & "
            f"{int(row['HiddenRegressions'])} & "
            f"{int(row['HiddenJointGainMissing'])} & "
            f"{int(row['AdaptiveGainMissing'])}/{int(row['AdaptiveCases'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Exposure-probe omission check over 90 transformed tables. Visible rules use uniqueness, nearest-neighbor risk, and sensitive-attribute predictability; AuxLink and MemAUC are then revealed. Learned-attack entries give misses over evaluated cases. ``No gain'' means a held-out delta is not below $-0.01$. These probes are not privacy ground truth.}\\label{tab:exposure_omission}",
            "\\end{table}",
        ]
    )
    (output_dir / "exposure_omission_test.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_risk_coverage_plot(
    curve: pd.DataFrame,
    matched: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {
        "AUC gate": "#4C78A8",
        "Scalar mean": "#F58518",
        "Pareto + AUC tie-break": "#54A24B",
        "Non-compensatory gates": "#B04A5A",
    }
    styles = {
        "AUC gate": "-",
        "Scalar mean": "--",
        "Pareto + AUC tie-break": "-.",
        "Non-compensatory gates": ":",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for label, group in curve.groupby("Rule", sort=False):
        ax.plot(
            group["Coverage"],
            group["HeldOutFailureRate"],
            label=label,
            color=colors[label],
            linestyle=styles[label],
            linewidth=1.8,
        )
    matched_coverage = float(matched["SelectedCases"].iloc[0] / matched["TotalCases"].iloc[0])
    ax.axvline(matched_coverage, color="#777777", linewidth=1.0, alpha=0.8)
    for _, row in matched.iterrows():
        ax.scatter(
            matched_coverage,
            row["HeldOutFailureRate"],
            s=30,
            color=colors[row["Rule"]],
            zorder=3,
        )
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Held-out failure rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_tolerance_sensitivity_plot(
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "AUC": "AUC tolerance",
        "ECE": "ECE tolerance",
        "Cost": "Cost tolerance",
    }
    max_abs = max(
        0.01,
        float(sensitivity["MatchedRiskDifference"].abs().max()),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    image = None
    for ax, metric in zip(axes, VISIBLE_TOLERANCES):
        group = sensitivity[
            sensitivity["VariedVisibleTolerance"] == metric
        ]
        matrix = group.pivot(
            index="HiddenScale",
            columns="VisibleScale",
            values="MatchedRiskDifference",
        ).sort_index(ascending=False)
        image = ax.imshow(
            matrix.to_numpy() * 100,
            cmap="PuOr",
            vmin=-max_abs * 100,
            vmax=max_abs * 100,
            aspect="auto",
        )
        ax.set_title(labels[metric], fontsize=10)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(
            [f"{value:g}" for value in matrix.columns],
            fontsize=8,
        )
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(
            [f"{value:g}" for value in matrix.index],
            fontsize=8,
        )
        ax.set_xlabel("visible scale", fontsize=8.5)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, column_index] * 100
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#202020",
                )
    axes[0].set_ylabel("hidden-check scale", fontsize=8.5)
    if image is not None:
        colorbar_axis = fig.add_axes([0.875, 0.20, 0.018, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            "Non-comp. minus scalar risk (points)",
            fontsize=8,
        )
        colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(
        left=0.07,
        right=0.84,
        bottom=0.18,
        top=0.86,
        wspace=0.16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-model, validation-selection, stability, and omission-test assets."
    )
    parser.add_argument(
        "--proxy-root",
        default="outputs/proxy_transform_audit",
    )
    parser.add_argument(
        "--report-csv",
        default="paper_assets/ruap_audit/ruap_report_card.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_assets/ruap_audit",
    )
    parser.add_argument(
        "--extended-exposure-csv",
        default="paper_assets/ruap_audit/extended_exposure_stress.csv",
    )
    parser.add_argument(
        "--adaptive-attack-csv",
        default="paper_assets/ruap_audit/adaptive_attack_screen.csv",
    )
    parser.add_argument(
        "--risk-coverage-figure",
        default="paper/figs/omission_risk_coverage.png",
    )
    parser.add_argument(
        "--tolerance-sensitivity-figure",
        default="paper/figs/omission_tolerance_sensitivity.png",
    )
    parser.add_argument(
        "--exposure-seeds",
        default="3407,3408,3409,3410,3411",
        help="Comma-separated seeds for repeated core exposure probes.",
    )
    parser.add_argument(
        "--exposure-max-rows",
        type=int,
        default=5000,
    )
    args = parser.parse_args()

    root = Path(args.proxy_root)
    report = pd.read_csv(args.report_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed, selected, utility_only_states = fixed_model_summary(root, report)
    exposure_seeds = [
        int(value.strip())
        for value in args.exposure_seeds.split(",")
        if value.strip()
    ]
    exposure_seed_detail, exposure_seed_summary = repeated_core_exposure(
        exposure_seeds,
        max_rows=args.exposure_max_rows,
    )
    states = joint_state_stability(root, exposure_seed_detail)
    repeated_report = report_with_repeated_exposure(
        report,
        exposure_seed_summary,
    )
    detail, omission = omission_stress_test(root)
    retained_cases = int(detail["NonCompensatory"].sum())
    matched = matched_retention_summary(detail, retained_cases)
    risk_coverage, aurc = risk_coverage_curve(detail)
    family = family_failure_summary(detail, retained_cases)
    family_out = leave_one_family_out(detail, retained_cases)
    ece_only = ece_only_failure_summary(detail, retained_cases)
    tolerance_detail, tolerance_summary = omission_tolerance_sensitivity(
        detail,
        retained_cases,
    )
    group_detail, group_macro, robustness = intervention_robustness(
        detail,
        retained_cases,
    )
    exposure_detail, exposure_summary = exposure_omission_test(
        Path(args.extended_exposure_csv),
        Path(args.adaptive_attack_csv),
    )

    fixed.to_csv(output_dir / "fixed_model_audit.csv", index=False)
    selected.to_csv(output_dir / "validation_selected_audit.csv", index=False)
    utility_only_states.to_csv(
        output_dir / "decision_state_utility_only.csv",
        index=False,
    )
    states.to_csv(output_dir / "decision_state_stability.csv", index=False)
    exposure_seed_detail.to_csv(
        output_dir / "exposure_seed_detail.csv",
        index=False,
    )
    exposure_seed_summary.to_csv(
        output_dir / "exposure_seed_summary.csv",
        index=False,
    )
    detail.to_csv(output_dir / "omission_stress_detail.csv", index=False)
    omission.to_csv(output_dir / "omission_stress_summary.csv", index=False)
    matched.to_csv(output_dir / "omission_matched_retention.csv", index=False)
    risk_coverage.to_csv(output_dir / "omission_risk_coverage.csv", index=False)
    aurc.to_csv(output_dir / "omission_aurc.csv", index=False)
    family.to_csv(output_dir / "omission_family_failures.csv", index=False)
    family_out.to_csv(
        output_dir / "omission_leave_one_family_out.csv",
        index=False,
    )
    ece_only.to_csv(
        output_dir / "omission_ece_only.csv",
        index=False,
    )
    tolerance_detail.to_csv(
        output_dir / "omission_tolerance_sensitivity.csv",
        index=False,
    )
    tolerance_summary.to_csv(
        output_dir / "omission_tolerance_sensitivity_summary.csv",
        index=False,
    )
    group_detail.to_csv(
        output_dir / "omission_intervention_detail.csv",
        index=False,
    )
    group_macro.to_csv(
        output_dir / "omission_intervention_macro.csv",
        index=False,
    )
    robustness.to_csv(
        output_dir / "omission_leave_one_out.csv",
        index=False,
    )
    exposure_detail.to_csv(
        output_dir / "exposure_omission_detail.csv",
        index=False,
    )
    exposure_summary.to_csv(
        output_dir / "exposure_omission_summary.csv",
        index=False,
    )

    write_main_audit_table(fixed, selected, repeated_report, output_dir)
    write_state_table(states, output_dir)
    write_exposure_seed_table(exposure_seed_summary, output_dir)
    write_omission_table(omission, output_dir)
    write_matched_retention_table(matched, output_dir)
    write_family_failure_table(family, retained_cases, output_dir)
    write_leave_one_family_out_table(family_out, output_dir)
    write_ece_only_table(ece_only, output_dir)
    write_tolerance_sensitivity_table(tolerance_summary, output_dir)
    write_intervention_robustness_table(group_macro, robustness, output_dir)
    write_exposure_omission_table(exposure_summary, output_dir)
    write_risk_coverage_plot(
        risk_coverage,
        matched,
        Path(args.risk_coverage_figure),
    )
    write_tolerance_sensitivity_plot(
        tolerance_detail,
        Path(args.tolerance_sensitivity_figure),
    )
    print(
        f"wrote {len(fixed)} audit rows, {len(states)} stability rows, "
        f"{len(detail)} utility omission cases, and "
        f"{len(exposure_detail)} exposure omission cases to {output_dir}"
    )


if __name__ == "__main__":
    main()
