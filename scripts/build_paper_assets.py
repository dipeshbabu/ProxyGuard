from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PAPER_MODELS = [
    "logreg_baseline",
    "xgb_baseline",
    "lightgbm_baseline",
    "catboost_baseline",
    "histgb_baseline",
    "compact_xgb",
    "tabpfn_baseline",
    "tabicl_baseline",
    "reliability_ensemble",
    "rc_stack",
    "rc_stack_dr",
    "rrc_stack",
]

DISPLAY_MODELS = {
    "logreg_baseline": "LogReg",
    "xgb_baseline": "XGB",
    "lightgbm_baseline": "LightGBM",
    "catboost_baseline": "CatBoost",
    "histgb_baseline": "HistGB",
    "compact_xgb": "Compact XGB",
    "tabpfn_baseline": "TabPFN",
    "tabicl_baseline": "TabICL",
    "reliability_ensemble": "RCE",
    "rc_stack": "RC-Stack",
    "rc_stack_dr": "RC-Stack-DR",
    "rrc_stack": "RRC-Stack",
}

DISPLAY_DATASETS = {
    "adult_income": "Adult",
    "australian_credit": "Australian",
    "bank_marketing": "Bank",
    "breast_cancer_wdbc": "WDBC",
    "compas_recidivism": "COMPAS",
    "german_credit": "German",
    "give_me_some_credit": "GMSC",
    "heart_disease": "Heart",
    "mammographic_mass": "Mammography",
    "taiwan_default": "Taiwan",
}


class LatexRaw(str):
    pass


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_aggregate_tables(output_root: Path, mode: str) -> pd.DataFrame:
    rows = []
    mode_dir = output_root / mode
    if not mode_dir.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        aggregate = read_csv_if_exists(dataset_dir / "aggregate_metrics.csv")
        if aggregate.empty:
            continue
        aggregate.insert(0, "Dataset", dataset_dir.name)
        aggregate.insert(1, "Run", mode)
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_split_tables(output_root: Path, mode: str) -> pd.DataFrame:
    rows = []
    mode_dir = output_root / mode
    if not mode_dir.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        split_metrics = read_csv_if_exists(dataset_dir / "split_metrics.csv")
        if split_metrics.empty:
            continue
        split_metrics.insert(0, "Dataset", dataset_dir.name)
        split_metrics.insert(1, "Run", mode)
        rows.append(split_metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_subgroup_tables(output_root: Path, mode: str) -> pd.DataFrame:
    rows = []
    mode_dir = output_root / mode
    if not mode_dir.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        subgroup_metrics = read_csv_if_exists(dataset_dir / "subgroup_metrics.csv")
        if subgroup_metrics.empty:
            continue
        rows.append(subgroup_metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_legacy_aggregate_tables(dataset_root: Path, run_label: str) -> pd.DataFrame:
    rows = []
    if not dataset_root.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        aggregate = read_csv_if_exists(dataset_dir / "aggregate_metrics.csv")
        if aggregate.empty:
            continue
        aggregate.insert(0, "Dataset", dataset_dir.name)
        aggregate.insert(1, "Run", run_label)
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_legacy_split_tables(dataset_root: Path, run_label: str) -> pd.DataFrame:
    rows = []
    if not dataset_root.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        split_metrics = read_csv_if_exists(dataset_dir / "split_metrics.csv")
        if split_metrics.empty:
            continue
        split_metrics.insert(0, "Dataset", dataset_dir.name)
        split_metrics.insert(1, "Run", run_label)
        rows.append(split_metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prefer_primary(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return fallback
    if fallback.empty:
        return primary
    key = ["Dataset", "Model"]
    primary_keys = set(map(tuple, primary[key].to_numpy()))
    fallback_keep = fallback[~fallback[key].apply(tuple, axis=1).isin(primary_keys)]
    return pd.concat([primary, fallback_keep], ignore_index=True)


def prefer_primary_splits(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return fallback
    if fallback.empty:
        return primary
    key = ["Dataset", "Model", "split_seed"]
    primary_keys = set(map(tuple, primary[key].to_numpy()))
    fallback_keep = fallback[~fallback[key].apply(tuple, axis=1).isin(primary_keys)]
    return pd.concat([primary, fallback_keep], ignore_index=True)


def build_summary_table(calibrated: pd.DataFrame) -> pd.DataFrame:
    if calibrated.empty:
        return pd.DataFrame()
    cols = [
        "Dataset",
        "Model",
        "AUC",
        "AUC_ci95",
        "AUPRC",
        "AUPRC_ci95",
        "Brier",
        "Brier_ci95",
        "ECE (10-bin)",
        "ECE (10-bin)_ci95",
        "ECE (15-bin)",
        "ECE (15-bin)_ci95",
        "ECE (20-bin)",
        "ECE (20-bin)_ci95",
        "AdaptiveECE (10-bin)",
        "AdaptiveECE (10-bin)_ci95",
        "LogLoss",
        "LogLoss_ci95",
        "CalibrationSlope",
        "CalibrationSlope_ci95",
        "DecisionCost2x",
        "DecisionCost2x_ci95",
        "DecisionCost2xRelApproveAll",
        "DecisionCost2xRelApproveAll_ci95",
        "DecisionCost5x",
        "DecisionCost5x_ci95",
        "DecisionCost5xRelApproveAll",
        "DecisionCost5xRelApproveAll_ci95",
        "DecisionCost10x",
        "DecisionCost10x_ci95",
        "DecisionCost10xRelApproveAll",
        "DecisionCost10xRelApproveAll_ci95",
        "DecisionCost20x",
        "DecisionCost20x_ci95",
        "DecisionCost20xRelApproveAll",
        "DecisionCost20xRelApproveAll_ci95",
        "FeatureCount",
        "PreprocessTimeSec",
        "ModelFitTimeSec",
        "CalibrationTimeSec",
        "TrainTimeSec",
        "InferenceTimeSec",
        "PeakMemoryMB",
        "n_splits",
    ]
    present = [col for col in cols if col in calibrated.columns]
    summary = calibrated.loc[calibrated["Model"].isin(PAPER_MODELS), present].copy()
    return summary.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def build_calibration_delta(calibrated: pd.DataFrame, uncalibrated: pd.DataFrame) -> pd.DataFrame:
    if calibrated.empty or uncalibrated.empty:
        return pd.DataFrame()
    metrics = [
        "AUC",
        "Brier",
        "ECE (10-bin)",
        "ECE (15-bin)",
        "ECE (20-bin)",
        "AdaptiveECE (10-bin)",
        "LogLoss",
        "CalibrationSlope",
    ]
    left_source = calibrated[calibrated["Model"].isin(PAPER_MODELS)].copy()
    right_source = uncalibrated[uncalibrated["Model"].isin(PAPER_MODELS)].copy()
    left = left_source[["Dataset", "Model"] + [col for col in metrics if col in left_source.columns]].copy()
    right = right_source[["Dataset", "Model"] + [col for col in metrics if col in right_source.columns]].copy()
    merged = left.merge(right, on=["Dataset", "Model"], suffixes=("_calibrated", "_uncalibrated"))
    for metric in metrics:
        cal_col = f"{metric}_calibrated"
        raw_col = f"{metric}_uncalibrated"
        if cal_col in merged.columns and raw_col in merged.columns:
            merged[f"{metric}_delta_cal_minus_raw"] = merged[cal_col] - merged[raw_col]
    return merged.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def collect_weak_label_summary(output_root: Path) -> pd.DataFrame:
    collected = []
    for source_order, root in enumerate([output_root / "weak_label", Path("outputs") / "weak_label"]):
        if not root.exists():
            continue
        for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for dataset_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
                aggregate = read_csv_if_exists(dataset_dir / "aggregate_metrics.csv")
                if aggregate.empty:
                    continue
                aggregate.insert(0, "Variant", variant_dir.name)
                aggregate.insert(0, "Dataset", dataset_dir.name)
                aggregate["_source_order"] = source_order
                collected.append(aggregate)
    weak_summary = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    if weak_summary.empty:
        weak_summary = read_csv_if_exists(output_root / "weak_label" / "weak_label_summary.csv")
    if weak_summary.empty:
        weak_summary = read_csv_if_exists(Path("outputs") / "weak_label" / "weak_label_summary.csv")
    if weak_summary.empty:
        return weak_summary
    if "Dataset" not in weak_summary.columns:
        weak_summary.insert(0, "Dataset", "german_credit")
    keep_cols = [
        "Dataset",
        "Variant",
        "Model",
        "AUC",
        "AUPRC",
        "Brier",
        "ECE (10-bin)",
        "LogLoss",
        "CalibrationSlope",
        "FeatureCount",
        "n_splits",
    ]
    present = [col for col in keep_cols if col in weak_summary.columns]
    filtered = weak_summary.loc[weak_summary["Model"].isin(PAPER_MODELS)].copy()
    if "_source_order" in filtered.columns:
        filtered = filtered.sort_values("_source_order")
        filtered = filtered.drop_duplicates(["Dataset", "Variant", "Model"], keep="first")
    else:
        filtered = filtered.drop_duplicates(["Dataset", "Variant", "Model"], keep="first")
    return filtered[present].copy()


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def fmt_pm(mean_value: object, ci_value: object, digits: int = 3) -> str:
    mean_text = fmt_float(mean_value, digits=digits)
    ci_text = fmt_float(ci_value, digits=digits)
    if mean_text == "--" or ci_text == "--":
        return mean_text
    return LatexRaw(rf"{mean_text} $\pm$ {ci_text}")


def write_latex_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    column_spec = "ll" + "r" * max(0, len(headers) - 2)
    lines = [
        r"\begin{tabular}{" + column_spec + r"}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [str(cell) if isinstance(cell, LatexRaw) else latex_escape(cell) for cell in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_results_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = PAPER_MODELS
    table = summary[summary["Model"].isin(key_models)].copy()
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("AUC")),
                fmt_float(row.get("AUPRC")),
                fmt_float(row.get("Brier")),
                fmt_float(row.get("ECE (10-bin)")),
                fmt_float(row.get("CalibrationSlope")),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(path, ["Dataset", "Model", "AUC", "AUPRC", "Brier", "ECE", "Slope", "Splits"], rows)


def write_variability_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = PAPER_MODELS
    table = summary[summary["Model"].isin(key_models)].copy()
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_pm(row.get("AUC"), row.get("AUC_ci95")),
                fmt_pm(row.get("Brier"), row.get("Brier_ci95")),
                fmt_pm(row.get("ECE (10-bin)"), row.get("ECE (10-bin)_ci95")),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(path, ["Dataset", "Model", "AUC", "Brier", "ECE", "Splits"], rows)


def _format_win_counts(left_wins: int, right_wins: int, ties: int, left_label: str, right_label: str) -> str:
    if ties:
        return f"{left_wins}/{right_wins}/{ties}"
    return f"{left_wins}/{right_wins}"


def build_paired_win_counts(split_metrics: pd.DataFrame) -> pd.DataFrame:
    if split_metrics.empty or "split_seed" not in split_metrics.columns:
        return pd.DataFrame()

    comparisons = [
        ("xgb_baseline", "tabpfn_baseline"),
        ("xgb_baseline", "tabicl_baseline"),
        ("histgb_baseline", "rc_stack"),
        ("histgb_baseline", "rrc_stack"),
        ("xgb_baseline", "reliability_ensemble"),
        ("xgb_baseline", "rc_stack"),
        ("compact_xgb", "reliability_ensemble"),
        ("compact_xgb", "rc_stack"),
        ("reliability_ensemble", "tabpfn_baseline"),
        ("reliability_ensemble", "tabicl_baseline"),
        ("reliability_ensemble", "rc_stack"),
        ("rc_stack", "rc_stack_dr"),
        ("rc_stack", "rrc_stack"),
        ("rc_stack_dr", "rrc_stack"),
        ("xgb_baseline", "rc_stack_dr"),
        ("xgb_baseline", "rrc_stack"),
        ("tabicl_baseline", "rc_stack_dr"),
        ("tabicl_baseline", "rrc_stack"),
        ("compact_xgb", "tabpfn_baseline"),
        ("compact_xgb", "tabicl_baseline"),
        ("tabpfn_baseline", "rc_stack"),
        ("tabicl_baseline", "rc_stack"),
        ("xgb_baseline", "compact_xgb"),
        ("tabpfn_baseline", "tabicl_baseline"),
    ]
    rows = []
    for dataset_name, dataset_df in split_metrics.groupby("Dataset"):
        for left_model, right_model in comparisons:
            left = dataset_df[dataset_df["Model"] == left_model].set_index("split_seed")
            right = dataset_df[dataset_df["Model"] == right_model].set_index("split_seed")
            common_seeds = sorted(set(left.index) & set(right.index))
            if not common_seeds:
                continue
            left = left.loc[common_seeds]
            right = right.loc[common_seeds]

            auc_delta = left["AUC"].to_numpy(dtype=float) - right["AUC"].to_numpy(dtype=float)
            brier_delta = right["Brier"].to_numpy(dtype=float) - left["Brier"].to_numpy(dtype=float)
            ece_delta = right["ECE (10-bin)"].to_numpy(dtype=float) - left["ECE (10-bin)"].to_numpy(dtype=float)

            def counts(delta: pd.Series | list[float]):
                values = pd.Series(delta).round(12)
                return int((values > 0).sum()), int((values < 0).sum()), int((values == 0).sum())

            auc_left, auc_right, auc_tie = counts(auc_delta)
            brier_left, brier_right, brier_tie = counts(brier_delta)
            ece_left, ece_right, ece_tie = counts(ece_delta)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Comparison": f"{DISPLAY_MODELS[left_model]} vs {DISPLAY_MODELS[right_model]}",
                    "AUC wins": _format_win_counts(auc_left, auc_right, auc_tie, left_model, right_model),
                    "Brier wins": _format_win_counts(brier_left, brier_right, brier_tie, left_model, right_model),
                    "ECE wins": _format_win_counts(ece_left, ece_right, ece_tie, left_model, right_model),
                    "Common splits": len(common_seeds),
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        holm_value = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, holm_value)
        adjusted[original_index] = running_max
    return adjusted


def build_paired_stat_tests(split_metrics: pd.DataFrame) -> pd.DataFrame:
    if split_metrics.empty or "split_seed" not in split_metrics.columns:
        return pd.DataFrame()

    comparisons = [
        ("xgb_baseline", "tabpfn_baseline"),
        ("xgb_baseline", "tabicl_baseline"),
        ("histgb_baseline", "rc_stack"),
        ("histgb_baseline", "rrc_stack"),
        ("xgb_baseline", "reliability_ensemble"),
        ("xgb_baseline", "rc_stack"),
        ("compact_xgb", "reliability_ensemble"),
        ("compact_xgb", "rc_stack"),
        ("reliability_ensemble", "tabpfn_baseline"),
        ("reliability_ensemble", "tabicl_baseline"),
        ("reliability_ensemble", "rc_stack"),
        ("rc_stack", "rc_stack_dr"),
        ("rc_stack", "rrc_stack"),
        ("rc_stack_dr", "rrc_stack"),
        ("xgb_baseline", "rc_stack_dr"),
        ("xgb_baseline", "rrc_stack"),
        ("tabicl_baseline", "rc_stack_dr"),
        ("tabicl_baseline", "rrc_stack"),
        ("compact_xgb", "tabpfn_baseline"),
        ("compact_xgb", "tabicl_baseline"),
        ("tabpfn_baseline", "rc_stack"),
        ("tabicl_baseline", "rc_stack"),
        ("xgb_baseline", "compact_xgb"),
        ("tabpfn_baseline", "tabicl_baseline"),
    ]
    metrics = [
        ("AUC", True),
        ("Brier", False),
        ("ECE (10-bin)", False),
        ("DecisionCost5x", False),
        ("DecisionCost10x", False),
    ]
    rows = []
    for dataset_name, dataset_df in split_metrics.groupby("Dataset"):
        for left_model, right_model in comparisons:
            left = dataset_df[dataset_df["Model"] == left_model].set_index("split_seed")
            right = dataset_df[dataset_df["Model"] == right_model].set_index("split_seed")
            common_seeds = sorted(set(left.index) & set(right.index))
            if not common_seeds:
                continue
            left = left.loc[common_seeds]
            right = right.loc[common_seeds]
            for metric, higher_is_better in metrics:
                if metric not in left.columns or metric not in right.columns:
                    continue
                left_values = left[metric].to_numpy(dtype=float)
                right_values = right[metric].to_numpy(dtype=float)
                diff = left_values - right_values if higher_is_better else right_values - left_values
                diff = diff[pd.notna(diff)]
                if len(diff) < 2:
                    continue
                if (pd.Series(diff).round(12) == 0).all():
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Comparison": f"{DISPLAY_MODELS[left_model]} vs {DISPLAY_MODELS[right_model]}",
                        "Metric": metric,
                        "MeanDeltaLeftBetter": float(diff.mean()),
                        "MedianDeltaLeftBetter": float(pd.Series(diff).median()),
                        "p_value": p_value,
                        "Common splits": len(diff),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm"] = _holm_adjust(result["p_value"].fillna(1.0).astype(float).tolist())
    return result


def build_subgroup_reliability_summary(subgroup_metrics: pd.DataFrame, min_support: int = 10) -> pd.DataFrame:
    if subgroup_metrics.empty:
        return pd.DataFrame()
    required = {"Dataset", "Model", "split_seed", "SubgroupName", "SubgroupValue", "Support", "Brier", "ECE"}
    if not required.issubset(subgroup_metrics.columns):
        return pd.DataFrame()

    working = subgroup_metrics.copy()
    working = working[working["Model"].isin(PAPER_MODELS)]
    working = working[pd.to_numeric(working["Support"], errors="coerce").fillna(0) >= min_support]
    if working.empty:
        return pd.DataFrame()

    per_split_rows = []
    group_cols = ["Dataset", "Model", "split_seed", "SubgroupName"]
    for group_key, group in working.groupby(group_cols):
        if group["SubgroupValue"].nunique() < 2:
            continue
        dataset_name, model_name, split_seed, subgroup_name = group_key
        ece_values = pd.to_numeric(group["ECE"], errors="coerce").dropna()
        brier_values = pd.to_numeric(group["Brier"], errors="coerce").dropna()
        if ece_values.empty or brier_values.empty:
            continue
        per_split_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "split_seed": split_seed,
                "SubgroupName": subgroup_name,
                "WorstGroupECE": float(ece_values.max()),
                "ECESpread": float(ece_values.max() - ece_values.min()),
                "WorstGroupBrier": float(brier_values.max()),
                "BrierSpread": float(brier_values.max() - brier_values.min()),
                "MinSupport": int(pd.to_numeric(group["Support"], errors="coerce").min()),
                "n_groups": int(group["SubgroupValue"].nunique()),
            }
        )
    per_split = pd.DataFrame(per_split_rows)
    if per_split.empty:
        return per_split

    rows = []
    for (dataset_name, model_name), group in per_split.groupby(["Dataset", "Model"]):
        rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "WorstGroupECE": float(group["WorstGroupECE"].mean()),
                "ECESpread": float(group["ECESpread"].mean()),
                "WorstGroupBrier": float(group["WorstGroupBrier"].mean()),
                "BrierSpread": float(group["BrierSpread"].mean()),
                "MinSupport": int(group["MinSupport"].min()),
                "SubgroupAudits": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["Dataset", "Model"]).reset_index(drop=True)


def write_paired_win_counts_latex(win_counts: pd.DataFrame, path: Path) -> None:
    if win_counts.empty:
        write_latex_table(path, ["Dataset", "Comparison", "AUC wins", "Brier wins", "ECE wins", "Splits"], [])
        return
    rows = []
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table = win_counts.copy()
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table = table.sort_values(["dataset_order", "Comparison"])
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                row["Comparison"],
                row["AUC wins"],
                row["Brier wins"],
                row["ECE wins"],
                row["Common splits"],
            ]
        )
    write_latex_table(path, ["Dataset", "Comparison", "AUC wins", "Brier wins", "ECE wins", "Splits"], rows)


def write_paired_stat_tests_latex(stat_tests: pd.DataFrame, path: Path) -> None:
    if stat_tests.empty:
        write_latex_table(path, ["Dataset", "Comparison", "Metric", "Mean delta", "p", "Holm p", "Splits"], [])
        return
    key_metrics = {"AUC", "Brier", "ECE (10-bin)", "DecisionCost5x"}
    key_comparisons = {
        "XGB vs RCE",
        "XGB vs RC-Stack",
        "Compact XGB vs RCE",
        "Compact XGB vs RC-Stack",
        "RCE vs TabPFN",
        "RCE vs TabICL",
        "RCE vs RC-Stack",
        "RC-Stack vs RC-Stack-DR",
        "XGB vs RC-Stack-DR",
        "TabICL vs RC-Stack-DR",
        "TabPFN vs RC-Stack",
        "TabICL vs RC-Stack",
        "XGB vs TabPFN",
        "XGB vs TabICL",
    }
    table = stat_tests[
        stat_tests["Metric"].isin(key_metrics)
        & stat_tests["Comparison"].isin(key_comparisons)
    ].copy()
    if table.empty:
        table = stat_tests.copy()
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    metric_order = {"AUC": 0, "Brier": 1, "ECE (10-bin)": 2, "DecisionCost5x": 3, "DecisionCost10x": 4}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["metric_order"] = table["Metric"].map(metric_order).fillna(999)
    table = table.sort_values(["dataset_order", "Comparison", "metric_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                row["Comparison"],
                row["Metric"].replace(" (10-bin)", ""),
                fmt_float(row.get("MeanDeltaLeftBetter"), digits=4),
                fmt_float(row.get("p_value"), digits=4),
                fmt_float(row.get("p_holm"), digits=4),
                str(int(row.get("Common splits", 0))),
            ]
        )
    write_latex_table(path, ["Dataset", "Comparison", "Metric", "Mean delta", "p", "Holm p", "Splits"], rows)


def write_decision_cost_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = ["xgb_baseline", "compact_xgb", "tabpfn_baseline", "tabicl_baseline", "reliability_ensemble", "rc_stack", "rc_stack_dr"]
    table = summary[summary["Model"].isin(key_models)].copy()
    if table.empty:
        write_latex_table(path, ["Dataset", "Model", "AUC", "ECE", "Cost5x", "Cost10x", "Splits"], [])
        return
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("AUC")),
                fmt_float(row.get("ECE (10-bin)")),
                fmt_float(row.get("DecisionCost5x")),
                fmt_float(row.get("DecisionCost10x")),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(path, ["Dataset", "Model", "AUC", "ECE", "Cost5x", "Cost10x", "Splits"], rows)


def write_subgroup_reliability_latex(subgroup_summary: pd.DataFrame, path: Path) -> None:
    key_models = ["xgb_baseline", "compact_xgb", "tabpfn_baseline", "tabicl_baseline", "reliability_ensemble", "rc_stack", "rc_stack_dr"]
    table = subgroup_summary[subgroup_summary["Model"].isin(key_models)].copy()
    if table.empty:
        write_latex_table(
            path,
            ["Dataset", "Model", "Worst ECE", "ECE spread", "Worst Brier", "Brier spread", "Audits"],
            [],
        )
        return
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("WorstGroupECE")),
                fmt_float(row.get("ECESpread")),
                fmt_float(row.get("WorstGroupBrier")),
                fmt_float(row.get("BrierSpread")),
                str(int(row.get("SubgroupAudits", 0))),
            ]
        )
    write_latex_table(
        path,
        ["Dataset", "Model", "Worst ECE", "ECE spread", "Worst Brier", "Brier spread", "Audits"],
        rows,
    )


def write_efficiency_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = PAPER_MODELS
    table = summary[summary["Model"].isin(key_models)].copy()
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("PreprocessTimeSec"), digits=3),
                fmt_float(row.get("ModelFitTimeSec"), digits=3),
                fmt_float(row.get("CalibrationTimeSec"), digits=3),
                fmt_float(row.get("TrainTimeSec"), digits=3),
                fmt_float(row.get("InferenceTimeSec"), digits=3),
                fmt_float(row.get("PeakMemoryMB"), digits=1),
                fmt_float(row.get("FeatureCount"), digits=0),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(
        path,
        ["Dataset", "Model", "Prep s", "Fit s", "Cal s", "Total Fit s", "Infer s", "Peak MB", "Features", "Splits"],
        rows,
    )


def write_ece_sensitivity_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = PAPER_MODELS
    table = summary[summary["Model"].isin(key_models)].copy()
    model_order = {model: index for index, model in enumerate(key_models)}
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("ECE (10-bin)")),
                fmt_float(row.get("ECE (15-bin)")),
                fmt_float(row.get("ECE (20-bin)")),
                fmt_float(row.get("AdaptiveECE (10-bin)")),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(
        path,
        ["Dataset", "Model", "ECE10", "ECE15", "ECE20", "Adaptive ECE10", "Splits"],
        rows,
    )


def plot_auc_ece_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    key_models = PAPER_MODELS
    table = summary[summary["Model"].isin(key_models)].copy()
    if table.empty:
        return
    colors = {
        "logreg_baseline": "#6b7280",
        "xgb_baseline": "#2563eb",
        "lightgbm_baseline": "#0891b2",
        "catboost_baseline": "#ca8a04",
        "histgb_baseline": "#64748b",
        "compact_xgb": "#059669",
        "tabpfn_baseline": "#dc2626",
        "tabicl_baseline": "#7c3aed",
        "reliability_ensemble": "#111827",
        "rc_stack": "#be123c",
        "rc_stack_dr": "#0f766e",
        "rrc_stack": "#9333ea",
    }
    markers = {
        "logreg_baseline": "o",
        "xgb_baseline": "s",
        "lightgbm_baseline": "P",
        "catboost_baseline": "X",
        "histgb_baseline": "<",
        "compact_xgb": "^",
        "tabpfn_baseline": "D",
        "tabicl_baseline": "v",
        "reliability_ensemble": "*",
        "rc_stack": "h",
        "rc_stack_dr": "p",
        "rrc_stack": "X",
    }
    datasets = [dataset for dataset in DISPLAY_DATASETS if dataset in set(table["Dataset"])]
    ncols = max(1, (len(datasets) + 1) // 2)
    fig, axes = plt.subplots(2, ncols, figsize=(max(8.0, 2.4 * ncols), 5.8), sharex=False, sharey=False)
    axes = axes.ravel()
    for axis, dataset in zip(axes, datasets):
        subset = table[table["Dataset"] == dataset]
        for _, row in subset.iterrows():
            model = row["Model"]
            label = DISPLAY_MODELS.get(model, model)
            axis.scatter(
                row["ECE (10-bin)"],
                row["AUC"],
                color=colors.get(model, "#111827"),
                marker=markers.get(model, "o"),
                s=58,
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=3,
            )
        axis.set_title(DISPLAY_DATASETS.get(dataset, dataset), fontsize=10)
        axis.set_xlabel("ECE (lower is better)", fontsize=8)
        axis.set_ylabel("AUC (higher is better)", fontsize=8)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        axis.tick_params(labelsize=8)
    for axis in axes[len(datasets):]:
        axis.axis("off")
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))
    ordered_labels = [DISPLAY_MODELS[model] for model in key_models if DISPLAY_MODELS[model] in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        ncol=min(5, max(1, len(ordered_labels))),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_weak_label_sensitivity(weak_label: pd.DataFrame, path: Path, dataset: str = "german_credit") -> None:
    if weak_label.empty:
        return
    variants = ["baseline", "label_noise_05", "label_noise_10", "proxy_drop"]
    if dataset != "german_credit":
        variants = ["baseline", "label_noise_05", "label_noise_10"]
    models = ["logreg_baseline", "xgb_baseline", "compact_xgb"]
    table = weak_label[
        (weak_label["Dataset"] == dataset)
        & weak_label["Variant"].isin(variants)
        & weak_label["Model"].isin(models)
    ].copy()
    if table.empty:
        return
    colors = {
        "logreg_baseline": "#6b7280",
        "xgb_baseline": "#2563eb",
        "compact_xgb": "#059669",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    x_positions = list(range(len(variants)))
    labels = [variant.replace("_", " ") for variant in variants]
    for model in models:
        subset = table[table["Model"] == model].set_index("Variant").reindex(variants)
        axes[0].plot(
            x_positions,
            subset["AUC"],
            marker="o",
            linewidth=1.8,
            color=colors[model],
            label=DISPLAY_MODELS[model],
        )
        axes[1].plot(
            x_positions,
            subset["ECE (10-bin)"],
            marker="o",
            linewidth=1.8,
            color=colors[model],
            label=DISPLAY_MODELS[model],
        )
    for axis, ylabel in zip(axes, ["AUC", "ECE"]):
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        axis.tick_params(labelsize=8)
    dataset_label = DISPLAY_DATASETS.get(dataset, dataset)
    axes[0].set_title(f"{dataset_label}: discrimination shifts", fontsize=10)
    axes[1].set_title(f"{dataset_label}: calibration shifts", fontsize=10)
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_weak_label_latex(weak_label: pd.DataFrame, path: Path) -> None:
    if weak_label.empty:
        write_latex_table(path, ["Dataset", "Variant", "Model", "AUC", "ECE", "Splits"], [])
        return
    variants = ["baseline", "label_noise_05", "label_noise_10", "proxy_drop"]
    models = ["logreg_baseline", "xgb_baseline", "compact_xgb"]
    table = weak_label[weak_label["Variant"].isin(variants) & weak_label["Model"].isin(models)].copy()
    dataset_order = {dataset: index for index, dataset in enumerate(DISPLAY_DATASETS)}
    variant_order = {variant: index for index, variant in enumerate(variants)}
    model_order = {model: index for index, model in enumerate(models)}
    table["dataset_order"] = table["Dataset"].map(dataset_order).fillna(999)
    table["variant_order"] = table["Variant"].map(variant_order).fillna(999)
    table["model_order"] = table["Model"].map(model_order).fillna(999)
    table = table.sort_values(["dataset_order", "variant_order", "model_order"])
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"]),
                str(row["Variant"]).replace("_", " "),
                DISPLAY_MODELS.get(row["Model"], row["Model"]),
                fmt_float(row.get("AUC")),
                fmt_float(row.get("ECE (10-bin)")),
                str(int(float(row.get("n_splits", 0)))) if not pd.isna(row.get("n_splits")) else "--",
            ]
        )
    write_latex_table(path, ["Dataset", "Variant", "Model", "AUC", "ECE", "Splits"], rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final paper tables and figures.")
    parser.add_argument("--output-root", default="outputs/fmsd", help="Root created by run_fmsd_experiments.py.")
    parser.add_argument("--asset-root", default="paper_assets/fmsd", help="Directory for paper-ready CSV/Markdown assets.")
    parser.add_argument(
        "--efficiency-root",
        default=None,
        help="Optional timing-probe root. If omitted, outputs/efficiency_probe is used when present.",
    )
    parser.add_argument("--include-tabpfn", action="store_true", help="Accepted for compatibility; final assets include completed TabPFN rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    asset_root = Path(args.asset_root)
    efficiency_root = Path(args.efficiency_root) if args.efficiency_root else Path("outputs") / "efficiency_probe"
    asset_root.mkdir(parents=True, exist_ok=True)

    calibrated = prefer_primary(
        collect_aggregate_tables(output_root, "benchmark_calibrated"),
        collect_legacy_aggregate_tables(Path("outputs") / "benchmark", "benchmark_calibrated"),
    )
    calibrated_splits = prefer_primary_splits(
        collect_split_tables(output_root, "benchmark_calibrated"),
        collect_legacy_split_tables(Path("outputs") / "benchmark", "benchmark_calibrated"),
    )
    uncalibrated = prefer_primary(
        collect_aggregate_tables(output_root, "benchmark_uncalibrated"),
        collect_legacy_aggregate_tables(Path("outputs_no_cal") / "benchmark", "benchmark_uncalibrated"),
    )
    summary = build_summary_table(calibrated)
    efficiency_summary = summary
    if efficiency_root.exists():
        efficiency_calibrated = collect_aggregate_tables(efficiency_root, "benchmark_calibrated")
        if not efficiency_calibrated.empty:
            efficiency_summary = build_summary_table(efficiency_calibrated)
    calibration_delta = build_calibration_delta(calibrated, uncalibrated)
    weak_label = collect_weak_label_summary(output_root)
    win_counts = build_paired_win_counts(calibrated_splits)
    paired_tests = build_paired_stat_tests(calibrated_splits)
    subgroup_summary = build_subgroup_reliability_summary(
        collect_subgroup_tables(output_root, "benchmark_calibrated")
    )

    summary.to_csv(asset_root / "main_summary_table.csv", index=False)
    efficiency_summary.to_csv(asset_root / "efficiency_summary_table.csv", index=False)
    calibration_delta.to_csv(asset_root / "calibration_delta_table.csv", index=False)
    weak_label.to_csv(asset_root / "weak_label_sensitivity_table.csv", index=False)
    win_counts.to_csv(asset_root / "paired_win_counts.csv", index=False)
    paired_tests.to_csv(asset_root / "paired_stat_tests.csv", index=False)
    subgroup_summary.to_csv(asset_root / "subgroup_reliability_summary.csv", index=False)
    write_main_results_latex(summary, asset_root / "main_results_table.tex")
    write_variability_latex(summary, asset_root / "main_results_with_ci_table.tex")
    write_paired_win_counts_latex(win_counts, asset_root / "paired_win_counts_table.tex")
    write_paired_stat_tests_latex(paired_tests, asset_root / "paired_stat_tests_table.tex")
    write_decision_cost_latex(summary, asset_root / "decision_cost_table.tex")
    write_subgroup_reliability_latex(subgroup_summary, asset_root / "subgroup_reliability_table.tex")
    write_efficiency_latex(efficiency_summary, asset_root / "efficiency_table.tex")
    write_ece_sensitivity_latex(summary, asset_root / "ece_sensitivity_table.tex")
    write_weak_label_latex(weak_label, asset_root / "weak_label_two_dataset_table.tex")
    plot_auc_ece_tradeoff(summary, asset_root / "auc_ece_tradeoff.png")
    plot_weak_label_sensitivity(weak_label, asset_root / "weak_label_sensitivity.png", dataset="german_credit")
    plot_weak_label_sensitivity(
        weak_label,
        asset_root / "weak_label_sensitivity_australian.png",
        dataset="australian_credit",
    )
    print(f"Paper assets written under {asset_root.resolve()}")


if __name__ == "__main__":
    main()
