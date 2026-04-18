from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PAPER_MODELS = [
    "logreg_baseline",
    "xgb_baseline",
    "compact_xgb",
    "tabpfn_baseline",
]

DISPLAY_MODELS = {
    "logreg_baseline": "LogReg",
    "xgb_baseline": "XGB",
    "compact_xgb": "Compact XGB",
    "tabpfn_baseline": "TabPFN",
}

DISPLAY_DATASETS = {
    "australian_credit": "Australian",
    "german_credit": "German",
    "give_me_some_credit": "GMSC",
    "taiwan_default": "Taiwan",
}


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


def prefer_primary(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return fallback
    if fallback.empty:
        return primary
    key = ["Dataset", "Model"]
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
        "Brier",
        "ECE (10-bin)",
        "LogLoss",
        "CalibrationSlope",
        "FeatureCount",
        "n_splits",
    ]
    present = [col for col in cols if col in calibrated.columns]
    summary = calibrated.loc[calibrated["Model"].isin(PAPER_MODELS), present].copy()
    return summary.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def build_calibration_delta(calibrated: pd.DataFrame, uncalibrated: pd.DataFrame) -> pd.DataFrame:
    if calibrated.empty or uncalibrated.empty:
        return pd.DataFrame()
    metrics = ["AUC", "Brier", "ECE (10-bin)", "LogLoss", "CalibrationSlope"]
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
    weak_summary = read_csv_if_exists(output_root / "weak_label" / "weak_label_summary.csv")
    if weak_summary.empty:
        weak_summary = read_csv_if_exists(Path("outputs") / "weak_label" / "weak_label_summary.csv")
    if weak_summary.empty:
        return weak_summary
    keep_cols = [
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
    return weak_summary.loc[weak_summary["Model"].isin(PAPER_MODELS), present].copy()


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


def write_latex_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    column_spec = "ll" + "r" * max(0, len(headers) - 2)
    lines = [
        r"\begin{tabular}{" + column_spec + r"}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_results_latex(summary: pd.DataFrame, path: Path) -> None:
    key_models = ["logreg_baseline", "xgb_baseline", "compact_xgb", "tabpfn_baseline"]
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


def plot_auc_ece_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    key_models = ["logreg_baseline", "xgb_baseline", "compact_xgb", "tabpfn_baseline"]
    table = summary[summary["Model"].isin(key_models)].copy()
    if table.empty:
        return
    colors = {
        "logreg_baseline": "#6b7280",
        "xgb_baseline": "#2563eb",
        "compact_xgb": "#059669",
        "tabpfn_baseline": "#dc2626",
    }
    markers = {
        "logreg_baseline": "o",
        "xgb_baseline": "s",
        "compact_xgb": "^",
        "tabpfn_baseline": "D",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=False, sharey=False)
    axes = axes.ravel()
    datasets = [dataset for dataset in DISPLAY_DATASETS if dataset in set(table["Dataset"])]
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
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_weak_label_sensitivity(weak_label: pd.DataFrame, path: Path) -> None:
    if weak_label.empty:
        return
    variants = ["baseline", "label_noise_05", "label_noise_10", "proxy_drop"]
    models = ["logreg_baseline", "xgb_baseline", "compact_xgb"]
    table = weak_label[weak_label["Variant"].isin(variants) & weak_label["Model"].isin(models)].copy()
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
    axes[0].set_title("Discrimination under weak-label shifts", fontsize=10)
    axes[1].set_title("Calibration under weak-label shifts", fontsize=10)
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final paper tables and figures.")
    parser.add_argument("--output-root", default="outputs/fmsd", help="Root created by run_fmsd_experiments.py.")
    parser.add_argument("--asset-root", default="paper_assets/fmsd", help="Directory for paper-ready CSV/Markdown assets.")
    parser.add_argument("--include-tabpfn", action="store_true", help="Accepted for compatibility; final assets include completed TabPFN rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    asset_root = Path(args.asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    calibrated = prefer_primary(
        collect_aggregate_tables(output_root, "benchmark_calibrated"),
        collect_legacy_aggregate_tables(Path("outputs") / "benchmark", "benchmark_calibrated"),
    )
    uncalibrated = prefer_primary(
        collect_aggregate_tables(output_root, "benchmark_uncalibrated"),
        collect_legacy_aggregate_tables(Path("outputs_no_cal") / "benchmark", "benchmark_uncalibrated"),
    )
    summary = build_summary_table(calibrated)
    calibration_delta = build_calibration_delta(calibrated, uncalibrated)
    weak_label = collect_weak_label_summary(output_root)

    summary.to_csv(asset_root / "main_summary_table.csv", index=False)
    calibration_delta.to_csv(asset_root / "calibration_delta_table.csv", index=False)
    weak_label.to_csv(asset_root / "weak_label_sensitivity_table.csv", index=False)
    write_main_results_latex(summary, asset_root / "main_results_table.tex")
    plot_auc_ece_tradeoff(summary, asset_root / "auc_ece_tradeoff.png")
    plot_weak_label_sensitivity(weak_label, asset_root / "weak_label_sensitivity.png")
    print(f"Paper assets written under {asset_root.resolve()}")


if __name__ == "__main__":
    main()
