from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ruap_audit import DISPLAY_DATASETS, DISPLAY_VARIANTS


DATASET_DISPLAY = {
    **DISPLAY_DATASETS,
    "heart_disease": "Heart",
    "mammographic_mass": "Mammo",
    "breast_cancer_wdbc": "WDBC",
    "give_me_some_credit": "GMSC",
}


VARIANT_ORDER = [
    "laplace_noise_20",
    "rank_swap_10",
    "coarsen_quartile",
    "sensitive_mask",
    "synthetic_marginal",
    "noisy_synthetic_marginal",
    "dp_marginal_e1",
]


def read_proxy_outputs(root: Path) -> pd.DataFrame:
    rows = []
    for aggregate_path in sorted((root / "proxy_transform").glob("*/*/aggregate_metrics.csv")):
        variant = aggregate_path.parent.parent.name
        dataset = aggregate_path.parent.name
        frame = pd.read_csv(aggregate_path)
        frame.insert(0, "Variant", variant)
        frame.insert(0, "Dataset", dataset)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_screen(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = frame[frame["Model"] == model].copy()
    rows = []
    for dataset, dataset_frame in frame.groupby("Dataset"):
        baseline = dataset_frame[dataset_frame["Variant"] == "baseline"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for _, row in dataset_frame[dataset_frame["Variant"] != "baseline"].iterrows():
            if row["Variant"] not in VARIANT_ORDER:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": row["Variant"],
                    "AUC": float(row["AUC"]),
                    "AUCDelta": float(row["AUC"] - baseline_row["AUC"]),
                    "ECE": float(row["ECE (10-bin)"]),
                    "ECEDelta": float(row["ECE (10-bin)"] - baseline_row["ECE (10-bin)"]),
                    "Cost5x": float(row["DecisionCost5x"]),
                    "CostDelta": float(row["DecisionCost5x"] - baseline_row["DecisionCost5x"]),
                    "Splits": int(row["n_splits"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    result["VariantOrder"] = result["Variant"].map(order)
    return result.sort_values(["Dataset", "VariantOrder"]).drop(columns=["VariantOrder"]).reset_index(drop=True)


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def write_latex_table(screen: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Transform & $\\Delta$AUC & $\\Delta$ECE & $\\Delta$Cost5x & Splits \\\\",
        "\\midrule",
    ]
    for _, row in screen.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{DISPLAY_VARIANTS.get(row['Variant'], row['Variant'].replace('_', ' '))} & "
            f"{fmt_delta(row['AUCDelta'])} & {fmt_delta(row['ECEDelta'])} & "
            f"{fmt_delta(row['CostDelta'])} & {int(row['Splits'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Targeted proxy utility screen. Deltas compare each transformed table with the original table for the same dataset using XGBoost and five repeated splits. Lower ECE and Cost5x deltas are better. Marginal synthetic rows resample each feature independently without labels; noisy marginal adds uncalibrated Laplace count noise; DP marginal uses Laplace-noised independent histograms with $\\epsilon=1$ under fixed-bin assumptions. These are compact synthetic stress baselines, not full generative-model baselines.}\\label{tab:targeted_proxy_utility_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "targeted_proxy_utility_screen.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted proxy utility screen from proxy-transform outputs.")
    parser.add_argument("--proxy-root", default="outputs/proxy_transform_screen")
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    parser.add_argument("--model", default="xgb_baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screen = build_screen(read_proxy_outputs(Path(args.proxy_root)), model=args.model)
    screen.to_csv(output_dir / "targeted_proxy_utility_screen.csv", index=False)
    write_latex_table(screen, output_dir)
    print(f"wrote {len(screen)} targeted utility rows to {output_dir}")


if __name__ == "__main__":
    main()
