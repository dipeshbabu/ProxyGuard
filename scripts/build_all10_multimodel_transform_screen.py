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
    "adult_income": "Adult",
    "bank_marketing": "Bank",
    "breast_cancer_wdbc": "WDBC",
    "give_me_some_credit": "GMSC",
    "heart_disease": "Heart",
    "mammographic_mass": "Mammo",
}

MODEL_DISPLAY = {
    "logreg_baseline": "LogReg",
    "histgb_baseline": "HistGB",
}

VARIANT_ORDER = ["coarsen_quartile", "sensitive_mask", "dp_marginal_e1"]


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def build_screen(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    for dataset, dataset_frame in frame.groupby("Dataset"):
        for variant in VARIANT_ORDER:
            variant_frame = dataset_frame[dataset_frame["Variant"] == variant]
            if variant_frame.empty:
                continue
            out: dict[str, object] = {"Dataset": dataset, "Variant": variant}
            for model in MODEL_DISPLAY:
                baseline = dataset_frame[(dataset_frame["Variant"] == "baseline") & (dataset_frame["Model"] == model)]
                row = variant_frame[variant_frame["Model"] == model]
                if baseline.empty or row.empty:
                    out[f"{model}_auc_delta"] = np.nan
                    out[f"{model}_cost_delta"] = np.nan
                    continue
                base = baseline.iloc[0]
                current = row.iloc[0]
                out[f"{model}_auc_delta"] = float(current["AUC"] - base["AUC"])
                out[f"{model}_cost_delta"] = float(current["DecisionCost5x"] - base["DecisionCost5x"])
            rows.append(out)
    result = pd.DataFrame(rows)
    result["VariantOrder"] = result["Variant"].map({name: index for index, name in enumerate(VARIANT_ORDER)})
    return result.sort_values(["Dataset", "VariantOrder"]).drop(columns=["VariantOrder"]).reset_index(drop=True)


def write_latex(screen: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.04}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Transform & LogReg $\\Delta$AUC & HistGB $\\Delta$AUC & LogReg $\\Delta$Cost5x & HistGB $\\Delta$Cost5x \\\\",
        "\\midrule",
    ]
    for _, row in screen.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{DISPLAY_VARIANTS.get(row['Variant'], row['Variant'].replace('_', ' '))} & "
            f"{fmt_delta(row['logreg_baseline_auc_delta'])} & "
            f"{fmt_delta(row['histgb_baseline_auc_delta'])} & "
            f"{fmt_delta(row['logreg_baseline_cost_delta'])} & "
            f"{fmt_delta(row['histgb_baseline_cost_delta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{All-ten two-model transform utility extension. Deltas compare each transformed table with the original table for the same dataset and model using one repeated split. The table is a coverage check for transform sensitivity across all datasets, not a replacement for the six-model repeated-split proxy audit in the main text. Lower Cost5x deltas are better.}\\label{tab:all10_multimodel_utility_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "all10_multimodel_transform_screen.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all-ten two-model transform utility screen.")
    parser.add_argument("--summary", default="outputs/proxy_transform_all10_fast/proxy_transform_summary.csv")
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screen = build_screen(Path(args.summary))
    screen.to_csv(output_dir / "all10_multimodel_transform_screen.csv", index=False)
    write_latex(screen, output_dir)
    print(f"wrote {len(screen)} all-ten multi-model transform rows to {output_dir}")


if __name__ == "__main__":
    main()
