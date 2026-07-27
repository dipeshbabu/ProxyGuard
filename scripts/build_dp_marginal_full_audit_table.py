from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ruap_audit import DISPLAY_DATASETS


DATASETS = ["australian_credit", "german_credit", "compas_recidivism"]


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def display_model(name: str) -> str:
    return {
        "xgb_baseline": "XGB",
        "compact_xgb": "Compact XGB",
        "tabpfn_baseline": "TabPFN",
        "tabicl_baseline": "TabICL",
        "rc_stack": "RC-Stack",
        "rrc_stack": "RRC-Stack",
    }.get(name, name.replace("_", " "))


def build_table(proxy_root: Path, exposure_csv: Path) -> pd.DataFrame:
    exposure = pd.read_csv(exposure_csv)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        baseline_path = proxy_root / "proxy_transform" / "baseline" / dataset / "aggregate_metrics.csv"
        dp_path = proxy_root / "proxy_transform" / "dp_marginal_e1" / dataset / "aggregate_metrics.csv"
        if not baseline_path.exists() or not dp_path.exists():
            continue
        baseline = pd.read_csv(baseline_path)
        dp = pd.read_csv(dp_path)
        completed_models = sorted(set(dp["Model"]))
        matched_baseline = baseline[baseline["Model"].isin(completed_models)]
        if dp.empty or matched_baseline.empty:
            continue
        best_dp = dp.loc[dp["AUC"].idxmax()]
        exposure_row = exposure[
            (exposure["Dataset"] == dataset) & (exposure["Variant"] == "dp_marginal_e1")
        ].iloc[0]
        rows.append(
            {
                "Dataset": DISPLAY_DATASETS.get(dataset, dataset),
                "Models": len(completed_models),
                "BestAUCModel": best_dp["Model"],
                "BestAUCModelDisplay": display_model(str(best_dp["Model"])),
                "AUCDelta": float(best_dp["AUC"] - matched_baseline["AUC"].max()),
                "ECEDelta": float(best_dp["ECE (10-bin)"] - matched_baseline["ECE (10-bin)"].min()),
                "CostDelta": float(best_dp["DecisionCost5x"] - matched_baseline["DecisionCost5x"].min()),
                "LeakDelta": float(exposure_row["SensitivePredictabilityDelta"]),
                "MemAUCDelta": float(exposure_row["MemberAUCDelta"]),
            }
        )
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{0.92\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Completed models & Highest-AUC model & $\\Delta$AUC & $\\Delta$ECE & $\\Delta$Cost & $\\Delta$Leak / $\\Delta$MemAUC \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Dataset']} & {int(row['Models'])} & {row['BestAUCModelDisplay']} & "
            f"{fmt_delta(row['AUCDelta'])} & {fmt_delta(row['ECEDelta'])} & "
            f"{fmt_delta(row['CostDelta'])} & {fmt_delta(row['LeakDelta'])} / {fmt_delta(row['MemAUCDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{DP marginal check on the three main proxy datasets. The displayed row has the highest test AUC among completed models and is descriptive, not a selection procedure. Deltas use the original-table baseline over the same model set. Lower ECE, Cost5x, Leak, and MemAUC deltas are better.}",
            "\\label{tab:dp_marginal_full_audit}",
            "\\end{table}",
        ]
    )
    (output_dir / "dp_marginal_full_audit_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact DP marginal full-audit table.")
    parser.add_argument("--proxy-root", default="outputs/proxy_transform_audit")
    parser.add_argument("--exposure-csv", default="paper_assets/ruap_audit/extended_exposure_stress.csv")
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(Path(args.proxy_root), Path(args.exposure_csv))
    table.to_csv(output_dir / "dp_marginal_full_audit_table.csv", index=False)
    write_latex(table, output_dir)
    print(f"wrote {len(table)} DP marginal full-audit rows to {output_dir}")


if __name__ == "__main__":
    main()
