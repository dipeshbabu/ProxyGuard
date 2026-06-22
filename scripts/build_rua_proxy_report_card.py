from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DISPLAY_DATASETS = {
    "australian_credit": "Australian",
    "german_credit": "German",
}

DISPLAY_VARIANTS = {
    "baseline": "Original",
    "numeric_noise_10": "10\\% numeric noise",
    "numeric_noise_20": "20\\% numeric noise",
    "coarsen_quartile": "Quartile coarsening",
    "feature_mask_20": "20\\% feature mask",
}

DISPLAY_MODELS = {
    "xgb_baseline": "XGB",
    "compact_xgb": "Compact XGB",
    "tabpfn_baseline": "TabPFN",
    "tabicl_baseline": "TabICL",
    "rc_stack": "RC-Stack",
    "rrc_stack": "RRC-Stack",
}


def read_proxy_outputs(root: Path) -> pd.DataFrame:
    rows = []
    for aggregate_path in sorted((root / "proxy_transform").glob("*/*/aggregate_metrics.csv")):
        variant = aggregate_path.parent.parent.name
        dataset = aggregate_path.parent.name
        frame = pd.read_csv(aggregate_path)
        frame.insert(0, "Variant", variant)
        frame.insert(0, "Dataset", dataset)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def best_row(frame: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.Series:
    idx = frame[metric].idxmax() if higher_is_better else frame[metric].idxmin()
    return frame.loc[idx]


def build_report_card(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_by_dataset = {}
    for dataset, dataset_frame in results.groupby("Dataset"):
        baseline = dataset_frame[dataset_frame["Variant"] == "baseline"]
        if baseline.empty:
            continue
        baseline_by_dataset[dataset] = {
            "AUC": best_row(baseline, "AUC", True)["AUC"],
            "ECE (10-bin)": best_row(baseline, "ECE (10-bin)", False)["ECE (10-bin)"],
            "DecisionCost5x": best_row(baseline, "DecisionCost5x", False)["DecisionCost5x"],
        }

    for (dataset, variant), frame in results.groupby(["Dataset", "Variant"]):
        if dataset not in baseline_by_dataset:
            continue
        auc = best_row(frame, "AUC", True)
        ece = best_row(frame, "ECE (10-bin)", False)
        cost = best_row(frame, "DecisionCost5x", False)
        baseline = baseline_by_dataset[dataset]
        rows.append(
            {
                "Dataset": dataset,
                "Variant": variant,
                "BestAUCModel": auc["Model"],
                "BestAUC": auc["AUC"],
                "AUCDelta": auc["AUC"] - baseline["AUC"],
                "BestECEModel": ece["Model"],
                "BestECE": ece["ECE (10-bin)"],
                "ECEDelta": ece["ECE (10-bin)"] - baseline["ECE (10-bin)"],
                "BestCostModel": cost["Model"],
                "BestCost5x": cost["DecisionCost5x"],
                "CostDelta": cost["DecisionCost5x"] - baseline["DecisionCost5x"],
                "n_splits": int(frame["n_splits"].min()),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    variant_order = {variant: index for index, variant in enumerate(DISPLAY_VARIANTS)}
    report["VariantOrder"] = report["Variant"].map(variant_order).fillna(99)
    return report.sort_values(["Dataset", "VariantOrder"]).drop(columns=["VariantOrder"]).reset_index(drop=True)


def fmt_delta(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def display_model(name: str) -> str:
    return DISPLAY_MODELS.get(name, name.replace("_", "\\_"))


def write_latex_table(report: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\caption{Proxy-transform RUA report card. Deltas compare the best model under each transform with the best model on the original version of the same dataset. Lower ECE and Cost5x are better.}",
        "\\label{tab:proxy_rua}",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Dataset & Proxy & AUC winner & $\\Delta$AUC & $\\Delta$ECE & $\\Delta$Cost5x \\\\",
        "\\midrule",
    ]
    for _, row in report.iterrows():
        if row["Variant"] == "baseline":
            continue
        dataset = DISPLAY_DATASETS.get(row["Dataset"], row["Dataset"].replace("_", "\\_"))
        variant = DISPLAY_VARIANTS.get(row["Variant"], row["Variant"].replace("_", "\\_"))
        lines.append(
            f"{dataset} & {variant} & {display_model(row['BestAUCModel'])} & "
            f"{fmt_delta(row['AUCDelta'])} & {fmt_delta(row['ECEDelta'])} & {fmt_delta(row['CostDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\vskip -0.08in",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RUA proxy-transform report card.")
    parser.add_argument("--input-root", default="outputs/proxy_transform_audit")
    parser.add_argument("--output-dir", default="paper_assets/proxy_transform_audit")
    args = parser.parse_args()

    results = read_proxy_outputs(Path(args.input_root))
    report = build_report_card(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "proxy_transform_rua_report_card.csv", index=False)
    write_latex_table(report, output_dir / "proxy_transform_rua_report_card.tex")
    print(f"wrote {len(report)} report-card rows to {output_dir}")


if __name__ == "__main__":
    main()
