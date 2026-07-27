from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def write_calibration_table(summary: pd.DataFrame, output_path: Path) -> None:
    headline = summary[summary["AuditN"].eq(500) & summary["Alpha"].eq(0.05)].copy()
    rows = [
        (
            latex_escape(row.Method),
            f"{100.0 * row.FalseAcceptanceRate:.1f}\\%",
            f"{100.0 * row.ValidCandidatePower:.1f}\\%",
        )
        for row in headline.itertuples(index=False)
    ]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Rule & False acceptance & Valid-candidate power \\",
        r"\midrule",
    ]
    lines.extend(f"{method} & {false_acceptance} & {power} \\\\" for method, false_acceptance, power in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_real_audit_table(summary: pd.DataFrame, output_path: Path) -> None:
    summary = summary.copy()
    summary["Dataset"] = summary["Candidate"].str.split("/", n=1).str[0]
    grouped = (
        summary.groupby("Dataset", sort=True)
        .agg(
            AuditN=("AuditNMin", "first"),
            Validated=("Validated", "sum"),
            Violations=("ViolationDetected", "sum"),
            Unresolved=("Status", lambda values: int((values == "Unresolved").sum())),
        )
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Audit $n$ & Validated & Violations & Unresolved \\",
        r"\midrule",
    ]
    for row in grouped.itertuples(index=False):
        dataset = latex_escape(str(row.Dataset).replace("_", " ").title())
        lines.append(
            f"{dataset} & {int(row.AuditN):,} & {int(row.Validated)} & "
            f"{int(row.Violations)} & {int(row.Unresolved)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aim_audit_table(summary: pd.DataFrame, output_path: Path) -> None:
    summary = summary.sort_values(["Dataset", "Epsilon"]).copy()
    status_symbol = {
        "Validated": "Val.",
        "Violation detected": "Viol.",
        "Unresolved": "Unres.",
    }
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Dataset & Audit $n$ & $\epsilon$ & $\Delta$AUC & $\Delta$Cost5x & Decision \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        dataset = latex_escape(str(row.Dataset).replace("_", " ").title())
        cost_change = float(row.ProxyCost5x) - float(row.SourceCost5x)
        lines.append(
            f"{dataset} & {int(row.AuditN):,} & {float(row.Epsilon):g} & "
            f"{float(row.AUCChange):+.3f} & {cost_change:+.3f} & "
            f"{status_symbol[str(row.Status)]} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables for the ProxyGuard manuscript.")
    parser.add_argument(
        "--calibration-summary",
        default="outputs/proxyguard_calibration/calibration_summary.csv",
    )
    parser.add_argument(
        "--real-summary",
        default="outputs/proxyguard_real_audit/candidate_summary.csv",
    )
    parser.add_argument(
        "--aim-summary",
        default="outputs/proxyguard_aim_audit/candidate_summary.csv",
    )
    parser.add_argument("--output-dir", default="paper/proxyguard/tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_calibration_table(
        pd.read_csv(args.calibration_summary),
        output_dir / "calibration_headline.tex",
    )
    write_real_audit_table(
        pd.read_csv(args.real_summary),
        output_dir / "real_audit_by_dataset.tex",
    )
    write_aim_audit_table(
        pd.read_csv(args.aim_summary),
        output_dir / "aim_audit.tex",
    )


if __name__ == "__main__":
    main()
