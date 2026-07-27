from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset(value: str) -> str:
    return {
        "heart_disease": "Heart",
        "german_credit": "German",
        "taiwan_default": "Taiwan",
    }.get(value, value.replace("_", " ").title())


def _mechanism_status(value: str) -> str:
    return {
        "Mechanism validated": "Validated",
        "Reliability violation detected": "Violation",
        "Unresolved": "Unresolved",
    }[value]


def write_mechanism_calibration_table(
    summary: pd.DataFrame,
    output_path: Path,
    audit_size: int = 500,
    releases: int = 50,
) -> None:
    frame = summary[
        summary["AuditN"].eq(audit_size) & summary["Releases"].eq(releases)
    ]
    labels = {
        "Plug-in release fraction": "Plug-in fraction",
        "Point rule + binomial": "Point rule + binomial",
        "Per-release IUT + binomial": "Per-release IUT + binomial",
        "Two-level ProxyGuard": "Two-level \\method{}",
        "Oracle release labels": "Oracle release labels",
    }
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & False mechanism validation & Power \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{labels[row.Method]} & {row.FalseMechanismValidation:.3f} & "
            f"{row.MechanismValidationPower:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_adaptive_table(summary: pd.DataFrame, output_path: Path) -> None:
    false = summary[summary["Analysis"].eq("False validation")].pivot(
        index="Round",
        columns="Method",
        values="Rate",
    )
    power = summary[
        summary["Analysis"].eq("Valid-candidate power")
        & summary["Method"].eq("Quadratic alpha spending")
    ].set_index("Round")["Rate"]
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        (
            r"Rounds & Fixed-$\alpha$ false validation & Spending false validation "
            r"& Spending power \\"
        ),
        r"\midrule",
    ]
    for round_number in (1, 10, 25, 50):
        lines.append(
            f"{round_number} & "
            f"{false.loc[round_number, 'Fixed alpha each round']:.3f} & "
            f"{false.loc[round_number, 'Quadratic alpha spending']:.3f} & "
            f"{power.loc[round_number]:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_real_mechanism_table(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.sort_values(["Dataset", "Epsilon"])
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        (
            r"Dataset & $\epsilon$ & Releases & Release V/U/F & One-sided bounds "
            r"& Mechanism decision \\"
        ),
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        unresolved = (
            int(row.Releases)
            - int(row.ValidatedReleases)
            - int(row.DetectedReleaseViolations)
        )
        lines.append(
            f"{_dataset(row.Dataset)} & {row.Epsilon:g} & {int(row.Releases)} & "
            f"{int(row.ValidatedReleases)}/{unresolved}/"
            f"{int(row.DetectedReleaseViolations)} & "
            f"[{row.ReliabilityLCB:.2f}, {row.ReliabilityUCB:.2f}] & "
            f"{_mechanism_status(row.Status)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_release_planning_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Reliability target & Mechanisms screened & All-good releases needed \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        reliability_percent = f"{100.0 * row.MinimumReliability:.0f}\\%"
        lines.append(
            f"{reliability_percent} & {int(row.Mechanisms)} & "
            f"{int(row.AllRecognizedGoodReleasesNeeded)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_near_boundary_table(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[
        summary["Method"].isin(
            ["Uncorrected release IUT", "Inner Holm (ProxyGuard)"]
        )
    ].copy()
    frame["_method_order"] = frame["Method"].map(
        {
            "Uncorrected release IUT": 0,
            "Inner Holm (ProxyGuard)": 1,
        }
    )
    frame = frame.sort_values(["BadReleaseRisk", "_method_order"])
    labels = {
        "Uncorrected release IUT": "Separate release IUTs",
        "Inner Holm (ProxyGuard)": "Inner Holm",
    }
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        (
            r"Bad-release risk & Inner rule & Any false release label "
            r"& False mechanism validation \\"
        ),
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.BadReleaseRisk:.3f} & {labels[row.Method]} & "
            f"{row.FalseReleaseRecognitionFWER:.4f} & "
            f"{row.FalseMechanismValidation:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_alpha_allocation_table(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.sort_values("ReleaseErrorShare")
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        (
            r"$\lambda$ & $\alpha_{\mathrm R}$ & $\alpha_{\mathrm M}$ "
            r"& False mechanism validation & Power \\"
        ),
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.ReleaseErrorShare:.2f} & {row.ReleaseAlpha:.3f} & "
            f"{row.MechanismAlpha:.3f} & "
            f"{row.FalseMechanismValidation:.3f} & "
            f"{row.MechanismValidationPower:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_prospective_aim_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        (
            r"Mechanism & Releases & Release V/U/F & One-sided bounds "
            r"& Mean Cost5x regret & Decision \\"
        ),
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        unresolved = (
            int(row.Releases)
            - int(row.ValidatedReleases)
            - int(row.DetectedReleaseViolations)
        )
        lines.append(
            f"AIM, $\\epsilon={row.Epsilon:g}$ & {int(row.Releases)} & "
            f"{int(row.ValidatedReleases)}/{unresolved}/"
            f"{int(row.DetectedReleaseViolations)} & "
            f"[{row.ReliabilityLCB:.2f}, {row.ReliabilityUCB:.2f}] & "
            f"{row.MeanCost5xChange:+.3f} & {_mechanism_status(row.Status)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_positive_bootstrap_table(
    detail: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = (
        detail.groupby(["Requirement", "Estimand"], as_index=False)
        .agg(
            Tolerance=("Tolerance", "first"),
            MeanValue=("MeanValue", "mean"),
            MaxUCB=("SimultaneousUCB", "max"),
        )
        .sort_values(["Estimand", "Requirement"])
    )
    estimand_labels = {
        "absolute_risk": "Absolute risk",
        "relative_regret": "Relative regret",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Requirement & Form & Limit & Mean & Largest upper bound \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.Requirement} & "
            f"{estimand_labels.get(row.Estimand, row.Estimand)} & "
            f"{row.Tolerance:.3f} & {row.MeanValue:.3f} & "
            f"{row.MaxUCB:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_target_lineage_table(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = {
        "AIM informed replication": "AIM informed rerun",
        "Bootstrap informed positive control": "Bootstrap informed rerun",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Study & Audit $n$ & In pilot train & In pilot validation & In pilot audit \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{labels.get(row.Study, row.Study)} & "
            f"{int(row.ReplicationAuditN):,} & "
            f"{int(row.OverlapPilotTrain):,} & "
            f"{int(row.OverlapPilotValidation):,} & "
            f"{int(row.OverlapPilotAudit):,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, [line.replace(",", r"{,}") for line in lines])


def write_claim_status_table(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        (
            r"\begin{tabularx}{\textwidth}{@{}"
            r">{\raggedright\arraybackslash}p{3.0cm}"
            r">{\raggedright\arraybackslash}p{2.7cm}"
            r">{\raggedright\arraybackslash}Xcc@{}}"
        ),
        r"\toprule",
        (
            r"Experiment & Evidence status & Target lineage & Registered "
            r"& Guarantee \\"
        ),
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.Experiment} & {row.EvidenceClass} & {row.TargetLineage} & "
            f"{row.PreAccessRegistry} & {row.FormalGuarantee} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(output_path, lines)


def write_magic_sealed_table(
    detail: pd.DataFrame,
    output_path: Path,
) -> None:
    summary = detail.groupby("Requirement", as_index=True).agg(
        Mean=("MeanValue", "mean"),
        MaxUCB=("SimultaneousUCB", "max"),
        Limit=("Tolerance", "first"),
    )
    rows = [
        ("Brier", "Brier transfer", "Proxy Brier risk"),
        ("Clipped log loss", "Log-loss transfer", "Proxy log-loss risk"),
        ("Cost5x", "Cost5x transfer", "Proxy Cost5x risk"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Metric & Rel. mean & Rel. max UCB & Limit & Abs. mean "
            r"& Abs. max UCB & Limit \\"
        ),
        r"\midrule",
    ]
    for label, relative, absolute in rows:
        rel = summary.loc[relative]
        abs_ = summary.loc[absolute]
        lines.append(
            f"{label} & {rel.Mean:.4f} & {rel.MaxUCB:.4f} & "
            f"{rel.Limit:.3f} & {abs_.Mean:.4f} & {abs_.MaxUCB:.4f} & "
            f"{abs_.Limit:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(output_path, lines)


def write_absolute_risk_baseline_table(
    baselines: pd.DataFrame,
    detail: pd.DataFrame,
    output_path: Path,
) -> None:
    limits = (
        detail.loc[detail["Estimand"].eq("absolute_risk")]
        .groupby("Requirement")["Tolerance"]
        .first()
    )
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Reference & Brier & Clipped log loss & Cost5x \\",
        r"\midrule",
    ]
    for row in baselines.itertuples(index=False):
        lines.append(
            f"{row.Baseline} & {row.BrierRisk:.3f} & "
            f"{row.ClippedLogLossRisk:.3f} & {row.Cost5xRisk:.3f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                f"Registered ceiling & {limits['Proxy Brier risk']:.3f} & "
                f"{limits['Proxy log-loss risk']:.3f} & "
                f"{limits['Proxy Cost5x risk']:.3f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    _write(output_path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mechanism-level ProxyGuard paper tables."
    )
    parser.add_argument(
        "--study-root",
        default="outputs/proxyguard_mechanism_study",
    )
    parser.add_argument(
        "--audit-root",
        default="outputs/proxyguard_mechanism_audit",
    )
    parser.add_argument(
        "--revision-root",
        default="outputs/proxyguard_mechanism_revision_study",
    )
    parser.add_argument(
        "--prospective-aim-root",
        default="outputs/proxyguard_prospective_aim_mechanism",
    )
    parser.add_argument(
        "--positive-bootstrap-root",
        default="outputs/proxyguard_bootstrap_positive_replication",
    )
    parser.add_argument(
        "--lineage-root",
        default="outputs/proxyguard_target_lineage",
    )
    parser.add_argument(
        "--claim-status",
        default="registries/proxyguard_claim_status.csv",
    )
    parser.add_argument(
        "--magic-sealed-root",
        default="outputs/proxyguard_magic_sealed_mechanism",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/proxyguard/tables",
    )
    args = parser.parse_args()

    study_root = Path(args.study_root)
    audit_root = Path(args.audit_root)
    revision_root = Path(args.revision_root)
    prospective_aim_root = Path(args.prospective_aim_root)
    positive_bootstrap_root = Path(args.positive_bootstrap_root)
    lineage_root = Path(args.lineage_root)
    claim_status_path = Path(args.claim_status)
    magic_sealed_root = Path(args.magic_sealed_root)
    output_dir = Path(args.output_dir)
    write_mechanism_calibration_table(
        pd.read_csv(study_root / "mechanism_calibration.csv"),
        output_dir / "mechanism_calibration.tex",
    )
    write_adaptive_table(
        pd.read_csv(study_root / "adaptive_search.csv"),
        output_dir / "adaptive_search.tex",
    )
    write_real_mechanism_table(
        pd.read_csv(audit_root / "mechanism_summary_primary.csv"),
        output_dir / "aim_mechanism_reliability.tex",
    )
    write_release_planning_table(
        pd.read_csv(study_root / "release_planning.csv"),
        output_dir / "mechanism_planning.tex",
    )
    write_near_boundary_table(
        pd.read_csv(revision_root / "near_boundary_inner_correction.csv"),
        output_dir / "near_boundary_inner_correction.tex",
    )
    write_alpha_allocation_table(
        pd.read_csv(revision_root / "alpha_allocation.csv"),
        output_dir / "alpha_allocation.tex",
    )
    prospective_summary = prospective_aim_root / "mechanism_summary.csv"
    if prospective_summary.exists():
        write_prospective_aim_table(
            pd.read_csv(prospective_summary),
            output_dir / "prospective_aim_mechanism.tex",
        )
    positive_detail = positive_bootstrap_root / "requirement_detail.csv"
    if positive_detail.exists():
        detail = pd.read_csv(positive_detail)
        write_positive_bootstrap_table(
            detail,
            output_dir / "positive_bootstrap_requirements.tex",
        )
        baseline_path = positive_bootstrap_root / "absolute_risk_baselines.csv"
        if baseline_path.exists():
            write_absolute_risk_baseline_table(
                pd.read_csv(baseline_path),
                detail,
                output_dir / "positive_bootstrap_baselines.tex",
            )
    lineage_summary = lineage_root / "lineage_summary.csv"
    if lineage_summary.exists():
        write_target_lineage_table(
            pd.read_csv(lineage_summary),
            output_dir / "audit_lineage.tex",
        )
    if claim_status_path.exists():
        write_claim_status_table(
            pd.read_csv(claim_status_path),
            output_dir / "claim_status.tex",
        )
    magic_detail = magic_sealed_root / "requirement_detail.csv"
    if magic_detail.exists():
        write_magic_sealed_table(
            pd.read_csv(magic_detail),
            output_dir / "magic_sealed_requirements.tex",
        )


if __name__ == "__main__":
    main()
