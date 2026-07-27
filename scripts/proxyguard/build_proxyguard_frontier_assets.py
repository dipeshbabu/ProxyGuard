from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _status(value: str) -> str:
    return {
        "Validated": "Validated",
        "Violation detected": "Violation",
        "Unresolved": "Unresolved",
    }.get(value, value)


def _dataset(value: str) -> str:
    return {
        "heart_disease": "Heart",
        "german_credit": "German",
        "taiwan_default": "Taiwan",
    }.get(value, value.replace("_", " ").title())


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_out_of_mechanism_table(path: Path, input_path: Path) -> None:
    frame = pd.read_csv(input_path)
    invalid = frame[
        (frame["Truth"] == "invalid") & (frame["AuditN"] == 1000)
    ].pivot(index="Family", columns="Method", values="Rate")
    power = frame[
        (frame["Truth"] == "valid")
        & (frame["AuditN"] == 1000)
    ].pivot(index="Family", columns="Method", values="Rate")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        (
            "Held-out regret family & Point FWER & Uncorrected FWER & "
            "\\method{} FWER & \\method{} power \\\\"
        ),
        "\\midrule",
    ]
    for family in ("continuous beta", "rare-subgroup mixture", "correlated candidates"):
        lines.append(
            f"{family.capitalize()} & {invalid.loc[family, 'Point threshold']:.3f} & "
            f"{invalid.loc[family, 'Uncorrected IUT']:.3f} & "
            f"{invalid.loc[family, 'ProxyGuard']:.3f} & "
            f"{power.loc[family, 'ProxyGuard']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def write_target_reuse_table(path: Path, input_path: Path) -> None:
    frame = pd.read_csv(input_path)
    labels = {
        "Selected candidate, reused target": "Select, then reuse target",
        "Selected candidate, sealed target": "Select, then seal target",
        "Complete family, Holm correction": "Test full family with Holm",
    }
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Audit design & False validations & Rate & 95\\% Wilson interval \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{labels[row.Method]} & {int(row.FalseValidations):,}/{int(row.Trials):,} & "
            f"{row.FalseValidationRate:.3f} & "
            f"[{row.Wilson95Low:.3f}, {row.Wilson95High:.3f}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def write_shift_table(
    path: Path,
    summary_path: Path,
    detail_path: Path,
) -> None:
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    means = detail.pivot(
        index="Candidate",
        columns="Requirement",
        values="MeanRegret",
    )
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Proxy relation & $n$ & $\\Delta$AUC & Brier & Cost5x & Decision \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
        decision = (
            "Rel. validated"
            if row.Status == "Validated"
            else _status(row.Status)
        )
        lines.append(
            f"{row.Candidate} & {int(row.AuditN):,} & {row.AUCChange:+.3f} & "
            f"{means.loc[row.Candidate, 'Brier']:+.3f} & "
            f"{means.loc[row.Candidate, 'Cost5x']:+.3f} & "
            f"{decision} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def write_tabddpm_table(
    path: Path,
    summary_path: Path,
    detail_path: Path,
) -> None:
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    means = detail.pivot(
        index="Candidate",
        columns="Requirement",
        values="MeanRegret",
    )
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Dataset & $n$ & $\\Delta$AUC & Brier & Cost5x & Decision \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{_dataset(row.Dataset)} & {int(row.AuditN):,} & {row.AUCChange:+.3f} & "
            f"{means.loc[row.Candidate, 'Brier']:+.3f} & "
            f"{means.loc[row.Candidate, 'Cost5x']:+.3f} & "
            f"{_status(row.Status)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def write_repeated_aim_table(path: Path, input_path: Path) -> None:
    frame = pd.read_csv(input_path)
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        (
            "Dataset & $\\epsilon$ & Joint status & Individual status & "
            "$\\Delta$AUC mean $\\pm$ sd & Cost5x regret \\\\"
        ),
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        releases = int(row.Releases)
        joint_valid = int(row.JointValidated)
        joint_fail = int(row.JointViolations)
        individual_valid = int(row.IndividualValidated)
        individual_fail = int(row.IndividualViolations)
        lines.append(
            f"{_dataset(row.Dataset)} & {row.Epsilon:g} & "
            f"{joint_valid}/{releases - joint_valid - joint_fail}/{joint_fail} & "
            f"{individual_valid}/{releases - individual_valid - individual_fail}/"
            f"{individual_fail} & "
            f"{row.MeanAUCChange:+.3f} $\\pm$ {row.SDAUCChange:.3f} & "
            f"{row.MeanCost5xChange:+.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def write_privacy_table(
    path: Path,
    attack_path: Path,
    advantage_path: Path,
) -> None:
    attacks = pd.read_csv(attack_path)
    advantage = pd.read_csv(advantage_path).rename(columns={"Candidate": "Release"})
    best_indices = attacks.groupby(
        ["Generator", "Dataset"],
        sort=True,
    )["AUCMean"].idxmax()
    best = attacks.loc[best_indices].copy()
    state_counts = (
        advantage.assign(Status=advantage["Status"].map(_status))
        .merge(
            attacks[["Release", "Generator", "Dataset"]].drop_duplicates(),
            on="Release",
            how="left",
            validate="one_to_one",
        )
        .groupby(["Generator", "Dataset"])["Status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    lines = [
        "\\begin{tabular}{llrrrl}",
        "\\toprule",
        (
            "Generator & Dataset & Releases & Highest AUC & TPR@1\\% & "
            "Decision counts \\\\"
        ),
        "\\midrule",
    ]
    for row in best.sort_values(["Generator", "Dataset"]).itertuples(index=False):
        key = (row.Generator, row.Dataset)
        counts = state_counts.loc[key]
        total = int(counts.sum())
        validated = int(counts.get("Validated", 0))
        violations = int(counts.get("Violation", 0))
        unresolved = total - validated - violations
        lines.append(
            f"{row.Generator} & {_dataset(row.Dataset)} & {total} & "
            f"{row.AUCMean:.3f} ({row.Attack}) & {row.TPR1Mean:.3f} & "
            f"{validated}/{unresolved}/{violations} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def draw_attack_heatmap(path: Path, input_path: Path) -> None:
    frame = pd.read_csv(input_path)
    display = frame.copy()
    display["Row"] = display.apply(
        lambda row: (
            "TabDDPM"
            if row["Generator"] == "TabDDPM"
            else f"AIM eps {str(row['Setting']).replace('epsilon=', '')}"
        ),
        axis=1,
    )
    row_order = ["AIM eps 1", "AIM eps 5", "AIM eps 10", "TabDDPM"]
    attack_order = ["DCR", "density-only KDE", "DOMIAS-KDE", "Gen-LRA"]
    attack_labels = ["DCR", "Density", "DOMIAS", "Gen-LRA"]
    datasets = ["german_credit", "heart_disease", "taiwan_default"]
    figure = plt.figure(figsize=(7.2, 2.65))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        wspace=0.48,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])
    image = None
    for axis, dataset in zip(axes, datasets, strict=True):
        subset = display[display["Dataset"].eq(dataset)]
        matrix = subset.pivot(index="Row", columns="Attack", values="AUCMean")
        matrix = matrix.reindex(index=row_order, columns=attack_order)
        image = axis.imshow(
            matrix.to_numpy(),
            vmin=0.4,
            vmax=0.7,
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(_dataset(dataset), fontsize=9, pad=4)
        axis.set_xticks(
            np.arange(len(attack_labels)),
            attack_labels,
            rotation=28,
            ha="right",
        )
        axis.set_yticks(np.arange(len(row_order)), row_order)
        axis.tick_params(axis="both", labelsize=6.5)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.52 else "black",
                    fontsize=6.3,
                )
    if image is None:
        raise ValueError("Attack summary contains no registered datasets.")
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Attack AUC")
    colorbar.ax.tick_params(labelsize=7)
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.23,
        top=0.88,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the registered frontier-study paper assets."
    )
    parser.add_argument("--paper-root", default="paper/proxyguard")
    parser.add_argument(
        "--out-of-mechanism",
        default="outputs/proxyguard_out_of_mechanism/summary.csv",
    )
    parser.add_argument(
        "--target-reuse",
        default="outputs/proxyguard_target_reuse/summary.csv",
    )
    parser.add_argument(
        "--shifts",
        default="outputs/proxyguard_shift_audits",
    )
    parser.add_argument(
        "--tabddpm",
        default="outputs/proxyguard_tabddpm",
    )
    parser.add_argument(
        "--aim",
        default="outputs/proxyguard_repeated_aim",
    )
    parser.add_argument(
        "--attacks",
        default="outputs/proxyguard_privacy_attacks",
    )
    args = parser.parse_args()
    paper_root = Path(args.paper_root)
    tables = paper_root / "tables"
    figures = paper_root / "figs"
    write_out_of_mechanism_table(
        tables / "out_of_mechanism.tex",
        Path(args.out_of_mechanism),
    )
    write_target_reuse_table(
        tables / "target_reuse.tex",
        Path(args.target_reuse),
    )
    write_shift_table(
        tables / "shift_audits.tex",
        Path(args.shifts) / "candidate_summary.csv",
        Path(args.shifts) / "requirement_detail.csv",
    )
    write_tabddpm_table(
        tables / "tabddpm_audit.tex",
        Path(args.tabddpm) / "candidate_summary.csv",
        Path(args.tabddpm) / "requirement_detail.csv",
    )
    write_repeated_aim_table(
        tables / "repeated_aim.tex",
        Path(args.aim) / "mechanism_variability.csv",
    )
    write_privacy_table(
        tables / "privacy_attacks.tex",
        Path(args.attacks) / "attack_summary.csv",
        Path(args.attacks) / "attack_advantage_summary.csv",
    )
    draw_attack_heatmap(
        figures / "attack_heatmap.png",
        Path(args.attacks) / "attack_summary.csv",
    )


if __name__ == "__main__":
    main()
