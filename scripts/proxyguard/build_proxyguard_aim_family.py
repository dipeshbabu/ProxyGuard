from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import RiskRequirement, audit_proxy_candidates  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_family_audit(
    run_directories: list[Path],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loss_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for run_directory in run_directories:
        loss_frames.append(pd.read_csv(run_directory / "paired_losses.csv"))
        diagnostic_frames.append(pd.read_csv(run_directory / "diagnostics.csv"))

    losses = pd.concat(loss_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    if losses["Candidate"].duplicated().all():
        raise ValueError("No candidate audit records were found.")
    if diagnostics["Candidate"].duplicated().any():
        duplicates = sorted(
            diagnostics.loc[diagnostics["Candidate"].duplicated(), "Candidate"].unique()
        )
        raise ValueError(f"Duplicate candidate diagnostics: {duplicates}")

    candidate_regrets = {
        candidate: {
            "Brier": frame["brier_regret"].to_numpy(),
            "Clipped log loss": frame["logloss_regret"].to_numpy(),
            "Cost5x": frame["cost5x_regret"].to_numpy(),
        }
        for candidate, frame in losses.groupby("Candidate", sort=True)
    }
    requirements = [
        RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
        RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
        RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
    ]
    audit = audit_proxy_candidates(
        candidate_regrets,
        requirements=requirements,
        alpha=alpha,
        bound_method="empirical_bernstein",
    )
    summary = audit.candidate_summary.merge(
        diagnostics,
        on="Candidate",
        how="left",
        validate="one_to_one",
    )
    return summary, audit.requirement_detail, diagnostics, losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine independently generated AIM releases into one registered family."
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated directories containing paired_losses.csv and diagnostics.csv.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-root", default="outputs/proxyguard_aim_audit")
    args = parser.parse_args()

    run_directories = [Path(value) for value in parse_csv(args.runs)]
    summary, detail, diagnostics, losses = build_family_audit(
        run_directories,
        alpha=args.alpha,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "candidate_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "runs": [str(path) for path in run_directories],
                "alpha": args.alpha,
                "candidate_count": int(len(summary)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[
            ["Candidate", "AuditN", "Status", "AUCChange", "SourceCost5x", "ProxyCost5x"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
