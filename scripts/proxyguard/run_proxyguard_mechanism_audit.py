from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import (  # noqa: E402
    MechanismAuditResult,
    RiskRequirement,
    audit_proxy_mechanisms,
)


REQUIREMENTS = [
    RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
]


def mechanism_name(candidate: str) -> str:
    parts = str(candidate).split("::")
    if len(parts) < 3 or not parts[1].startswith("aim_e"):
        raise ValueError(f"Unexpected AIM candidate name: {candidate!r}")
    return f"{parts[0]}::{parts[1]}"


def build_real_mechanism_audit(
    losses: pd.DataFrame,
    minimum_reliability: float,
    total_alpha: float,
    release_error_share: float,
    mechanism_count_mode: str = "holm",
    collective_dependence_verified: bool = False,
    bound_method: str = "empirical_bernstein",
) -> MechanismAuditResult:
    required_columns = {
        "Candidate",
        "brier_regret",
        "logloss_regret",
        "cost5x_regret",
    }
    missing = required_columns - set(losses)
    if missing:
        raise ValueError(f"Paired-loss file is missing columns: {sorted(missing)}")

    candidate_regrets = {
        str(candidate): {
            "Brier": frame["brier_regret"].to_numpy(),
            "Clipped log loss": frame["logloss_regret"].to_numpy(),
            "Cost5x": frame["cost5x_regret"].to_numpy(),
        }
        for candidate, frame in losses.groupby("Candidate", sort=True)
    }
    release_to_mechanism = {
        candidate: mechanism_name(candidate) for candidate in candidate_regrets
    }
    return audit_proxy_mechanisms(
        candidate_regrets,
        release_to_mechanism,
        requirements=REQUIREMENTS,
        minimum_reliability=minimum_reliability,
        total_alpha=total_alpha,
        release_error_share=release_error_share,
        mechanism_count_mode=mechanism_count_mode,
        collective_dependence_verified=collective_dependence_verified,
        bound_method=bound_method,
    )


def add_mechanism_diagnostics(
    mechanism_summary: pd.DataFrame,
    release_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = release_diagnostics.copy()
    diagnostics["Mechanism"] = diagnostics["Candidate"].map(mechanism_name)
    grouped = (
        diagnostics.groupby("Mechanism", as_index=False)
        .agg(
            Dataset=("Dataset", "first"),
            Epsilon=("Epsilon", "first"),
            MeanAUCChange=("AUCChange", "mean"),
            SDAUCChange=("AUCChange", "std"),
            MeanCost5xChange=("Cost5xChange", "mean"),
            SDCost5xChange=("Cost5xChange", "std"),
        )
    )
    return mechanism_summary.merge(
        grouped,
        on="Mechanism",
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply two-level ProxyGuard to the existing repeated AIM releases."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_mechanism_registry.json",
    )
    parser.add_argument(
        "--losses",
        default="outputs/proxyguard_repeated_aim/paired_losses.csv",
    )
    parser.add_argument(
        "--release-summary",
        default="outputs/proxyguard_repeated_aim/release_summary.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_mechanism_audit",
    )
    parser.add_argument(
        "--mechanism-count-mode",
        choices=("holm", "simes"),
        default="holm",
        help="Aggregate release evidence by release-level Holm certification or Simes count.",
    )
    parser.add_argument(
        "--collective-dependence-verified",
        action="store_true",
        help=(
            "Assert that Simes mode has independent audit batches or a "
            "registered PRDS justification."
        ),
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mechanism_settings = registry["mechanism_reliability"]
    real_settings = registry["real_release_reanalysis"]
    losses = pd.read_csv(args.losses)
    diagnostics = pd.read_csv(args.release_summary)
    primary_reliability = float(real_settings["primary_minimum_reliability"])
    sensitivity_reliability = float(
        mechanism_settings["sensitivity_minimum_reliability"]
    )
    common = {
        "total_alpha": float(mechanism_settings["total_alpha"]),
        "release_error_share": float(mechanism_settings["release_error_share"]),
    }

    primary = build_real_mechanism_audit(
        losses,
        minimum_reliability=primary_reliability,
        mechanism_count_mode=args.mechanism_count_mode,
        collective_dependence_verified=args.collective_dependence_verified,
        **common,
    )
    sensitivity = build_real_mechanism_audit(
        losses,
        minimum_reliability=sensitivity_reliability,
        mechanism_count_mode=args.mechanism_count_mode,
        collective_dependence_verified=args.collective_dependence_verified,
        **common,
    )
    primary_summary = add_mechanism_diagnostics(
        primary.mechanism_summary,
        diagnostics,
    )
    sensitivity_summary = add_mechanism_diagnostics(
        sensitivity.mechanism_summary,
        diagnostics,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    primary.release_audit.candidate_summary.to_csv(
        output_root / "release_summary.csv",
        index=False,
    )
    primary.release_audit.requirement_detail.to_csv(
        output_root / "release_requirement_detail.csv",
        index=False,
    )
    primary_summary.to_csv(
        output_root / "mechanism_summary_primary.csv",
        index=False,
    )
    sensitivity_summary.to_csv(
        output_root / "mechanism_summary_sensitivity.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "source_losses": args.losses,
                "source_release_summary": args.release_summary,
                "analysis_status": real_settings["analysis_status"],
                "primary_minimum_reliability": primary_reliability,
                "sensitivity_minimum_reliability": sensitivity_reliability,
                "mechanism_count_mode": args.mechanism_count_mode,
                "collective_dependence_verified": args.collective_dependence_verified,
                **common,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        primary_summary[
            [
                "Mechanism",
                "Releases",
                "ValidatedReleases",
                "DetectedReleaseViolations",
                "ReliabilityLCB",
                "ReliabilityUCB",
                "Status",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
