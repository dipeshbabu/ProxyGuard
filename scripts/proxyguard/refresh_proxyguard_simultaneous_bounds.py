from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
)
from scripts.proxyguard.run_proxyguard_privacy_attacks import (  # noqa: E402
    build_attack_advantage_audit,
)


SUMMARY_COLUMNS = {
    "Requirements",
    "AuditNMin",
    "CandidatePValue",
    "HolmAdjustedPValue",
    "Validated",
    "ViolationDetected",
    "Status",
    "JointFamilyStatus",
    "WorstUpperRegretBound",
    "IndividualReleaseStatus",
    "IndividualWorstUpperRegretBound",
}
REGRET_COLUMNS = {
    "Brier": "brier_regret",
    "Clipped log loss": "logloss_regret",
    "Cost5x": "cost5x_regret",
}


def _requirements(registry: dict) -> list[RiskRequirement]:
    return [
        RiskRequirement(
            name=item["name"],
            tolerance=float(item["tolerance"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            estimand=str(item.get("estimand", "relative_regret")),
        )
        for item in registry["risk_control"]["requirements"]
    ]


def _candidate_regrets(
    paired_losses: pd.DataFrame,
    requirements: list[RiskRequirement],
) -> dict[str, dict[str, pd.Series]]:
    return {
        candidate: {
            requirement.name: group[REGRET_COLUMNS[requirement.name]]
            for requirement in requirements
        }
        for candidate, group in paired_losses.groupby("Candidate", sort=False)
    }


def _diagnostic_columns(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in summary.columns
        if column == "Candidate" or column not in SUMMARY_COLUMNS
    ]
    return summary[columns]


def refresh_prediction_audit(
    output_root: Path,
    registry: dict,
    repeated_releases: bool,
) -> None:
    requirements = _requirements(registry)
    paired_losses = pd.read_csv(output_root / "paired_losses.csv")
    old_summary = pd.read_csv(output_root / (
        "release_summary.csv" if repeated_releases else "candidate_summary.csv"
    ))
    candidate_regrets = _candidate_regrets(paired_losses, requirements)
    joint = audit_proxy_candidates(
        candidate_regrets,
        requirements,
        alpha=float(registry["risk_control"]["alpha"]),
        bound_method=str(registry["risk_control"]["bound_method"]),
    )
    diagnostics = _diagnostic_columns(old_summary)

    if repeated_releases:
        summary = joint.candidate_summary.rename(
            columns={"Status": "JointFamilyStatus"}
        ).merge(diagnostics, on="Candidate", how="left", validate="one_to_one")
        individual_rows = []
        for candidate, regret_map in candidate_regrets.items():
            individual = audit_proxy_candidates(
                {candidate: regret_map},
                requirements,
                alpha=float(registry["risk_control"]["alpha"]),
                bound_method=str(registry["risk_control"]["bound_method"]),
            )
            individual_rows.append(
                individual.candidate_summary[
                    ["Candidate", "Status", "WorstUpperRegretBound"]
                ].rename(
                    columns={
                        "Status": "IndividualReleaseStatus",
                        "WorstUpperRegretBound": "IndividualWorstUpperRegretBound",
                    }
                )
            )
        summary = summary.merge(
            pd.concat(individual_rows, ignore_index=True),
            on="Candidate",
            how="left",
            validate="one_to_one",
        )
        summary.to_csv(output_root / "release_summary.csv", index=False)
    else:
        summary = joint.candidate_summary.merge(
            diagnostics,
            on="Candidate",
            how="left",
            validate="one_to_one",
        )
        summary.to_csv(output_root / "candidate_summary.csv", index=False)
    joint.requirement_detail.to_csv(output_root / "requirement_detail.csv", index=False)


def refresh_attack_audit(output_root: Path, registry: dict) -> None:
    scores = pd.read_csv(output_root / "attack_scores.csv")
    claim = registry["attack_suite"]["risk_controlled_claim"]
    summary, detail, diagnostics = build_attack_advantage_audit(
        scores,
        registry["attack_suite"],
        alpha=float(registry["risk_control"]["alpha"]),
        bound_method=str(claim["bound_method"]),
    )
    summary.to_csv(output_root / "attack_advantage_summary.csv", index=False)
    detail.to_csv(output_root / "attack_advantage_detail.csv", index=False)
    diagnostics.to_csv(output_root / "attack_advantage_diagnostics.csv", index=False)


def main() -> None:
    registry = json.loads(
        Path("registries/proxyguard_frontier_registry.json").read_text(
            encoding="utf-8"
        )
    )
    refresh_prediction_audit(
        Path("outputs/proxyguard_repeated_aim"),
        registry,
        repeated_releases=True,
    )
    refresh_prediction_audit(
        Path("outputs/proxyguard_shift_audits"),
        registry,
        repeated_releases=False,
    )
    refresh_prediction_audit(
        Path("outputs/proxyguard_tabddpm"),
        registry,
        repeated_releases=False,
    )
    refresh_attack_audit(
        Path("outputs/proxyguard_privacy_attacks"),
        registry,
    )
    print("Refreshed simultaneous bounds from frozen per-record artifacts.")


if __name__ == "__main__":
    main()
