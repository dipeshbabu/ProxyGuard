from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from proxyguard.shared_target import (
    project_cost_normalized_plan,
    recommend_cost_normalized_audit,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(settings: dict[str, Any], ratio: float, direct: float, named: float):
    return recommend_cost_normalized_audit(
        total_budget=float(settings["total_budget"]),
        target_record_cost=float(settings["target_record_cost"]),
        release_cost=float(settings["target_record_cost"]) * ratio,
        candidate_target_records=settings["candidate_target_records"],
        candidate_releases=settings["candidate_releases"],
        tolerances=settings["tolerances"],
        candidate_slacks=settings["candidate_slacks"],
        expected_direct_score_probability=direct,
        expected_named_recognition_probability=named,
        error_rate=float(settings["error_rate"]),
        mechanisms=int(settings["mechanisms"]),
        target_error_fractions=settings["target_error_fractions"],
        named_release_error_shares=settings["named_release_error_shares"],
    )


def run_sensitivity(registry_path: Path, output_root: Path) -> pd.DataFrame:
    expected = registry_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if _sha256(registry_path) != expected:
        raise ValueError("Cost-sensitivity registry digest mismatch.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity output: {output_root}")
    output_root.mkdir(parents=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    settings = registry["cost_planning"]
    true_direct = float(settings["true_direct_score_probability"])
    true_named = float(settings["true_named_recognition_probability"])
    rows: list[dict[str, Any]] = []
    for ratio_value in settings["release_to_target_cost_ratios"]:
        ratio = float(ratio_value)
        oracle = _plan(settings, ratio, true_direct, true_named)
        oracle_bound = project_cost_normalized_plan(
            oracle,
            tolerances=settings["tolerances"],
            direct_score_probability=true_direct,
            named_recognition_probability=true_named,
            error_rate=float(settings["error_rate"]),
            mechanisms=int(settings["mechanisms"]),
        )
        for direct_error_value in settings["pilot_probability_errors"]:
            for named_error_value in settings["pilot_probability_errors"]:
                direct_error = float(direct_error_value)
                named_error = float(named_error_value)
                assumed_direct = min(1.0, max(0.0, true_direct + direct_error))
                assumed_named = min(1.0, max(0.0, true_named + named_error))
                chosen = _plan(settings, ratio, assumed_direct, assumed_named)
                achieved = project_cost_normalized_plan(
                    chosen,
                    tolerances=settings["tolerances"],
                    direct_score_probability=true_direct,
                    named_recognition_probability=true_named,
                    error_rate=float(settings["error_rate"]),
                    mechanisms=int(settings["mechanisms"]),
                )
                rows.append(
                    {
                        "ReleaseToTargetCostRatio": ratio,
                        "DirectPilotError": direct_error,
                        "NamedPilotError": named_error,
                        "AssumedDirectProbability": assumed_direct,
                        "AssumedNamedProbability": assumed_named,
                        "ChosenMode": chosen.mode,
                        "ChosenTargetRecords": chosen.target_records,
                        "ChosenReleases": chosen.releases,
                        "AchievedProjectedBound": achieved,
                        "OracleMode": oracle.mode,
                        "OracleTargetRecords": oracle.target_records,
                        "OracleReleases": oracle.releases,
                        "OracleProjectedBound": oracle_bound,
                        "Regret": max(0.0, oracle_bound - achieved),
                    }
                )
    frame = pd.DataFrame(rows)
    frame_path = output_root / "pilot_sensitivity.csv"
    frame.to_csv(frame_path, index=False)
    summary = (
        frame.assign(ModeChanged=frame["ChosenMode"] != frame["OracleMode"])
        .groupby("ReleaseToTargetCostRatio", as_index=False)
        .agg(
            OracleMode=("OracleMode", "first"),
            OracleProjectedBound=("OracleProjectedBound", "first"),
            ModeChangeRate=("ModeChanged", "mean"),
            MeanRegret=("Regret", "mean"),
            MaximumRegret=("Regret", "max"),
        )
    )
    summary_path = output_root / "pilot_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "rows": len(frame),
        "sensitivity_sha256": _sha256(frame_path),
        "summary_sha256": _sha256(summary_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test cost-normalized audit planning.")
    parser.add_argument(
        "--registry", default="registries/proxyguard_cost_planning_sensitivity.json"
    )
    parser.add_argument(
        "--output-root", default="outputs/proxyguard_cost_planning_sensitivity"
    )
    args = parser.parse_args()
    print(run_sensitivity(Path(args.registry), Path(args.output_root)).to_string(index=False))


if __name__ == "__main__":
    main()
