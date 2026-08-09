from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from proxyguard.shared_target import recommend_cost_normalized_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cost_planning(registry_path: Path, output_root: Path) -> pd.DataFrame:
    sidecar = registry_path.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(registry_path) != expected:
        raise ValueError("Cost-planning registry digest mismatch.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite cost-planning output: {output_root}")
    output_root.mkdir(parents=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    settings = registry["cost_planning"]
    rows: list[dict[str, float | int | str | None]] = []
    for ratio in settings["release_to_target_cost_ratios"]:
        plan = recommend_cost_normalized_audit(
            total_budget=float(settings["total_budget"]),
            target_record_cost=float(settings["target_record_cost"]),
            release_cost=float(settings["target_record_cost"]) * float(ratio),
            candidate_target_records=settings["candidate_target_records"],
            candidate_releases=settings["candidate_releases"],
            tolerances=settings["tolerances"],
            candidate_slacks=settings["candidate_slacks"],
            expected_direct_score_probability=float(
                settings["expected_direct_score_probability"]
            ),
            expected_named_recognition_probability=float(
                settings["expected_named_recognition_probability"]
            ),
            error_rate=float(settings["error_rate"]),
            mechanisms=int(settings["mechanisms"]),
            target_error_fractions=settings["target_error_fractions"],
            named_release_error_shares=settings["named_release_error_shares"],
        )
        rows.append(
            {
                "ReleaseToTargetCostRatio": float(ratio),
                "RecommendedMode": plan.mode,
                "TargetRecords": plan.target_records,
                "Releases": plan.releases,
                "Slack": plan.slack,
                "ProjectedReliabilityLCB": plan.projected_reliability_lower_bound,
                "TotalCost": plan.total_cost,
                "TargetCost": plan.target_cost,
                "MechanismDrawCost": plan.release_cost,
            }
        )
    frame = pd.DataFrame(rows)
    output_path = output_root / "cost_normalized_plans.csv"
    frame.to_csv(output_path, index=False)
    manifest = {
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan ProxyGuard under a two-axis cost budget.")
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_cost_planning.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_cost_planning",
    )
    args = parser.parse_args()
    result = run_cost_planning(Path(args.registry), Path(args.output_root))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
