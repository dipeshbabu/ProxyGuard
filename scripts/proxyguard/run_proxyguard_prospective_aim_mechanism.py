from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.run_proxyguard_mechanism_audit import (  # noqa: E402
    add_mechanism_diagnostics,
    build_real_mechanism_audit,
)
from scripts.proxyguard.run_proxyguard_repeated_aim import run_repeated_aim  # noqa: E402


def registered_mechanism_settings(registry: dict) -> dict[str, float | str]:
    settings = registry["prospective_aim_mechanism"]
    return {
        "minimum_reliability": float(settings["minimum_reliability"]),
        "total_alpha": float(settings["total_alpha"]),
        "release_error_share": float(settings["release_error_share"]),
        "analysis_status": str(settings["analysis_status"]),
    }


def registered_mechanism_release_indices(amendment: dict) -> list[int]:
    values = amendment["prospective_aim_mechanism"]["mechanism_release_indices"]
    indices = [int(value) for value in values]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Mechanism release indices must be nonempty and unique.")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prospectively registered AIM mechanism replication."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_mechanism_revision_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_prospective_aim_mechanism",
    )
    parser.add_argument(
        "--amendment",
        default=(
            "registries/"
            "proxyguard_mechanism_revision_amendment_20260726.json"
        ),
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    amendment_path = Path(args.amendment)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    release_summary, requirement_detail, variability, losses = run_repeated_aim(
        registry,
        output_root,
    )
    release_summary.to_csv(output_root / "release_summary.csv", index=False)
    requirement_detail.to_csv(
        output_root / "release_requirement_detail_initial.csv",
        index=False,
    )
    variability.to_csv(output_root / "release_variability.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)

    settings = registered_mechanism_settings(registry)
    mechanism_release_indices = registered_mechanism_release_indices(amendment)
    registered_release_count = int(
        registry["modern_generators"]["aim"]["releases_per_cell"]
    )
    if not set(mechanism_release_indices).issubset(
        range(1, registered_release_count + 1)
    ):
        raise ValueError("Amended mechanism indices exceed the registered releases.")
    mechanism_losses = losses[
        losses["Release"].isin(mechanism_release_indices)
    ].copy()
    mechanism_release_summary = release_summary[
        release_summary["Release"].isin(mechanism_release_indices)
    ].copy()
    if mechanism_losses["Release"].nunique() != len(mechanism_release_indices):
        raise RuntimeError("One or more amended mechanism releases are missing.")
    audit = build_real_mechanism_audit(
        mechanism_losses,
        minimum_reliability=float(settings["minimum_reliability"]),
        total_alpha=float(settings["total_alpha"]),
        release_error_share=float(settings["release_error_share"]),
    )
    mechanism_summary = add_mechanism_diagnostics(
        audit.mechanism_summary,
        mechanism_release_summary,
    )
    audit.release_audit.candidate_summary.to_csv(
        output_root / "two_level_release_summary.csv",
        index=False,
    )
    audit.release_audit.requirement_detail.to_csv(
        output_root / "two_level_requirement_detail.csv",
        index=False,
    )
    mechanism_summary.to_csv(output_root / "mechanism_summary.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256_file": str(registry_path.with_suffix(".sha256")),
                "amendment": str(amendment_path),
                "amendment_sha256_file": str(
                    amendment_path.with_suffix(".sha256")
                ),
                "mechanism_release_indices": mechanism_release_indices,
                "analysis_status": settings["analysis_status"],
                "registered_aim": registry["modern_generators"]["aim"],
                "mechanism_claim": registry["prospective_aim_mechanism"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        mechanism_summary[
            [
                "Mechanism",
                "Releases",
                "ValidatedReleases",
                "DetectedReleaseViolations",
                "ReliabilityLCB",
                "ReliabilityUCB",
                "Status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
