from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.run_proxyguard_mechanism_study import (  # noqa: E402
    exact_badness_pvalues,
    holm_rejections,
    outer_rejections,
    simulate_mechanism_setting,
)


INNER_METHODS = (
    "Uncorrected release IUT",
    "Inner Holm (ProxyGuard)",
    "Oracle release labels",
)


def _release_batch(
    rng: np.random.Generator,
    repetitions: int,
    mechanisms: int,
    releases: int,
    requirements: int,
    audit_size: int,
    tolerance: float,
    safe_risk: float,
    bad_release_risk: float,
    reliability: float,
    release_alpha: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    truly_good = rng.random((repetitions, mechanisms, releases)) < reliability
    probabilities = np.full(
        (repetitions, mechanisms, releases, requirements),
        safe_risk,
        dtype=float,
    )
    probabilities[..., 0] = np.where(
        truly_good,
        safe_risk,
        bad_release_risk,
    )
    counts = rng.binomial(audit_size, probabilities)
    component_pvalues = exact_badness_pvalues(counts, audit_size, tolerance)
    release_pvalues = component_pvalues.max(axis=3)
    uncorrected = release_pvalues <= release_alpha
    corrected = holm_rejections(
        release_pvalues.reshape(repetitions, mechanisms * releases),
        release_alpha,
    ).reshape(repetitions, mechanisms, releases)
    return truly_good, {
        "Uncorrected release IUT": uncorrected,
        "Inner Holm (ProxyGuard)": corrected,
        "Oracle release labels": truly_good,
    }


def simulate_near_boundary_setting(
    *,
    rng: np.random.Generator,
    repetitions: int,
    mechanisms: int,
    releases: int,
    requirements: int,
    audit_size: int,
    tolerance: float,
    safe_risk: float,
    bad_release_risk: float,
    boundary_reliability: float,
    valid_reliability: float,
    total_alpha: float,
    release_error_share: float,
    batch_size: int = 250,
) -> pd.DataFrame:
    release_alpha = total_alpha * release_error_share
    mechanism_alpha = total_alpha - release_alpha
    false_release_totals = {method: 0.0 for method in INNER_METHODS}
    false_mechanism_totals = {method: 0.0 for method in INNER_METHODS}
    power_totals = {method: 0.0 for method in INNER_METHODS}

    completed = 0
    while completed < repetitions:
        current = min(batch_size, repetitions - completed)
        boundary_good, boundary_labels = _release_batch(
            rng,
            current,
            mechanisms,
            releases,
            requirements,
            audit_size,
            tolerance,
            safe_risk,
            bad_release_risk,
            boundary_reliability,
            release_alpha,
        )
        _, valid_labels = _release_batch(
            rng,
            current,
            mechanisms,
            releases,
            requirements,
            audit_size,
            tolerance,
            safe_risk,
            bad_release_risk,
            valid_reliability,
            release_alpha,
        )
        for method in INNER_METHODS:
            labels = boundary_labels[method]
            false_release_totals[method] += float(
                np.logical_and(labels, ~boundary_good)
                .reshape(current, -1)
                .any(axis=1)
                .sum()
            )
            mechanism_decisions = outer_rejections(
                labels.sum(axis=2),
                releases,
                boundary_reliability,
                mechanism_alpha,
            )
            false_mechanism_totals[method] += float(
                mechanism_decisions.any(axis=1).sum()
            )
            valid_decisions = outer_rejections(
                valid_labels[method].sum(axis=2),
                releases,
                boundary_reliability,
                mechanism_alpha,
            )
            power_totals[method] += float(valid_decisions.mean(axis=1).sum())
        completed += current

    rows = []
    for method in INNER_METHODS:
        rows.append(
            {
                "BadReleaseRisk": bad_release_risk,
                "Method": method,
                "AuditN": audit_size,
                "Releases": releases,
                "Mechanisms": mechanisms,
                "Requirements": requirements,
                "MinimumReliability": boundary_reliability,
                "ReleaseAlpha": release_alpha,
                "MechanismAlpha": mechanism_alpha,
                "Repetitions": repetitions,
                "FalseReleaseRecognitionFWER": (
                    false_release_totals[method] / repetitions
                ),
                "FalseMechanismValidation": (
                    false_mechanism_totals[method] / repetitions
                ),
                "MechanismValidationPower": power_totals[method] / repetitions,
            }
        )
    return pd.DataFrame(rows)


def run_near_boundary_study(registry: dict) -> pd.DataFrame:
    settings = registry["near_boundary_inner_correction"]
    rng = np.random.default_rng(int(settings["seed"]))
    frames = []
    for bad_risk in settings["bad_release_risks"]:
        frames.append(
            simulate_near_boundary_setting(
                rng=rng,
                repetitions=int(settings["repetitions"]),
                mechanisms=int(settings["mechanisms"]),
                releases=int(settings["releases"]),
                requirements=int(settings["requirements"]),
                audit_size=int(settings["audit_size"]),
                tolerance=float(settings["tolerance"]),
                safe_risk=float(settings["safe_release_risk"]),
                bad_release_risk=float(bad_risk),
                boundary_reliability=float(settings["minimum_reliability"]),
                valid_reliability=float(settings["valid_reliability"]),
                total_alpha=float(settings["total_alpha"]),
                release_error_share=float(settings["release_error_share"]),
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_alpha_allocation_study(registry: dict) -> pd.DataFrame:
    settings = registry["alpha_allocation"]
    rng = np.random.default_rng(int(settings["seed"]))
    frames = []
    for release_error_share in settings["release_error_shares"]:
        result = simulate_mechanism_setting(
            rng=rng,
            repetitions=int(settings["repetitions"]),
            mechanisms=int(settings["mechanisms"]),
            releases=int(settings["releases"]),
            requirements=int(settings["requirements"]),
            audit_size=int(settings["audit_size"]),
            tolerance=float(settings["tolerance"]),
            safe_risk=float(settings["safe_release_risk"]),
            bad_release_risk=float(settings["bad_release_risk"]),
            boundary_reliability=float(settings["minimum_reliability"]),
            valid_reliability=float(settings["valid_reliability"]),
            total_alpha=float(settings["total_alpha"]),
            release_error_share=float(release_error_share),
        )
        frame = result[result["Method"].eq("Two-level ProxyGuard")].copy()
        frame["ReleaseErrorShare"] = float(release_error_share)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered mechanism-level revision experiments."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_mechanism_revision_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_mechanism_revision_study",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    near_boundary = run_near_boundary_study(registry)
    allocation = run_alpha_allocation_study(registry)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    near_boundary.to_csv(output_root / "near_boundary_inner_correction.csv", index=False)
    allocation.to_csv(output_root / "alpha_allocation.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256_file": str(registry_path.with_suffix(".sha256")),
                "near_boundary_rows": len(near_boundary),
                "alpha_allocation_rows": len(allocation),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(near_boundary.to_string(index=False))
    print()
    print(
        allocation[
            [
                "ReleaseErrorShare",
                "ReleaseAlpha",
                "MechanismAlpha",
                "FalseMechanismValidation",
                "MechanismValidationPower",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
