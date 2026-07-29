from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from proxyguard.core import clopper_pearson_lower_bound
from proxyguard.shared_target import (
    hybrid_reliability_lower_bound,
    shared_target_conditional_witness_lower_bound,
    shared_target_tensor_polynomial_lower_bound,
    shared_target_witness_lower_bound,
    witness_reliability_lower_from_mean,
)
from scripts.proxyguard.run_proxyguard_mechanism_study import (
    holm_release_counts,
    release_collective_fisher_counts,
    release_collective_simes_counts,
)


METHODS = (
    "Named-release Holm (shared target)",
    "Tail-Simes (matched independent target)",
    "Tail-Fisher (matched independent target)",
    "Direct witness (shared target)",
    "Conditional witness (shared target)",
    "Direct tensor polynomial (shared target)",
    "Hybrid (shared target)",
    "Oracle release labels",
)


def _registry_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_release_risks(
    *,
    rng: np.random.Generator,
    mechanisms: int,
    releases: int,
    requirements: int,
    reliability: float,
    tolerance: float,
    valid_risk_low: float,
    valid_risk_high: float,
    invalid_excess_low: float,
    invalid_excess_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = rng.random((mechanisms, releases)) < reliability
    risks = rng.uniform(
        valid_risk_low,
        valid_risk_high,
        size=(mechanisms, releases, requirements),
    )
    invalid_mechanism, invalid_release = np.where(~valid)
    failing_requirement = rng.integers(
        0,
        requirements,
        size=invalid_mechanism.size,
    )
    risks[
        invalid_mechanism,
        invalid_release,
        failing_requirement,
    ] = tolerance + rng.uniform(
        invalid_excess_low,
        invalid_excess_high,
        size=invalid_mechanism.size,
    )
    return risks, valid


def _candidate_pvalues(counts: np.ndarray, audit_size: int, tolerance: float) -> np.ndarray:
    requirement_pvalues = binom.cdf(counts, audit_size, tolerance)
    return requirement_pvalues.max(axis=-1)


def _tensor_design_points(settings: dict) -> np.ndarray:
    rng = np.random.default_rng(int(settings["tensor_design_seed"]))
    risks, _ = _draw_release_risks(
        rng=rng,
        mechanisms=1,
        releases=int(settings["tensor_design_points"]),
        requirements=int(settings["requirements"]),
        reliability=float(settings["tensor_design_reliability"]),
        tolerance=float(settings["tolerance"]),
        valid_risk_low=float(settings["valid_risk_low"]),
        valid_risk_high=float(settings["valid_risk_high"]),
        invalid_excess_low=float(settings["invalid_excess_low"]),
        invalid_excess_high=float(settings["invalid_excess_high"]),
    )
    return risks[0]


def _identified_bounds(
    pvalues: np.ndarray,
    *,
    releases: int,
    mechanisms: int,
    release_error: float,
    mechanism_error: float,
) -> np.ndarray:
    counts = holm_release_counts(
        pvalues[None, :, :],
        release_error / mechanisms,
    )[0]
    return np.asarray(
        [
            clopper_pearson_lower_bound(
                int(count),
                releases,
                mechanism_error / mechanisms,
            )
            for count in counts
        ],
        dtype=float,
    )


def _collective_bounds(
    pvalues: np.ndarray,
    *,
    releases: int,
    mechanisms: int,
    release_error: float,
    mechanism_error: float,
    method: str,
) -> np.ndarray:
    shaped = pvalues[None, :, :]
    if method == "simes":
        counts = release_collective_simes_counts(
            shaped,
            release_error / mechanisms,
        )[0]
    elif method == "fisher":
        counts = release_collective_fisher_counts(
            shaped,
            release_error / mechanisms,
        )[0]
    else:
        raise ValueError("method must be 'simes' or 'fisher'.")
    return np.asarray(
        [
            clopper_pearson_lower_bound(
                int(count),
                releases,
                mechanism_error / mechanisms,
            )
            for count in counts
        ],
        dtype=float,
    )


def _simulate_repetition(
    *,
    rng: np.random.Generator,
    settings: dict,
    reliability: float,
    repetition: int,
) -> dict[str, np.ndarray]:
    mechanisms = int(settings["mechanisms"])
    releases = int(settings["releases"])
    requirements = int(settings["requirements"])
    target_records = int(settings["target_records"])
    tolerance = float(settings["tolerance"])
    total_alpha = float(settings["total_alpha"])
    tensor_design_points = _tensor_design_points(settings)

    risks, valid = _draw_release_risks(
        rng=rng,
        mechanisms=mechanisms,
        releases=releases,
        requirements=requirements,
        reliability=reliability,
        tolerance=tolerance,
        valid_risk_low=float(settings["valid_risk_low"]),
        valid_risk_high=float(settings["valid_risk_high"]),
        invalid_excess_low=float(settings["invalid_excess_low"]),
        invalid_excess_high=float(settings["invalid_excess_high"]),
    )

    shared_uniforms = rng.random((target_records, requirements))
    shared_losses = (
        shared_uniforms[None, None, :, :]
        < risks[:, :, None, :]
    )
    shared_counts = shared_losses.sum(axis=2)
    shared_pvalues = _candidate_pvalues(
        shared_counts,
        target_records,
        tolerance,
    )
    identified = _identified_bounds(
        shared_pvalues,
        releases=releases,
        mechanisms=mechanisms,
        release_error=total_alpha / 2.0,
        mechanism_error=total_alpha / 2.0,
    )

    independent_size = max(1, target_records // releases)
    independent_counts = rng.binomial(
        independent_size,
        risks,
    )
    independent_pvalues = _candidate_pvalues(
        independent_counts,
        independent_size,
        tolerance,
    )
    simes = _collective_bounds(
        independent_pvalues,
        releases=releases,
        mechanisms=mechanisms,
        release_error=total_alpha / 2.0,
        mechanism_error=total_alpha / 2.0,
        method="simes",
    )
    fisher = _collective_bounds(
        independent_pvalues,
        releases=releases,
        mechanisms=mechanisms,
        release_error=total_alpha / 2.0,
        mechanism_error=total_alpha / 2.0,
        method="fisher",
    )

    direct = np.empty(mechanisms, dtype=float)
    conditional = np.empty(mechanisms, dtype=float)
    direct_tensor = np.empty(mechanisms, dtype=float)
    hybrid = np.empty(mechanisms, dtype=float)
    hybrid_identified = _identified_bounds(
        shared_pvalues,
        releases=releases,
        mechanisms=mechanisms,
        release_error=total_alpha / 4.0,
        mechanism_error=total_alpha / 4.0,
    )
    for mechanism_index in range(mechanisms):
        direct_result = shared_target_witness_lower_bound(
            shared_losses[mechanism_index],
            tolerances=tolerance,
            slacks=float(settings["witness_slack"]),
            block_size=int(settings["witness_block_size"]),
            ramp_widths=float(settings["witness_ramp_width"]),
            error_rate=total_alpha,
            mechanisms=mechanisms,
            block_seed=int(settings["block_seed"]) + repetition,
        )
        direct[mechanism_index] = direct_result.reliability_lower_bound
        conditional[mechanism_index] = (
            shared_target_conditional_witness_lower_bound(
                shared_losses[mechanism_index],
                tolerances=tolerance,
                slacks=float(settings["witness_slack"]),
                ramp_widths=float(settings["conditional_ramp_width"]),
                error_rate=total_alpha,
                target_error_fraction=float(
                    settings["conditional_target_error_fraction"]
                ),
                mechanisms=mechanisms,
            ).reliability_lower_bound
        )
        direct_tensor[mechanism_index] = (
            shared_target_tensor_polynomial_lower_bound(
                shared_losses[mechanism_index],
                tolerances=tolerance,
                margins=float(settings["tensor_margin"]),
                degree=int(settings["tensor_degree"]),
                coefficient_floor=float(settings["tensor_coefficient_floor"]),
                design_points=tensor_design_points,
                error_rate=total_alpha,
                mechanisms=mechanisms,
                block_seed=int(settings["block_seed"]) + repetition,
            ).reliability_lower_bound
        )
        direct_for_hybrid, _, _ = witness_reliability_lower_from_mean(
            direct_result.witness_mean,
            releases=releases,
            target_blocks=direct_result.target_blocks,
            invalid_release_witness_ceiling=(
                direct_result.invalid_release_witness_ceiling
            ),
            error_rate=total_alpha / 2.0,
            mechanisms=mechanisms,
        )
        hybrid[mechanism_index] = hybrid_reliability_lower_bound(
            hybrid_identified[mechanism_index],
            direct_for_hybrid,
            identified_error_rate=total_alpha / 2.0,
            shared_target_error_rate=total_alpha / 2.0,
            total_error_rate=total_alpha,
        )

    oracle = np.asarray(
        [
            clopper_pearson_lower_bound(
                int(valid[mechanism_index].sum()),
                releases,
                total_alpha / mechanisms,
            )
            for mechanism_index in range(mechanisms)
        ],
        dtype=float,
    )
    return {
        "Named-release Holm (shared target)": identified,
        "Tail-Simes (matched independent target)": simes,
        "Tail-Fisher (matched independent target)": fisher,
        "Direct witness (shared target)": direct,
        "Conditional witness (shared target)": conditional,
        "Direct tensor polynomial (shared target)": direct_tensor,
        "Hybrid (shared target)": hybrid,
        "Oracle release labels": oracle,
    }


def run_study(registry: dict) -> pd.DataFrame:
    settings = registry["shared_target_study"]
    rng = np.random.default_rng(int(settings["seed"]))
    eta0 = float(settings["minimum_reliability"])
    reliability_values = [eta0, *map(float, settings["power_reliabilities"])]
    repetitions = int(settings["repetitions"])
    mechanisms = int(settings["mechanisms"])
    rows: list[dict[str, float | int | str | bool]] = []

    for reliability in reliability_values:
        decisions = {
            method: np.empty((repetitions, mechanisms), dtype=bool)
            for method in METHODS
        }
        lower_bounds = {
            method: np.empty((repetitions, mechanisms), dtype=float)
            for method in METHODS
        }
        for repetition in range(repetitions):
            bounds = _simulate_repetition(
                rng=rng,
                settings=settings,
                reliability=reliability,
                repetition=repetition,
            )
            for method in METHODS:
                lower_bounds[method][repetition] = bounds[method]
                decisions[method][repetition] = bounds[method] > eta0

        boundary = reliability == eta0
        for method in METHODS:
            family_decision = decisions[method].any(axis=1).astype(float)
            per_mechanism_decision = decisions[method].mean(axis=1)
            rows.append(
                {
                    "Method": method,
                    "TrueReliability": reliability,
                    "Boundary": boundary,
                    "FamilywiseValidationRate": float(family_decision.mean()),
                    "FamilywiseValidationSE": float(
                        family_decision.std(ddof=1) / np.sqrt(repetitions)
                    ),
                    "MechanismValidationPower": float(
                        per_mechanism_decision.mean()
                    ),
                    "MechanismValidationPowerSE": float(
                        per_mechanism_decision.std(ddof=1)
                        / np.sqrt(repetitions)
                    ),
                    "MeanLowerBound": float(lower_bounds[method].mean()),
                    "MedianLowerBound": float(
                        np.median(lower_bounds[method])
                    ),
                    "Repetitions": repetitions,
                    "Mechanisms": mechanisms,
                    "Releases": int(settings["releases"]),
                    "SharedTargetN": int(settings["target_records"]),
                    "IndependentTargetPerRelease": max(
                        1,
                        int(settings["target_records"])
                        // int(settings["releases"]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared-target direct reliability study."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_shared_target_pilot.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_shared_target_pilot",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_study(registry)
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256": _registry_hash(registry_path),
                "analysis_status": registry["analysis_status"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
