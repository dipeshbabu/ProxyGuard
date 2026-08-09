from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from proxyguard.core import (
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
)
from proxyguard.shared_target import shared_target_conditional_mean_lower_bound


SCORE_ONLY = "Uncorrected score-only"
DIRECT = "Corrected direct shared-target"
ORACLE = "Oracle release labels"


def _registry_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_shared_target_means(
    *,
    rng: np.random.Generator,
    releases: int,
    target_records: int,
    reliability: float,
    valid_risk_low: float,
    valid_risk_high: float,
    invalid_risk_low: float,
    invalid_risk_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw threshold losses for independent releases on one target sample."""

    valid = rng.random(releases) < reliability
    risks = np.empty(releases, dtype=float)
    risks[valid] = rng.uniform(valid_risk_low, valid_risk_high, int(valid.sum()))
    risks[~valid] = rng.uniform(
        invalid_risk_low,
        invalid_risk_high,
        int((~valid).sum()),
    )
    target_uniforms = np.sort(rng.random(target_records))
    empirical_means = np.searchsorted(
        target_uniforms,
        risks,
        side="right",
    ) / target_records
    return empirical_means[:, None], valid


def _simulate_cell(
    *,
    rng: np.random.Generator,
    settings: dict,
    releases: int,
    target_records: int,
    normalized_slack: float,
    true_reliability: float | None = None,
) -> dict[str, float]:
    reliability = (
        float(settings["minimum_reliability"])
        if true_reliability is None
        else float(true_reliability)
    )
    means, valid = _draw_shared_target_means(
        rng=rng,
        releases=releases,
        target_records=target_records,
        reliability=reliability,
        valid_risk_low=float(settings["valid_risk_low"]),
        valid_risk_high=float(settings["valid_risk_high"]),
        invalid_risk_low=float(settings["invalid_risk_low"]),
        invalid_risk_high=float(settings["invalid_risk_high"]),
    )
    lower = float(settings["lower_bound"])
    upper = float(settings["upper_bound"])
    tolerance = float(settings["tolerance"])
    slack = normalized_slack * (upper - lower)
    direct = shared_target_conditional_mean_lower_bound(
        means,
        target_records=target_records,
        tolerances=[tolerance],
        slacks=[slack],
        lower_bounds=[lower],
        upper_bounds=[upper],
        error_rate=float(settings["total_alpha"]),
        target_error_fraction=float(settings["target_error_fraction"]),
        mechanisms=1,
    )
    scores = means[:, 0] <= tolerance - slack
    return {
        SCORE_ONLY: direct.conditional_score_lower_bound,
        DIRECT: direct.reliability_lower_bound,
        ORACLE: clopper_pearson_lower_bound(
            int(valid.sum()),
            releases,
            float(settings["total_alpha"]),
        ),
        "score_mean": float(scores.mean()),
        "invalid_score_mass": float(np.mean((~valid) & scores)),
        "invalid_release_score_ceiling": direct.invalid_release_score_ceiling,
        "target_contamination_allowance": direct.target_contamination_allowance,
    }


def _summarize(
    *,
    method: str,
    bounds: np.ndarray,
    score_means: np.ndarray,
    invalid_score_masses: np.ndarray,
    invalid_ceiling: float,
    contamination_allowance: float,
    settings: dict,
    releases: int,
    target_records: int,
    normalized_slack: float,
    true_reliability: float,
) -> dict[str, float | int | str]:
    eta0 = float(settings["minimum_reliability"])
    decisions = bounds > eta0
    events = int(decisions.sum())
    trials = int(decisions.size)
    rate = float(decisions.mean())
    standard_error = (
        float(decisions.std(ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0
    )
    return {
        "Method": method,
        "Releases": releases,
        "TargetN": target_records,
        "TrueReliability": true_reliability,
        "NormalizedSlack": normalized_slack,
        "ValidationRate": rate,
        "ValidationSE": standard_error,
        "ValidationEvents": events,
        "ValidationTrials": trials,
        "MCOneSidedUpper95": clopper_pearson_upper_bound(events, trials, 0.05),
        "MeanLowerBound": float(bounds.mean()),
        "MedianLowerBound": float(np.median(bounds)),
        "MeanScoreRate": float(score_means.mean()),
        "MeanInvalidScoreMass": float(invalid_score_masses.mean()),
        "InvalidReleaseScoreCeiling": invalid_ceiling,
        "TargetContaminationAllowance": contamination_allowance,
    }


def _run_cells(
    registry: dict,
    cells: list[tuple[int, int, float, float]],
) -> pd.DataFrame:
    settings = registry["false_pass_diagnostic"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    rows: list[dict[str, float | int | str]] = []

    for releases, target_records, normalized_slack, true_reliability in cells:
        bounds = {
            SCORE_ONLY: np.empty(repetitions, dtype=float),
            DIRECT: np.empty(repetitions, dtype=float),
            ORACLE: np.empty(repetitions, dtype=float),
        }
        score_means = np.empty(repetitions, dtype=float)
        invalid_score_masses = np.empty(repetitions, dtype=float)
        invalid_ceiling = np.nan
        contamination_allowance = np.nan
        for repetition in range(repetitions):
            result = _simulate_cell(
                rng=rng,
                settings=settings,
                releases=releases,
                target_records=target_records,
                normalized_slack=normalized_slack,
                true_reliability=true_reliability,
            )
            for method in bounds:
                bounds[method][repetition] = float(result[method])
            score_means[repetition] = float(result["score_mean"])
            invalid_score_masses[repetition] = float(result["invalid_score_mass"])
            invalid_ceiling = float(result["invalid_release_score_ceiling"])
            contamination_allowance = float(
                result["target_contamination_allowance"]
            )
        for method, method_bounds in bounds.items():
            rows.append(
                _summarize(
                    method=method,
                    bounds=method_bounds,
                    score_means=score_means,
                    invalid_score_masses=invalid_score_masses,
                    invalid_ceiling=invalid_ceiling,
                    contamination_allowance=contamination_allowance,
                    settings=settings,
                    releases=releases,
                    target_records=target_records,
                    normalized_slack=normalized_slack,
                    true_reliability=true_reliability,
                )
            )
    return pd.DataFrame(rows)


def run_pilot(registry: dict) -> pd.DataFrame:
    settings = registry["false_pass_diagnostic"]
    cells = [
        (int(releases), int(target_records), float(slack), float(reliability))
        for releases in settings["release_counts"]
        for target_records in settings["target_sizes"]
        for slack in settings["normalized_slacks"]
        for reliability in settings.get(
            "true_reliabilities",
            [settings["minimum_reliability"]],
        )
    ]
    return _run_cells(registry, cells)


def run_confirmation(registry: dict) -> pd.DataFrame:
    settings = registry["false_pass_diagnostic"]
    cells = [
        (
            int(settings["selected_release_count"]),
            int(settings["selected_target_records"]),
            float(settings["selected_normalized_slack"]),
            float(reliability),
        )
        for reliability in settings.get(
            "true_reliabilities",
            [settings["minimum_reliability"]],
        )
    ]
    return _run_cells(registry, cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show why a shared-target score-frequency bound needs an invalid-"
            "release contamination correction."
        )
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_false_pass_diagnostic_pilot.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_false_pass_diagnostic_pilot",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    stage = str(registry["stage"])
    if stage == "pilot":
        summary = run_pilot(registry)
    elif stage == "confirmation":
        summary = run_confirmation(registry)
    else:
        raise ValueError("stage must be 'pilot' or 'confirmation'.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256": _registry_hash(registry_path),
                "analysis_status": registry["analysis_status"],
                "stage": stage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
