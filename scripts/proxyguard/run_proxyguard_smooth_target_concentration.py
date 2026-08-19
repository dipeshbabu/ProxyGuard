from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proxyguard.core import clopper_pearson_lower_bound, clopper_pearson_upper_bound
from proxyguard.shared_target import (
    shared_target_conditional_mean_lower_bound,
    shared_target_smooth_conditional_mean_lower_bound,
)


HARD_MARKOV = "Hard score + Markov"
SMOOTH_MARKOV = "Ramp score + Markov"
SMOOTH_CONCENTRATION = "Ramp score + target concentration"
ORACLE = "Oracle release labels"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_registry(path: Path) -> None:
    digest_path = path.with_suffix(".sha256")
    if not digest_path.exists():
        return
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    if _sha256(path) != expected:
        raise ValueError(f"Registry digest mismatch: {path}")


def _draw_release_means(
    *,
    rng: np.random.Generator,
    releases: int,
    target_records: int,
    reliability: float,
    valid_risk: float,
    invalid_risk: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = rng.random(releases) < reliability
    target_uniforms = np.sort(rng.random(target_records))
    valid_mean = np.searchsorted(target_uniforms, valid_risk, side="right")
    invalid_mean = np.searchsorted(target_uniforms, invalid_risk, side="right")
    means = np.where(valid, valid_mean, invalid_mean) / target_records
    return means[:, None], valid


def _evaluate(
    *,
    rng: np.random.Generator,
    settings: dict[str, Any],
    releases: int,
    target_records: int,
    ramp_width: float,
    reliability: float,
) -> dict[str, float]:
    means, valid = _draw_release_means(
        rng=rng,
        releases=releases,
        target_records=target_records,
        reliability=reliability,
        valid_risk=float(settings["valid_risk"]),
        invalid_risk=float(settings["invalid_risk"]),
    )
    common = {
        "target_records": target_records,
        "tolerances": [float(settings["tolerance"])],
        "slacks": [float(settings["slack"])],
        "lower_bounds": [float(settings["lower_bound"])],
        "upper_bounds": [float(settings["upper_bound"])],
        "error_rate": float(settings["total_alpha"]),
        "target_error_fraction": float(settings["target_error_fraction"]),
        "mechanisms": 1,
    }
    hard = shared_target_conditional_mean_lower_bound(means, **common)
    smooth = shared_target_smooth_conditional_mean_lower_bound(
        means,
        ramp_widths=[ramp_width],
        integration_bins=int(settings["integration_bins"]),
        **common,
    )
    return {
        HARD_MARKOV: hard.reliability_lower_bound,
        SMOOTH_MARKOV: max(
            0.0,
            smooth.conditional_score_lower_bound - smooth.markov_contamination_allowance,
        ),
        SMOOTH_CONCENTRATION: smooth.reliability_lower_bound,
        ORACLE: clopper_pearson_lower_bound(
            int(valid.sum()),
            releases,
            float(settings["total_alpha"]),
        ),
        "SmoothScoreMean": smooth.conditional_score_mean,
        "SmoothInvalidCeiling": smooth.invalid_release_score_ceiling,
        "SmoothTargetRadius": smooth.target_concentration_radius,
        "SmoothMarkovAllowance": smooth.markov_contamination_allowance,
        "SmoothConcentrationAllowance": smooth.target_contamination_allowance,
    }


def _run_cells(
    registry: dict[str, Any],
    cells: list[tuple[int, int, float, float]],
) -> pd.DataFrame:
    settings = registry["smooth_target_concentration"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    methods = [HARD_MARKOV, SMOOTH_MARKOV, SMOOTH_CONCENTRATION, ORACLE]
    rows: list[dict[str, float | int | str]] = []
    eta0 = float(settings["minimum_reliability"])

    for releases, target_records, ramp_width, reliability in cells:
        bounds = {method: np.empty(repetitions, dtype=float) for method in methods}
        diagnostics: dict[str, float] = {}
        score_means = np.empty(repetitions, dtype=float)
        for repetition in range(repetitions):
            result = _evaluate(
                rng=rng,
                settings=settings,
                releases=releases,
                target_records=target_records,
                ramp_width=ramp_width,
                reliability=reliability,
            )
            for method in methods:
                bounds[method][repetition] = result[method]
            score_means[repetition] = result["SmoothScoreMean"]
            diagnostics = {
                name: result[name]
                for name in (
                    "SmoothInvalidCeiling",
                    "SmoothTargetRadius",
                    "SmoothMarkovAllowance",
                    "SmoothConcentrationAllowance",
                )
            }
        for method in methods:
            validations = bounds[method] > eta0
            events = int(validations.sum())
            rows.append(
                {
                    "Method": method,
                    "Releases": releases,
                    "TargetN": target_records,
                    "RampWidth": ramp_width,
                    "TrueReliability": reliability,
                    "ValidationRate": float(validations.mean()),
                    "ValidationEvents": events,
                    "ValidationTrials": repetitions,
                    "MCOneSidedUpper95": clopper_pearson_upper_bound(
                        events,
                        repetitions,
                        0.05,
                    ),
                    "MeanLowerBound": float(bounds[method].mean()),
                    "MedianLowerBound": float(np.median(bounds[method])),
                    "MeanSmoothScore": float(score_means.mean()),
                    **diagnostics,
                }
            )
    return pd.DataFrame(rows)


def run_registry(registry: dict[str, Any]) -> pd.DataFrame:
    settings = registry["smooth_target_concentration"]
    if registry["stage"] == "pilot":
        cells = [
            (int(releases), int(target), float(ramp), float(reliability))
            for releases in settings["release_counts"]
            for target in settings["target_sizes"]
            for ramp in settings["ramp_widths"]
            for reliability in settings["true_reliabilities"]
        ]
    elif registry["stage"] == "confirmation":
        cells = [
            (
                int(settings["selected_release_count"]),
                int(settings["selected_target_records"]),
                float(settings["selected_ramp_width"]),
                float(reliability),
            )
            for reliability in settings["true_reliabilities"]
        ]
    else:
        raise ValueError("stage must be 'pilot' or 'confirmation'.")
    return _run_cells(registry, cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare smooth target concentration with expectation-only corrections."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_smooth_target_concentration_pilot.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_smooth_target_concentration_pilot",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    _check_registry(registry_path)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    summary = run_registry(registry)
    output_root.mkdir(parents=True)
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "summary_sha256": _sha256(summary_path),
        "analysis_status": registry["analysis_status"],
        "stage": registry["stage"],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
