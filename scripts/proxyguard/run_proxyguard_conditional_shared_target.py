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
    shared_target_conditional_witness_lower_bound,
)
from scripts.proxyguard.run_proxyguard_mechanism_study import (
    holm_release_counts,
)


METHODS = (
    "Named-release Holm",
    "Conditional shared-target",
    "Oracle release labels",
)


def _registry_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_losses(
    *,
    rng: np.random.Generator,
    releases: int,
    target_size: int,
    reliability: float,
    valid_risk_low: float,
    valid_risk_high: float,
    invalid_risk_low: float,
    invalid_risk_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = rng.random(releases) < reliability
    risks = np.where(
        valid,
        rng.uniform(valid_risk_low, valid_risk_high, size=releases),
        rng.uniform(invalid_risk_low, invalid_risk_high, size=releases),
    )
    target_uniforms = rng.random(target_size)
    losses = target_uniforms[None, :] < risks[:, None]
    return losses[:, :, None], valid


def _named_release_bound(
    losses: np.ndarray,
    *,
    tolerance: float,
    release_error_rate: float,
    mechanism_error_rate: float,
) -> float:
    releases, target_size, _ = losses.shape
    counts = losses[:, :, 0].sum(axis=1)
    pvalues = binom.cdf(counts, target_size, tolerance)
    recognized = int(
        holm_release_counts(
            pvalues.reshape(1, 1, releases),
            release_error_rate,
        )[0, 0]
    )
    return clopper_pearson_lower_bound(
        recognized,
        releases,
        mechanism_error_rate,
    )


def _simulate_cell(
    *,
    rng: np.random.Generator,
    settings: dict,
    releases: int,
    target_size: int,
    reliability: float,
) -> dict[str, float]:
    losses, valid = _draw_losses(
        rng=rng,
        releases=releases,
        target_size=target_size,
        reliability=reliability,
        valid_risk_low=float(settings["valid_risk_low"]),
        valid_risk_high=float(settings["valid_risk_high"]),
        invalid_risk_low=float(settings["invalid_risk_low"]),
        invalid_risk_high=float(settings["invalid_risk_high"]),
    )
    alpha = float(settings["total_alpha"])
    identified_fraction = float(
        settings["identified_release_error_fraction"]
    )
    named = _named_release_bound(
        losses,
        tolerance=float(settings["tolerance"]),
        release_error_rate=alpha * identified_fraction,
        mechanism_error_rate=alpha * (1.0 - identified_fraction),
    )
    conditional = shared_target_conditional_witness_lower_bound(
        losses,
        tolerances=float(settings["tolerance"]),
        slacks=float(settings["conditional_slack"]),
        error_rate=alpha,
        target_error_fraction=float(
            settings["conditional_target_error_fraction"]
        ),
    ).reliability_lower_bound
    oracle = clopper_pearson_lower_bound(
        int(valid.sum()),
        releases,
        alpha,
    )
    return {
        "Named-release Holm": named,
        "Conditional shared-target": conditional,
        "Oracle release labels": oracle,
    }


def run_study(registry: dict) -> pd.DataFrame:
    settings = registry["conditional_shared_target_study"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    eta0 = float(settings["minimum_reliability"])
    rows: list[dict[str, float | int | str | bool]] = []

    for releases in map(int, settings["release_counts"]):
        for target_size in map(int, settings["target_sizes"]):
            for reliability in map(float, settings["reliabilities"]):
                bounds = {
                    method: np.empty(repetitions, dtype=float)
                    for method in METHODS
                }
                for repetition in range(repetitions):
                    result = _simulate_cell(
                        rng=rng,
                        settings=settings,
                        releases=releases,
                        target_size=target_size,
                        reliability=reliability,
                    )
                    for method in METHODS:
                        bounds[method][repetition] = result[method]

                boundary = reliability == eta0
                for method in METHODS:
                    decisions = bounds[method] > eta0
                    rows.append(
                        {
                            "Method": method,
                            "Releases": releases,
                            "TargetN": target_size,
                            "TrueReliability": reliability,
                            "Boundary": boundary,
                            "ValidationRate": float(decisions.mean()),
                            "ValidationSE": float(
                                decisions.std(ddof=1)
                                / np.sqrt(repetitions)
                            ),
                            "MeanLowerBound": float(bounds[method].mean()),
                            "MedianLowerBound": float(
                                np.median(bounds[method])
                            ),
                            "Repetitions": repetitions,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the conditional shared-target reliability pilot."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_conditional_shared_target_pilot.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_conditional_shared_target_pilot",
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
