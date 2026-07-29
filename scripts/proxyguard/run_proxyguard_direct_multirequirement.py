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
from proxyguard.shared_target import (
    hybrid_reliability_lower_bound,
    shared_target_conditional_mean_lower_bound,
)
from scripts.proxyguard.run_proxyguard_mechanism_study import (
    holm_release_counts,
)


NAMED = "Named-release Holm"
DIRECT = "Direct shared-target"
HYBRID = "Preregistered hybrid"
ORACLE = "Oracle release labels"


def _registry_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empirical_bernstein_pvalues_from_summaries(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    target_records: int,
    tolerances: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized inversion of the empirical Bernstein upper bound."""

    gaps = tolerances.reshape(1, -1) - means
    widths = (upper_bounds - lower_bounds).reshape(1, -1)
    root_term = np.sqrt(2.0 * variances / target_records)
    linear_term = 7.0 * widths / (3.0 * (target_records - 1))
    discriminant = np.square(root_term) + 4.0 * linear_term * np.maximum(gaps, 0.0)
    sqrt_log = (
        -root_term + np.sqrt(np.maximum(discriminant, 0.0))
    ) / (2.0 * linear_term)
    log_term = np.square(np.maximum(sqrt_log, 0.0))
    pvalues = np.minimum(1.0, 2.0 * np.exp(-log_term))
    pvalues[gaps <= 0.0] = 1.0
    return pvalues


def _draw_summaries(
    *,
    rng: np.random.Generator,
    settings: dict,
    releases: int,
    target_records: int,
    reliability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw correlated bounded continuous losses on one shared target."""

    lower = np.asarray(settings["lower_bounds"], dtype=float)
    upper = np.asarray(settings["upper_bounds"], dtype=float)
    tolerances = np.asarray(settings["tolerances"], dtype=float)
    widths = upper - lower
    normalized_tolerances = (tolerances - lower) / widths
    requirements = tolerances.size

    valid = rng.random(releases) < reliability
    base_margins = rng.uniform(
        float(settings["valid_margin_low"]),
        float(settings["valid_margin_high"]),
        size=releases,
    )
    normalized_margins = base_margins[:, None] + rng.uniform(
        -float(settings["margin_jitter"]),
        float(settings["margin_jitter"]),
        size=(releases, requirements),
    )
    normalized_margins = np.maximum(normalized_margins, 1e-6)
    invalid_indices = np.flatnonzero(~valid)
    if invalid_indices.size:
        failed_requirements = rng.integers(
            0,
            requirements,
            size=invalid_indices.size,
        )
        normalized_margins[invalid_indices, failed_requirements] = -rng.uniform(
            float(settings["invalid_margin_low"]),
            float(settings["invalid_margin_high"]),
            size=invalid_indices.size,
        )
    normalized_means = normalized_tolerances.reshape(1, -1) - normalized_margins

    shared = rng.uniform(-1.0, 1.0, size=target_records)
    requirement_noise = rng.uniform(
        -1.0,
        1.0,
        size=(target_records, requirements),
    )
    interaction_noise = rng.uniform(
        -1.0,
        1.0,
        size=(target_records, requirements),
    )
    release_coefficients = rng.uniform(-1.0, 1.0, size=(releases, requirements))
    noise_model = str(settings.get("noise_model", "fixed_scale"))

    normalized_sample_means = np.empty_like(normalized_means)
    normalized_sample_variances = np.empty_like(normalized_means)
    for requirement in range(requirements):
        coefficient = release_coefficients[:, requirement]
        if noise_model == "fixed_scale":
            base = (
                float(settings["shared_noise_scale"]) * shared
                + float(settings["requirement_noise_scale"])
                * requirement_noise[:, requirement]
            )
            interaction = (
                float(settings["interaction_noise_scale"])
                * interaction_noise[:, requirement]
            )
            normalized_sample_means[:, requirement] = (
                normalized_means[:, requirement]
                + float(base.mean())
                + coefficient * float(interaction.mean())
            )
            base_variance = float(base.var(ddof=1))
            interaction_variance = float(interaction.var(ddof=1))
            covariance = float(np.cov(base, interaction, ddof=1)[0, 1])
            normalized_sample_variances[:, requirement] = (
                base_variance
                + 2.0 * coefficient * covariance
                + np.square(coefficient) * interaction_variance
            )
        elif noise_model == "bounded_amplitude":
            shared_weight = float(settings["shared_noise_weight"])
            requirement_weight = float(settings["requirement_noise_weight"])
            interaction_weight = float(settings["interaction_noise_weight"])
            if shared_weight + requirement_weight + interaction_weight > 1.0:
                raise ValueError("bounded-amplitude noise weights must sum to at most one.")
            base = (
                shared_weight * shared
                + requirement_weight * requirement_noise[:, requirement]
            )
            interaction = (
                interaction_weight * interaction_noise[:, requirement]
            )
            factor_mean = (
                float(base.mean())
                + coefficient * float(interaction.mean())
            )
            base_variance = float(base.var(ddof=1))
            interaction_variance = float(interaction.var(ddof=1))
            covariance = float(np.cov(base, interaction, ddof=1)[0, 1])
            factor_variance = (
                base_variance
                + 2.0 * coefficient * covariance
                + np.square(coefficient) * interaction_variance
            )
            amplitude = float(settings["amplitude_fraction"]) * np.minimum(
                normalized_means[:, requirement],
                1.0 - normalized_means[:, requirement],
            )
            normalized_sample_means[:, requirement] = (
                normalized_means[:, requirement] + amplitude * factor_mean
            )
            normalized_sample_variances[:, requirement] = (
                np.square(amplitude) * factor_variance
            )
        elif noise_model == "smoothed_bernoulli":
            continuous_fraction = float(settings["continuous_fraction"])
            if not 0.0 < continuous_fraction < 1.0:
                raise ValueError("continuous_fraction must lie strictly between zero and one.")
            thresholds = (
                normalized_means[:, requirement] - continuous_fraction / 2.0
            ) / (1.0 - continuous_fraction)
            if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
                raise RuntimeError(
                    "The smoothed-Bernoulli threshold left the unit interval."
                )
            uniforms = (shared + 1.0) / 2.0
            continuous_noise = (
                requirement_noise[:, requirement] + 1.0
            ) / 2.0
            order = np.argsort(uniforms)
            sorted_uniforms = uniforms[order]
            sorted_noise = continuous_noise[order]
            cumulative_noise = np.concatenate(
                ([0.0], np.cumsum(sorted_noise))
            )
            ranks = np.searchsorted(
                sorted_uniforms,
                thresholds,
                side="left",
            )
            bernoulli_means = ranks / target_records
            cross_means = cumulative_noise[ranks] / target_records
            noise_mean = float(continuous_noise.mean())
            noise_second_moment = float(np.square(continuous_noise).mean())
            normalized_sample_means[:, requirement] = (
                (1.0 - continuous_fraction) * bernoulli_means
                + continuous_fraction * noise_mean
            )
            second_moments = (
                np.square(1.0 - continuous_fraction) * bernoulli_means
                + 2.0
                * continuous_fraction
                * (1.0 - continuous_fraction)
                * cross_means
                + np.square(continuous_fraction) * noise_second_moment
            )
            normalized_sample_variances[:, requirement] = (
                target_records
                / (target_records - 1)
                * (
                    second_moments
                    - np.square(normalized_sample_means[:, requirement])
                )
            )
        else:
            raise ValueError(f"Unknown noise_model: {noise_model}")

    sample_means = lower.reshape(1, -1) + widths.reshape(1, -1) * normalized_sample_means
    sample_variances = (
        np.square(widths.reshape(1, -1)) * normalized_sample_variances
    )
    if np.any(sample_means <= lower.reshape(1, -1)) or np.any(
        sample_means >= upper.reshape(1, -1)
    ):
        raise RuntimeError("The registered noise design left the bounded-loss domain.")
    return sample_means, sample_variances, valid


def _named_lower_bound(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    target_records: int,
    settings: dict,
    total_error: float,
    release_error_fraction: float,
    mechanisms: int,
) -> float:
    tolerances = np.asarray(settings["tolerances"], dtype=float)
    lower = np.asarray(settings["lower_bounds"], dtype=float)
    upper = np.asarray(settings["upper_bounds"], dtype=float)
    component_pvalues = _empirical_bernstein_pvalues_from_summaries(
        means,
        variances,
        target_records=target_records,
        tolerances=tolerances,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    release_pvalues = component_pvalues.max(axis=1)
    release_alpha = total_error * release_error_fraction / mechanisms
    mechanism_alpha = total_error * (1.0 - release_error_fraction) / mechanisms
    recognized = int(
        holm_release_counts(
            release_pvalues.reshape(1, 1, -1),
            release_alpha,
        )[0, 0]
    )
    return clopper_pearson_lower_bound(
        recognized,
        means.shape[0],
        mechanism_alpha,
    )


def _direct_lower_bound(
    means: np.ndarray,
    *,
    target_records: int,
    settings: dict,
    total_error: float,
    target_error_fraction: float,
    normalized_slack: float,
    mechanisms: int,
) -> float:
    lower = np.asarray(settings["lower_bounds"], dtype=float)
    upper = np.asarray(settings["upper_bounds"], dtype=float)
    slacks = normalized_slack * (upper - lower)
    return shared_target_conditional_mean_lower_bound(
        means,
        target_records=target_records,
        tolerances=settings["tolerances"],
        slacks=slacks,
        lower_bounds=lower,
        upper_bounds=upper,
        error_rate=total_error,
        target_error_fraction=target_error_fraction,
        mechanisms=mechanisms,
    ).reliability_lower_bound


def _simulate_configuration(
    *,
    rng: np.random.Generator,
    settings: dict,
    releases: int,
    target_records: int,
    reliability: float,
    named_fraction: float,
    direct_fraction: float,
    normalized_slack: float,
    include_hybrid: bool,
) -> dict[str, np.ndarray]:
    mechanisms = int(settings["mechanisms"])
    total_error = float(settings["total_alpha"])
    results = {
        NAMED: np.empty(mechanisms, dtype=float),
        DIRECT: np.empty(mechanisms, dtype=float),
        ORACLE: np.empty(mechanisms, dtype=float),
    }
    if include_hybrid:
        results[HYBRID] = np.empty(mechanisms, dtype=float)

    for mechanism in range(mechanisms):
        means, variances, valid = _draw_summaries(
            rng=rng,
            settings=settings,
            releases=releases,
            target_records=target_records,
            reliability=reliability,
        )
        results[NAMED][mechanism] = _named_lower_bound(
            means,
            variances,
            target_records=target_records,
            settings=settings,
            total_error=total_error,
            release_error_fraction=named_fraction,
            mechanisms=mechanisms,
        )
        results[DIRECT][mechanism] = _direct_lower_bound(
            means,
            target_records=target_records,
            settings=settings,
            total_error=total_error,
            target_error_fraction=direct_fraction,
            normalized_slack=normalized_slack,
            mechanisms=mechanisms,
        )
        results[ORACLE][mechanism] = clopper_pearson_lower_bound(
            int(valid.sum()),
            releases,
            total_error / mechanisms,
        )
        if include_hybrid:
            named_half = _named_lower_bound(
                means,
                variances,
                target_records=target_records,
                settings=settings,
                total_error=total_error / 2.0,
                release_error_fraction=named_fraction,
                mechanisms=mechanisms,
            )
            direct_half = _direct_lower_bound(
                means,
                target_records=target_records,
                settings=settings,
                total_error=total_error / 2.0,
                target_error_fraction=direct_fraction,
                normalized_slack=normalized_slack,
                mechanisms=mechanisms,
            )
            results[HYBRID][mechanism] = hybrid_reliability_lower_bound(
                named_half,
                direct_half,
                identified_error_rate=total_error / 2.0,
                shared_target_error_rate=total_error / 2.0,
                total_error_rate=total_error,
            )
    return results


def run_pilot(registry: dict) -> pd.DataFrame:
    settings = registry["direct_multirequirement_study"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    eta0 = float(settings["minimum_reliability"])
    releases = int(settings["planning_release_count"])
    target_records = int(settings["planning_target_records"])
    reliabilities = [float(value) for value in settings["planning_reliabilities"]]
    named_grid = [float(value) for value in settings["named_release_error_fractions"]]
    direct_grid = [
        (
            float(fraction),
            float(slack),
        )
        for fraction in settings["direct_target_error_fractions"]
        for slack in settings["direct_normalized_slacks"]
    ]
    rows: list[dict[str, float | int | str | bool]] = []

    for reliability in reliabilities:
        named_bounds = {
            fraction: np.empty((repetitions, int(settings["mechanisms"])))
            for fraction in named_grid
        }
        direct_bounds = {
            configuration: np.empty((repetitions, int(settings["mechanisms"])))
            for configuration in direct_grid
        }
        for repetition in range(repetitions):
            mechanism_draws = [
                _draw_summaries(
                    rng=rng,
                    settings=settings,
                    releases=releases,
                    target_records=target_records,
                    reliability=reliability,
                )
                for _ in range(int(settings["mechanisms"]))
            ]
            for fraction in named_grid:
                for mechanism, (means, variances, _valid) in enumerate(mechanism_draws):
                    named_bounds[fraction][repetition, mechanism] = _named_lower_bound(
                        means,
                        variances,
                        target_records=target_records,
                        settings=settings,
                        total_error=float(settings["total_alpha"]),
                        release_error_fraction=fraction,
                        mechanisms=int(settings["mechanisms"]),
                    )
            for fraction, slack in direct_grid:
                for mechanism, (means, _variances, _valid) in enumerate(mechanism_draws):
                    direct_bounds[(fraction, slack)][
                        repetition,
                        mechanism,
                    ] = _direct_lower_bound(
                        means,
                        target_records=target_records,
                        settings=settings,
                        total_error=float(settings["total_alpha"]),
                        target_error_fraction=fraction,
                        normalized_slack=slack,
                        mechanisms=int(settings["mechanisms"]),
                    )

        boundary = reliability == eta0
        for fraction, bounds in named_bounds.items():
            decisions = bounds > eta0
            rows.append(
                _summary_row(
                    method=NAMED,
                    reliability=reliability,
                    boundary=boundary,
                    decisions=decisions,
                    bounds=bounds,
                    repetitions=repetitions,
                    releases=releases,
                    target_records=target_records,
                    named_fraction=fraction,
                )
            )
        for (fraction, slack), bounds in direct_bounds.items():
            decisions = bounds > eta0
            rows.append(
                _summary_row(
                    method=DIRECT,
                    reliability=reliability,
                    boundary=boundary,
                    decisions=decisions,
                    bounds=bounds,
                    repetitions=repetitions,
                    releases=releases,
                    target_records=target_records,
                    direct_fraction=fraction,
                    normalized_slack=slack,
                )
            )
    return pd.DataFrame(rows)


def _summary_row(
    *,
    method: str,
    reliability: float,
    boundary: bool,
    decisions: np.ndarray,
    bounds: np.ndarray,
    repetitions: int,
    releases: int,
    target_records: int,
    named_fraction: float | None = None,
    direct_fraction: float | None = None,
    normalized_slack: float | None = None,
) -> dict[str, float | int | str | bool]:
    family_decisions = decisions.any(axis=1)
    rate_observations = family_decisions if boundary else decisions.reshape(-1)
    validation_events = int(rate_observations.sum())
    validation_trials = int(rate_observations.size)
    rate = float(rate_observations.mean())
    standard_error = float(
        rate_observations.std(ddof=1) / np.sqrt(rate_observations.size)
    )
    return {
        "Method": method,
        "Releases": releases,
        "TargetN": target_records,
        "TrueReliability": reliability,
        "Boundary": boundary,
        "FamilywiseBoundaryRate": float(family_decisions.mean()) if boundary else np.nan,
        "ValidationRate": rate,
        "ValidationSE": standard_error,
        "ValidationEvents": validation_events,
        "ValidationTrials": validation_trials,
        "MCOneSidedUpper95": clopper_pearson_upper_bound(
            validation_events,
            validation_trials,
            0.05,
        ),
        "MeanLowerBound": float(bounds.mean()),
        "MedianLowerBound": float(np.median(bounds)),
        "NamedReleaseErrorFraction": named_fraction,
        "DirectTargetErrorFraction": direct_fraction,
        "DirectNormalizedSlack": normalized_slack,
        "Repetitions": repetitions,
    }


def run_confirmation(registry: dict) -> pd.DataFrame:
    settings = registry["direct_multirequirement_study"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    eta0 = float(settings["minimum_reliability"])
    named_fraction = float(settings["selected_named_release_error_fraction"])
    direct_fraction = float(settings["selected_direct_target_error_fraction"])
    normalized_slack = float(settings["selected_direct_normalized_slack"])
    rows: list[dict[str, float | int | str | bool]] = []

    for releases in map(int, settings["release_counts"]):
        for target_records in map(int, settings["target_sizes"]):
            for reliability in map(float, settings["reliabilities"]):
                methods = (NAMED, DIRECT, HYBRID, ORACLE)
                bounds = {
                    method: np.empty(
                        (
                            repetitions,
                            int(settings["mechanisms"]),
                        ),
                        dtype=float,
                    )
                    for method in methods
                }
                for repetition in range(repetitions):
                    result = _simulate_configuration(
                        rng=rng,
                        settings=settings,
                        releases=releases,
                        target_records=target_records,
                        reliability=reliability,
                        named_fraction=named_fraction,
                        direct_fraction=direct_fraction,
                        normalized_slack=normalized_slack,
                        include_hybrid=True,
                    )
                    for method in methods:
                        bounds[method][repetition] = result[method]
                boundary = reliability == eta0
                for method in methods:
                    decisions = bounds[method] > eta0
                    rows.append(
                        _summary_row(
                            method=method,
                            reliability=reliability,
                            boundary=boundary,
                            decisions=decisions,
                            bounds=bounds[method],
                            repetitions=repetitions,
                            releases=releases,
                            target_records=target_records,
                            named_fraction=named_fraction if method in (NAMED, HYBRID) else None,
                            direct_fraction=direct_fraction if method in (DIRECT, HYBRID) else None,
                            normalized_slack=(
                                normalized_slack if method in (DIRECT, HYBRID) else None
                            ),
                        )
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the correlated multi-requirement shared-target study."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_direct_multirequirement_pilot.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_direct_multirequirement_pilot",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = str(registry["stage"])
    if stage == "pilot":
        summary = run_pilot(registry)
    elif stage == "confirmation":
        summary = run_confirmation(registry)
    else:
        raise ValueError("stage must be 'pilot' or 'confirmation'.")
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
