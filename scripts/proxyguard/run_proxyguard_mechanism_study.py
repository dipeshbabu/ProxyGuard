from __future__ import annotations

import argparse
import json
from math import ceil, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


MECHANISM_METHODS = (
    "Plug-in release fraction",
    "Point rule + binomial",
    "Per-release IUT + binomial",
    "Two-level ProxyGuard",
    "Oracle release labels",
)


def holm_rejections(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Apply Holm's procedure to each row of a two-dimensional array."""

    if pvalues.ndim != 2:
        raise ValueError("pvalues must have shape (repetitions, hypotheses).")
    repetitions, hypotheses = pvalues.shape
    order = np.argsort(pvalues, axis=1, kind="stable")
    ordered = np.take_along_axis(pvalues, order, axis=1)
    thresholds = alpha / np.arange(hypotheses, 0, -1)
    passes = ordered <= thresholds
    rejected_ordered = np.cumprod(passes.astype(int), axis=1).astype(bool)
    rejected = np.zeros_like(rejected_ordered)
    rows = np.arange(repetitions)[:, None]
    rejected[rows, order] = rejected_ordered
    return rejected


def exact_badness_pvalues(
    counts: np.ndarray,
    audit_size: int,
    tolerance: float,
) -> np.ndarray:
    """Exact lower-tail p-values for H0: Bernoulli risk >= tolerance."""

    return binom.cdf(counts, audit_size, tolerance)


def release_validation_counts(
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
) -> dict[str, np.ndarray]:
    """Simulate realized releases and return recognized-good counts."""

    truly_good = rng.random((repetitions, mechanisms, releases)) < reliability
    probabilities = np.full(
        (repetitions, mechanisms, releases, requirements),
        safe_risk,
        dtype=float,
    )
    probabilities[..., 0] = np.where(truly_good, safe_risk, bad_release_risk)
    counts = rng.binomial(audit_size, probabilities)
    component_pvalues = exact_badness_pvalues(counts, audit_size, tolerance)
    release_pvalues = component_pvalues.max(axis=3)

    point_validated = (counts / audit_size <= tolerance).all(axis=3)
    per_release_validated = release_pvalues <= release_alpha
    family_validated = holm_rejections(
        release_pvalues.reshape(repetitions, mechanisms * releases),
        release_alpha,
    ).reshape(repetitions, mechanisms, releases)

    return {
        "Plug-in release fraction": point_validated.sum(axis=2),
        "Point rule + binomial": point_validated.sum(axis=2),
        "Per-release IUT + binomial": per_release_validated.sum(axis=2),
        "Two-level ProxyGuard": family_validated.sum(axis=2),
        "Oracle release labels": truly_good.sum(axis=2),
    }


def outer_rejections(
    recognized_good: np.ndarray,
    releases: int,
    minimum_reliability: float,
    alpha: float,
) -> np.ndarray:
    pvalues = binom.sf(recognized_good - 1, releases, minimum_reliability)
    return holm_rejections(pvalues, alpha)


def simulate_mechanism_setting(
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
) -> pd.DataFrame:
    release_alpha = total_alpha * release_error_share
    mechanism_alpha = total_alpha - release_alpha
    boundary_counts = release_validation_counts(
        rng,
        repetitions,
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
    valid_counts = release_validation_counts(
        rng,
        repetitions,
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

    rows: list[dict[str, float | int | str]] = []
    for method in MECHANISM_METHODS:
        if method == "Plug-in release fraction":
            boundary_rejections = (
                boundary_counts[method] / releases >= boundary_reliability
            )
            valid_rejections = valid_counts[method] / releases >= boundary_reliability
        else:
            outer_alpha = (
                mechanism_alpha
                if method in {"Per-release IUT + binomial", "Two-level ProxyGuard"}
                else total_alpha
            )
            boundary_rejections = outer_rejections(
                boundary_counts[method],
                releases,
                boundary_reliability,
                outer_alpha,
            )
            valid_rejections = outer_rejections(
                valid_counts[method],
                releases,
                boundary_reliability,
                outer_alpha,
            )
        false_by_repetition = boundary_rejections.any(axis=1).astype(float)
        power_by_repetition = valid_rejections.mean(axis=1)
        rows.append(
            {
                "Method": method,
                "AuditN": audit_size,
                "Releases": releases,
                "Mechanisms": mechanisms,
                "Requirements": requirements,
                "MinimumReliability": boundary_reliability,
                "ValidReliability": valid_reliability,
                "TotalAlpha": total_alpha,
                "ReleaseAlpha": release_alpha,
                "MechanismAlpha": mechanism_alpha,
                "Repetitions": repetitions,
                "FalseMechanismValidation": float(false_by_repetition.mean()),
                "FalseMechanismValidationSE": float(
                    false_by_repetition.std(ddof=1) / np.sqrt(repetitions)
                ),
                "MechanismValidationPower": float(power_by_repetition.mean()),
                "MechanismValidationPowerSE": float(
                    power_by_repetition.std(ddof=1) / np.sqrt(repetitions)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_mechanism_study(registry: dict) -> pd.DataFrame:
    settings = registry["mechanism_reliability"]
    rng = np.random.default_rng(int(settings["seed"]))
    frames: list[pd.DataFrame] = []
    for audit_size in settings["audit_sizes"]:
        for releases in settings["release_counts"]:
            frames.append(
                simulate_mechanism_setting(
                    rng=rng,
                    repetitions=int(settings["repetitions"]),
                    mechanisms=int(settings["mechanisms"]),
                    releases=int(releases),
                    requirements=int(settings["requirements"]),
                    audit_size=int(audit_size),
                    tolerance=float(settings["tolerance"]),
                    safe_risk=float(settings["safe_release_risk"]),
                    bad_release_risk=float(settings["bad_release_risk"]),
                    boundary_reliability=float(settings["boundary_reliability"]),
                    valid_reliability=float(settings["valid_reliability"]),
                    total_alpha=float(settings["total_alpha"]),
                    release_error_share=float(settings["release_error_share"]),
                )
            )
    return pd.concat(frames, ignore_index=True)


def run_adaptive_study(registry: dict) -> pd.DataFrame:
    settings = registry["adaptive_search"]
    rng = np.random.default_rng(int(settings["seed"]))
    repetitions = int(settings["repetitions"])
    requirements = int(settings["requirements"])
    audit_size = int(settings["audit_size"])
    tolerance = float(settings["tolerance"])
    total_alpha = float(settings["total_alpha"])
    maximum_rounds = max(int(value) for value in settings["rounds"])

    boundary_probabilities = np.full(
        (maximum_rounds, requirements),
        float(settings["nonbinding_risk"]),
    )
    boundary_probabilities[:, 0] = float(settings["boundary_risk"])
    counts = rng.binomial(
        audit_size,
        boundary_probabilities,
        size=(repetitions, maximum_rounds, requirements),
    )
    pvalues = exact_badness_pvalues(counts, audit_size, tolerance).max(axis=2)
    indices = np.arange(1, maximum_rounds + 1, dtype=float)
    spending = 6.0 * total_alpha / (np.pi**2 * indices**2)

    rows: list[dict[str, float | int | str]] = []
    for rounds in (int(value) for value in settings["rounds"]):
        for method, thresholds in (
            ("Fixed alpha each round", np.full(rounds, total_alpha)),
            ("Quadratic alpha spending", spending[:rounds]),
        ):
            false_by_repetition = (pvalues[:, :rounds] <= thresholds).any(axis=1).astype(float)
            rows.append(
                {
                    "Analysis": "False validation",
                    "Method": method,
                    "Round": rounds,
                    "Rate": float(false_by_repetition.mean()),
                    "SE": float(false_by_repetition.std(ddof=1) / np.sqrt(repetitions)),
                    "AuditN": audit_size,
                    "Repetitions": repetitions,
                }
            )

    valid_probabilities = np.full(requirements, float(settings["valid_risk"]))
    valid_counts = rng.binomial(
        audit_size,
        valid_probabilities,
        size=(repetitions, len(settings["valid_arrival_rounds"]), requirements),
    )
    valid_pvalues = exact_badness_pvalues(
        valid_counts,
        audit_size,
        tolerance,
    ).max(axis=2)
    for column, arrival in enumerate(
        int(value) for value in settings["valid_arrival_rounds"]
    ):
        for method, threshold in (
            ("Fixed alpha each round", total_alpha),
            ("Quadratic alpha spending", float(spending[arrival - 1])),
        ):
            detected = (valid_pvalues[:, column] <= threshold).astype(float)
            rows.append(
                {
                    "Analysis": "Valid-candidate power",
                    "Method": method,
                    "Round": arrival,
                    "Rate": float(detected.mean()),
                    "SE": float(detected.std(ddof=1) / np.sqrt(repetitions)),
                    "AuditN": audit_size,
                    "Repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def build_planning_table(registry: dict) -> pd.DataFrame:
    settings = registry["mechanism_reliability"]
    mechanism_alpha = float(settings["total_alpha"]) * (
        1.0 - float(settings["release_error_share"])
    )
    rows = []
    for minimum_reliability in (
        float(settings["primary_minimum_reliability"]),
        float(settings["sensitivity_minimum_reliability"]),
    ):
        for mechanisms in (1, int(settings["mechanisms"]), 9):
            threshold = mechanism_alpha / mechanisms
            releases = ceil(log(threshold) / log(minimum_reliability))
            rows.append(
                {
                    "MinimumReliability": minimum_reliability,
                    "Mechanisms": mechanisms,
                    "MechanismAlpha": mechanism_alpha,
                    "AllRecognizedGoodReleasesNeeded": releases,
                }
            )
    return pd.DataFrame(rows)


def plot_mechanism_study(summary: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "Plug-in release fraction": "#c51b7d",
        "Point rule + binomial": "#777777",
        "Per-release IUT + binomial": "#d95f02",
        "Two-level ProxyGuard": "#1b9e77",
        "Oracle release labels": "#7570b3",
    }
    markers = {
        "Plug-in release fraction": "X",
        "Point rule + binomial": "D",
        "Per-release IUT + binomial": "^",
        "Two-level ProxyGuard": "o",
        "Oracle release labels": "s",
    }
    largest_audit = int(summary["AuditN"].max())
    subset = summary[summary["AuditN"].eq(largest_audit)].sort_values("Releases")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25))
    for method in MECHANISM_METHODS:
        frame = subset[subset["Method"].eq(method)]
        axes[0].plot(
            frame["Releases"],
            frame["FalseMechanismValidation"],
            color=colors[method],
            marker=markers[method],
            linewidth=2.1,
            markersize=5,
            label=method,
        )
        axes[1].plot(
            frame["Releases"],
            frame["MechanismValidationPower"],
            color=colors[method],
            marker=markers[method],
            linewidth=2.1,
            markersize=5,
            label=method,
        )
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(r"Boundary reliability $\eta=0.8$")
    axes[0].set_xlabel("Independent releases per mechanism")
    axes[0].set_ylabel("False mechanism validation")
    axes[0].set_ylim(-0.01, 1.02)
    axes[1].set_title(r"Power at reliability $\eta=0.98$")
    axes[1].set_xlabel("Independent releases per mechanism")
    axes[1].set_ylabel("Mechanisms validated")
    axes[1].set_ylim(-0.01, 1.02)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered ProxyGuard mechanism and adaptive studies."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_mechanism_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_mechanism_study",
    )
    parser.add_argument(
        "--figure",
        default="paper/proxyguard/figs/mechanism_control.png",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mechanism = run_mechanism_study(registry)
    adaptive = run_adaptive_study(registry)
    planning = build_planning_table(registry)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mechanism.to_csv(output_root / "mechanism_calibration.csv", index=False)
    adaptive.to_csv(output_root / "adaptive_search.csv", index=False)
    planning.to_csv(output_root / "release_planning.csv", index=False)
    plot_mechanism_study(mechanism, Path(args.figure))
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256_file": str(registry_path.with_suffix(".sha256")),
                "mechanism_rows": len(mechanism),
                "adaptive_rows": len(adaptive),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    headline = mechanism[
        mechanism["AuditN"].eq(mechanism["AuditN"].max())
        & mechanism["Releases"].eq(mechanism["Releases"].max())
    ][
        [
            "Method",
            "FalseMechanismValidation",
            "MechanismValidationPower",
        ]
    ]
    print(headline.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print(planning.to_string(index=False))


if __name__ == "__main__":
    main()
