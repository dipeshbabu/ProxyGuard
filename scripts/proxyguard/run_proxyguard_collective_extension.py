from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.run_proxyguard_mechanism_study import (  # noqa: E402
    outer_rejections,
    release_validation_counts,
)


COLLECTIVE_METHODS = (
    "Two-level ProxyGuard",
    "Collective partial-conjunction release evidence",
    "Fisher-tail partial-conjunction benchmark",
)


def simulate_collective_setting(
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
    minimum_reliability: float,
    valid_reliability: float,
    total_alpha: float,
    release_error_share: float,
) -> pd.DataFrame:
    """Compare individual Holm recognition with collective release evidence."""

    release_alpha = total_alpha * release_error_share
    mechanism_alpha = total_alpha - release_alpha
    boundary_counts = release_validation_counts(
        rng=rng,
        repetitions=repetitions,
        mechanisms=mechanisms,
        releases=releases,
        requirements=requirements,
        audit_size=audit_size,
        tolerance=tolerance,
        safe_risk=safe_risk,
        bad_release_risk=bad_release_risk,
        reliability=minimum_reliability,
        release_alpha=release_alpha,
        include_collective_benchmarks=True,
    )
    valid_counts = release_validation_counts(
        rng=rng,
        repetitions=repetitions,
        mechanisms=mechanisms,
        releases=releases,
        requirements=requirements,
        audit_size=audit_size,
        tolerance=tolerance,
        safe_risk=safe_risk,
        bad_release_risk=bad_release_risk,
        reliability=valid_reliability,
        release_alpha=release_alpha,
        include_collective_benchmarks=True,
    )
    holm_valid = valid_counts["Two-level ProxyGuard"]
    collective_valid = valid_counts[
        "Collective partial-conjunction release evidence"
    ]
    simes_count_gain = collective_valid - holm_valid
    if np.any(simes_count_gain < 0):
        raise RuntimeError("Collective count dropped below the local Holm count.")

    rows: list[dict[str, float | int | str]] = []
    for method in COLLECTIVE_METHODS:
        count_gain = valid_counts[method] - holm_valid
        boundary_rejections = outer_rejections(
            boundary_counts[method],
            releases,
            minimum_reliability,
            mechanism_alpha,
        )
        valid_rejections = outer_rejections(
            valid_counts[method],
            releases,
            minimum_reliability,
            mechanism_alpha,
        )
        boundary_by_repetition = boundary_rejections.any(axis=1).astype(float)
        power_by_repetition = valid_rejections.mean(axis=1)
        rows.append(
            {
                "Method": method,
                "AuditN": audit_size,
                "Releases": releases,
                "Mechanisms": mechanisms,
                "Requirements": requirements,
                "SafeReleaseRisk": safe_risk,
                "MinimumReliability": minimum_reliability,
                "ValidReliability": valid_reliability,
                "FalseMechanismValidation": float(
                    boundary_by_repetition.mean()
                ),
                "FalseMechanismValidationSE": float(
                    boundary_by_repetition.std(ddof=1)
                    / np.sqrt(repetitions)
                ),
                "MechanismValidationPower": float(power_by_repetition.mean()),
                "MechanismValidationPowerSE": float(
                    power_by_repetition.std(ddof=1)
                    / np.sqrt(repetitions)
                ),
                "MeanRecognizedCount": float(valid_counts[method].mean()),
                "StrictCollectiveCountGainRate": float((count_gain > 0).mean()),
                "MeanCollectiveCountGain": float(count_gain.mean()),
                "Repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows)


def run_collective_grid(registry: dict) -> pd.DataFrame:
    settings = registry["collective_evidence"]
    rng = np.random.default_rng(int(settings["seed"]))
    frames: list[pd.DataFrame] = []
    for safe_risk in settings["safe_release_risks"]:
        for valid_reliability in settings["valid_reliabilities"]:
            for audit_size in settings["audit_sizes"]:
                for releases in settings["release_counts"]:
                    frames.append(
                        simulate_collective_setting(
                            rng=rng,
                            repetitions=int(settings["repetitions"]),
                            mechanisms=int(settings["mechanisms"]),
                            releases=int(releases),
                            requirements=int(settings["requirements"]),
                            audit_size=int(audit_size),
                            tolerance=float(settings["tolerance"]),
                            safe_risk=float(safe_risk),
                            bad_release_risk=float(
                                settings["bad_release_risk"]
                            ),
                            minimum_reliability=float(
                                settings["minimum_reliability"]
                            ),
                            valid_reliability=float(valid_reliability),
                            total_alpha=float(settings["total_alpha"]),
                            release_error_share=float(
                                settings["release_error_share"]
                            ),
                        )
                    )
    return pd.concat(frames, ignore_index=True)


def plot_collective_gain(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the full registered release-quality grid at the middle reliability."""

    valid_reliability = float(summary["ValidReliability"].median())
    safe_risks = sorted(summary["SafeReleaseRisk"].unique())
    figure, axes = plt.subplots(
        1,
        len(safe_risks),
        figsize=(10.5, 4.6),
        constrained_layout=True,
    )
    images = []
    for axis, safe_risk in zip(axes, safe_risks, strict=True):
        subset = summary[
            summary["SafeReleaseRisk"].eq(safe_risk)
            & summary["ValidReliability"].eq(valid_reliability)
        ]
        holm = subset[subset["Method"].eq("Two-level ProxyGuard")].pivot(
            index="AuditN",
            columns="Releases",
            values="MechanismValidationPower",
        )
        collective = subset[
            subset["Method"].eq(
                "Collective partial-conjunction release evidence"
            )
        ].pivot(
            index="AuditN",
            columns="Releases",
            values="MechanismValidationPower",
        )
        matrix = collective - holm
        image = axis.imshow(
            matrix.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=0.6,
        )
        images.append(image)
        axis.set_title(f"Valid-release risk {safe_risk:.2f}", fontsize=12)
        axis.set_xlabel("Independent releases")
        axis.set_ylabel("Audit records per release")
        axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
        axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
        axis.tick_params(labelsize=10)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix.iloc[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
    figure.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.03, label="Power gain")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered collective-evidence diagnostic grid."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_collective_extension_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_collective_extension",
    )
    parser.add_argument(
        "--figure",
        default="paper/proxyguard/figs/collective_power.png",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    summary = run_collective_grid(registry)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    plot_collective_gain(summary, Path(args.figure))

    slice_summary = summary[
        summary["SafeReleaseRisk"].eq(summary["SafeReleaseRisk"].median())
        & summary["ValidReliability"].eq(summary["ValidReliability"].median())
        & summary["AuditN"].eq(summary["AuditN"].max())
        & summary["Releases"].eq(summary["Releases"].max())
    ][
        [
            "Method",
            "FalseMechanismValidation",
            "MechanismValidationPower",
            "MeanRecognizedCount",
            "MeanCollectiveCountGain",
        ]
    ]
    print(slice_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
