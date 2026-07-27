from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = (
    "Point threshold",
    "Per-proxy IUT",
    "Bonferroni IUT",
    "ProxyGuard (Holm)",
)


def exact_bernoulli_badness_pvalue(
    successes: np.ndarray,
    audit_size: int,
    tolerance: float,
) -> np.ndarray:
    """Test H0: Bernoulli risk >= tolerance against the lower-risk alternative."""

    return binom.cdf(successes, audit_size, tolerance)


def holm_rejections(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Apply Holm's step-down procedure row-wise."""

    if pvalues.ndim != 2:
        raise ValueError("pvalues must have shape (repetitions, candidates).")
    repetitions, candidates = pvalues.shape
    order = np.argsort(pvalues, axis=1, kind="stable")
    ordered = np.take_along_axis(pvalues, order, axis=1)
    thresholds = alpha / np.arange(candidates, 0, -1)
    passes = ordered <= thresholds
    rejected_in_order = np.cumprod(passes.astype(int), axis=1).astype(bool)
    rejected = np.zeros_like(rejected_in_order)
    rows = np.arange(repetitions)[:, None]
    rejected[rows, order] = rejected_in_order
    return rejected


def candidate_decisions(
    counts: np.ndarray,
    audit_size: int,
    tolerance: float,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Return candidate-level validation decisions for each rule."""

    if counts.ndim != 3:
        raise ValueError("counts must have shape (repetitions, candidates, requirements).")
    requirement_pvalues = exact_bernoulli_badness_pvalue(counts, audit_size, tolerance)
    candidate_pvalues = requirement_pvalues.max(axis=2)
    candidates = counts.shape[1]
    return {
        "Point threshold": (counts / audit_size <= tolerance).all(axis=2),
        "Per-proxy IUT": candidate_pvalues <= alpha,
        "Bonferroni IUT": candidate_pvalues <= alpha / candidates,
        "ProxyGuard (Holm)": holm_rejections(candidate_pvalues, alpha),
    }


def invalid_probabilities(
    candidates: int,
    requirements: int,
    tolerance: float,
    invalid_gap: float,
    safe_risk: float,
) -> np.ndarray:
    probabilities = np.full((candidates, requirements), safe_risk, dtype=float)
    for candidate in range(candidates):
        probabilities[candidate, candidate % requirements] = tolerance + invalid_gap
    return probabilities


def simulate_setting(
    rng: np.random.Generator,
    repetitions: int,
    candidates: int,
    requirements: int,
    audit_size: int,
    tolerance: float,
    invalid_gap: float,
    safe_risk: float,
    alpha: float,
) -> pd.DataFrame:
    invalid_risk = invalid_probabilities(
        candidates,
        requirements,
        tolerance,
        invalid_gap,
        safe_risk,
    )
    invalid_counts = rng.binomial(
        audit_size,
        invalid_risk,
        size=(repetitions, candidates, requirements),
    )
    valid_counts = rng.binomial(
        audit_size,
        safe_risk,
        size=(repetitions, candidates, requirements),
    )
    invalid_decisions = candidate_decisions(invalid_counts, audit_size, tolerance, alpha)
    valid_decisions = candidate_decisions(valid_counts, audit_size, tolerance, alpha)

    rows = []
    for method in METHODS:
        false_acceptance_by_repeat = invalid_decisions[method].any(axis=1).astype(float)
        power_by_repeat = valid_decisions[method].mean(axis=1)
        rows.append(
            {
                "Method": method,
                "AuditN": audit_size,
                "Alpha": alpha,
                "Repetitions": repetitions,
                "Candidates": candidates,
                "Requirements": requirements,
                "Tolerance": tolerance,
                "InvalidGap": invalid_gap,
                "SafeRisk": safe_risk,
                "FalseAcceptanceRate": float(false_acceptance_by_repeat.mean()),
                "FalseAcceptanceSE": float(
                    false_acceptance_by_repeat.std(ddof=1) / np.sqrt(repetitions)
                ),
                "ValidCandidatePower": float(power_by_repeat.mean()),
                "ValidCandidatePowerSE": float(power_by_repeat.std(ddof=1) / np.sqrt(repetitions)),
            }
        )
    return pd.DataFrame(rows)


def run_study(
    repetitions: int,
    seed: int,
    candidates: int = 20,
    requirements: int = 3,
    tolerance: float = 0.10,
    invalid_gap: float = 0.002,
    safe_risk: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for alpha in (0.01, 0.025, 0.05, 0.10):
        frames.append(
            simulate_setting(
                rng=rng,
                repetitions=repetitions,
                candidates=candidates,
                requirements=requirements,
                audit_size=500,
                tolerance=tolerance,
                invalid_gap=invalid_gap,
                safe_risk=safe_risk,
                alpha=alpha,
            )
        )
    for audit_size in (50, 100, 250, 500, 1000):
        if audit_size == 500:
            continue
        frames.append(
            simulate_setting(
                rng=rng,
                repetitions=repetitions,
                candidates=candidates,
                requirements=requirements,
                audit_size=audit_size,
                tolerance=tolerance,
                invalid_gap=invalid_gap,
                safe_risk=safe_risk,
                alpha=0.05,
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_study(summary: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "Point threshold": "#777777",
        "Per-proxy IUT": "#d95f02",
        "Bonferroni IUT": "#7570b3",
        "ProxyGuard (Holm)": "#1b9e77",
    }
    styles = {
        "Point threshold": {"marker": "D", "linestyle": "-"},
        "Per-proxy IUT": {"marker": "^", "linestyle": "-"},
        "Bonferroni IUT": {"marker": "s", "linestyle": "--"},
        "ProxyGuard (Holm)": {"marker": "o", "linestyle": "-"},
    }
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.35))

    calibration = summary[summary["AuditN"].eq(500)].sort_values("Alpha")
    for method in METHODS:
        subset = calibration[calibration["Method"].eq(method)]
        axes[0].plot(
            subset["Alpha"],
            subset["FalseAcceptanceRate"],
            marker=styles[method]["marker"],
            linestyle=styles[method]["linestyle"],
            markersize=7 if method == "Bonferroni IUT" else 5,
            markerfacecolor="none" if method == "Bonferroni IUT" else colors[method],
            markeredgewidth=1.5,
            linewidth=2.2,
            label=method,
            color=colors[method],
        )
    alpha_values = sorted(calibration["Alpha"].unique())
    axes[0].plot(alpha_values, alpha_values, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel(r"Nominal false-acceptance level $\alpha$")
    axes[0].set_ylabel("Empirical false-acceptance rate")
    axes[0].set_title("Twenty invalid candidates")
    axes[0].set_ylim(-0.01, 1.02)

    power = summary[summary["Alpha"].eq(0.05)].sort_values("AuditN")
    for method in METHODS:
        subset = power[power["Method"].eq(method)]
        axes[1].plot(
            subset["AuditN"],
            subset["ValidCandidatePower"],
            marker=styles[method]["marker"],
            linestyle=styles[method]["linestyle"],
            markersize=7 if method == "Bonferroni IUT" else 5,
            markerfacecolor="none" if method == "Bonferroni IUT" else colors[method],
            markeredgewidth=1.5,
            linewidth=2.2,
            label=method,
            color=colors[method],
        )
    axes[1].set_xlabel("Independent audit records")
    axes[1].set_ylabel("Fraction of valid candidates validated")
    axes[1].set_title(r"Power at $\alpha=0.05$")
    axes[1].set_ylim(-0.01, 1.02)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate proxy false-acceptance rules in a controlled Bernoulli study."
    )
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", default="outputs/proxyguard_calibration")
    parser.add_argument(
        "--figure",
        default="paper/proxyguard/figs/risk_control_calibration.png",
    )
    args = parser.parse_args()

    summary = run_study(repetitions=args.repetitions, seed=args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "calibration_summary.csv", index=False)
    plot_study(summary, Path(args.figure))

    headline = summary[summary["AuditN"].eq(500) & summary["Alpha"].eq(0.05)][
        [
            "Method",
            "FalseAcceptanceRate",
            "ValidCandidatePower",
        ]
    ]
    print(headline.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
