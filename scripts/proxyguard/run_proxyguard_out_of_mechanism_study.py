from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import (  # noqa: E402
    holm_adjust,
)

Generator = Callable[[np.random.Generator, int, int, int, float, bool], np.ndarray]


def _continuous_beta(
    rng: np.random.Generator,
    candidates: int,
    requirements: int,
    audit_n: int,
    tolerance: float,
    valid: bool,
) -> np.ndarray:
    means = np.full((candidates, requirements), tolerance - 0.055)
    if not valid:
        means[:, 0] = tolerance
    concentration = 18.0
    unit_means = (means + 1.0) / 2.0
    alpha = unit_means * concentration
    beta = (1.0 - unit_means) * concentration
    draws = rng.beta(
        alpha[:, :, None],
        beta[:, :, None],
        size=(candidates, requirements, audit_n),
    )
    return 2.0 * draws - 1.0


def _rare_subgroup_mixture(
    rng: np.random.Generator,
    candidates: int,
    requirements: int,
    audit_n: int,
    tolerance: float,
    valid: bool,
) -> np.ndarray:
    rare = rng.random((candidates, 1, audit_n)) < 0.08
    common_noise = rng.normal(-0.08, 0.10, size=(candidates, requirements, audit_n))
    rare_noise = rng.normal(0.75, 0.08, size=(candidates, requirements, audit_n))
    target = tolerance - 0.04 if valid else tolerance
    base_mean = 0.08 * 0.75 + 0.92 * -0.08
    draws = np.where(rare, rare_noise, common_noise)
    draws[:, 0, :] += target - base_mean
    return np.clip(draws, -1.0, 1.0)


def _correlated_candidates(
    rng: np.random.Generator,
    candidates: int,
    requirements: int,
    audit_n: int,
    tolerance: float,
    valid: bool,
) -> np.ndarray:
    shared = rng.normal(0.0, 0.15, size=(1, 1, audit_n))
    candidate_noise = rng.normal(
        0.0,
        0.07,
        size=(candidates, requirements, audit_n),
    )
    means = np.full((candidates, requirements, 1), tolerance - 0.05)
    if not valid:
        means[:, 0, 0] = tolerance
    return np.clip(means + shared + candidate_noise, -1.0, 1.0)


FAMILIES: dict[str, Generator] = {
    "continuous beta": _continuous_beta,
    "rare-subgroup mixture": _rare_subgroup_mixture,
    "correlated candidates": _correlated_candidates,
}


def _point_validations(
    draws: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    return (draws.mean(axis=2) < tolerance).all(axis=1)


def _empirical_bernstein_badness_pvalues(
    draws: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    audit_n = draws.shape[2]
    means = draws.mean(axis=2)
    variances = draws.var(axis=2, ddof=1)
    gap = tolerance - means
    coefficient_linear = np.sqrt(2.0 * variances / audit_n)
    coefficient_quadratic = 14.0 / (3.0 * (audit_n - 1))
    minimum_log = np.log(2.0)
    minimum_radius = (
        coefficient_linear * np.sqrt(minimum_log)
        + coefficient_quadratic * minimum_log
    )
    pvalues = np.ones_like(means)
    invert = gap > minimum_radius
    root = (
        -coefficient_linear[invert]
        + np.sqrt(
            np.square(coefficient_linear[invert])
            + 4.0 * coefficient_quadratic * gap[invert]
        )
    ) / (2.0 * coefficient_quadratic)
    pvalues[invert] = np.minimum(1.0, 2.0 * np.exp(-np.square(root)))
    return pvalues


def _uncorrected_validations(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    return pvalues.max(axis=1) <= alpha


def _controlled_validations(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    candidate_pvalues = pvalues.max(axis=1)
    adjusted = holm_adjust(
        {
            f"proxy_{index + 1}": value
            for index, value in enumerate(candidate_pvalues)
        }
    )
    return np.asarray(
        [
            adjusted[f"proxy_{index + 1}"] <= alpha
            for index in range(len(candidate_pvalues))
        ],
        dtype=bool,
    )


def run_study(
    repetitions: int,
    audit_sizes: list[int],
    candidates: int,
    requirement_count: int,
    alpha: float,
    seed: int,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counters: dict[tuple[str, int, str, str], list[int]] = {}

    for family_name, generator in FAMILIES.items():
        for audit_n in audit_sizes:
            for truth in ("invalid", "valid"):
                valid = truth == "valid"
                keys = [
                    (family_name, audit_n, truth, method)
                    for method in ("Point threshold", "Uncorrected IUT", "ProxyGuard")
                ]
                for key in keys:
                    counters[key] = [0, 0]

                for _ in range(repetitions):
                    draws = generator(
                        rng,
                        candidates,
                        requirement_count,
                        audit_n,
                        tolerance,
                        valid,
                    )
                    point = _point_validations(draws, tolerance)
                    pvalues = _empirical_bernstein_badness_pvalues(draws, tolerance)
                    uncorrected = _uncorrected_validations(pvalues, alpha)
                    controlled = _controlled_validations(pvalues, alpha)
                    for method, decisions in (
                        ("Point threshold", point),
                        ("Uncorrected IUT", uncorrected),
                        ("ProxyGuard", controlled),
                    ):
                        key = (family_name, audit_n, truth, method)
                        if valid:
                            counters[key][0] += int(decisions.sum())
                            counters[key][1] += candidates
                        else:
                            counters[key][0] += int(decisions.any())
                            counters[key][1] += 1

    rows: list[dict[str, float | int | str]] = []
    for (family, audit_n, truth, method), (events, trials) in counters.items():
        rows.append(
            {
                "Family": family,
                "AuditN": audit_n,
                "Truth": truth,
                "Method": method,
                "Events": events,
                "Trials": trials,
                "Rate": events / trials,
                "Outcome": (
                    "Family-wise false acceptance"
                    if truth == "invalid"
                    else "Per-candidate validation power"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Truth", "Family", "AuditN", "Method"],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ProxyGuard on regret mechanisms held out from the original study."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_frontier_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_out_of_mechanism",
    )
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    setting = registry["out_of_mechanism_calibration"]
    repetitions = args.repetitions or int(setting["repetitions"])
    summary = run_study(
        repetitions=repetitions,
        audit_sizes=[int(value) for value in setting["audit_sizes"]],
        candidates=int(setting["candidates"]),
        requirement_count=int(setting["requirements"]),
        alpha=float(registry["risk_control"]["alpha"]),
        seed=int(setting["seed"]),
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "repetitions": repetitions,
                "registered_setting": setting,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    invalid = summary[summary["Truth"] == "invalid"]
    print(
        invalid[
            ["Family", "AuditN", "Method", "Rate"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
