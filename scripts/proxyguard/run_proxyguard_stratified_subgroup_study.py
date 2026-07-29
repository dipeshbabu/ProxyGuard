from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from scripts.proxyguard.run_proxyguard_mechanism_study import holm_rejections


def _audit_decisions(
    *,
    rng: np.random.Generator,
    repetitions: int,
    candidates: int,
    subgroup_sizes: np.ndarray,
    subgroup_risk: float,
    tolerance: float,
    alpha: float,
) -> np.ndarray:
    """Return Holm decisions for one subgroup-specific requirement."""

    sizes = np.broadcast_to(
        subgroup_sizes[:, None],
        (repetitions, candidates),
    )
    counts = rng.binomial(sizes, subgroup_risk)
    pvalues = np.ones((repetitions, candidates), dtype=float)
    observed = sizes > 0
    pvalues[observed] = binom.cdf(
        counts[observed],
        sizes[observed],
        tolerance,
    )
    return holm_rejections(pvalues, alpha)


def simulate_stratified_subgroup_study(registry: dict) -> pd.DataFrame:
    setting = registry["subgroup_audit"]
    rng = np.random.default_rng(int(setting["seed"]))
    repetitions = int(setting["repetitions"])
    candidates = int(setting["candidates"])
    prevalence = float(setting["subgroup_prevalence"])
    tolerance = float(setting["risk_tolerance"])
    alpha = float(setting["alpha"])
    rows: list[dict[str, float | int | str]] = []

    for audit_size, stratified_size in zip(
        setting["audit_sizes"],
        setting["stratified_subgroup_sizes"],
        strict=True,
    ):
        audit_size = int(audit_size)
        stratified_size = int(stratified_size)
        simple_sizes = rng.binomial(
            audit_size,
            prevalence,
            size=repetitions,
        )
        stratified_sizes = np.full(
            repetitions,
            stratified_size,
            dtype=int,
        )
        for design, subgroup_sizes in (
            ("Simple random", simple_sizes),
            ("Subgroup-stratified", stratified_sizes),
        ):
            boundary_decisions = _audit_decisions(
                rng=rng,
                repetitions=repetitions,
                candidates=candidates,
                subgroup_sizes=subgroup_sizes,
                subgroup_risk=float(setting["boundary_subgroup_risk"]),
                tolerance=tolerance,
                alpha=alpha,
            )
            valid_decisions = _audit_decisions(
                rng=rng,
                repetitions=repetitions,
                candidates=candidates,
                subgroup_sizes=subgroup_sizes,
                subgroup_risk=float(setting["valid_subgroup_risk"]),
                tolerance=tolerance,
                alpha=alpha,
            )
            fwer_by_repetition = boundary_decisions.any(axis=1).astype(float)
            power_by_repetition = valid_decisions.mean(axis=1)
            rows.append(
                {
                    "Design": design,
                    "AuditN": audit_size,
                    "RareGroupNMean": float(subgroup_sizes.mean()),
                    "RareGroupNMinimum": int(subgroup_sizes.min()),
                    "FalseValidationFWER": float(
                        fwer_by_repetition.mean()
                    ),
                    "FalseValidationFWERSE": float(
                        fwer_by_repetition.std(ddof=1)
                        / np.sqrt(repetitions)
                    ),
                    "ValidationPower": float(power_by_repetition.mean()),
                    "ValidationPowerSE": float(
                        power_by_repetition.std(ddof=1)
                        / np.sqrt(repetitions)
                    ),
                    "Repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered rare-subgroup audit simulation."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_stratified_subgroup_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_stratified_subgroup",
    )
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    summary = simulate_stratified_subgroup_study(registry)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
