from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.run_proxyguard_calibration_study import (  # noqa: E402
    exact_bernoulli_badness_pvalue,
    holm_rejections,
    invalid_probabilities,
)


def _wilson_interval(events: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    z = 1.959963984540054
    proportion = events / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    low = 0.0 if events == 0 else max(0.0, center - radius)
    high = 1.0 if events == trials else min(1.0, center + radius)
    return float(low), float(high)


def run_study(registry: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(registry["seed"]))
    repetitions = int(registry["repetitions"])
    candidates = int(registry["candidates"])
    requirements = int(registry["requirements"])
    audit_records = int(registry["audit_records"])
    tolerance = float(registry["tolerance"])
    alpha = float(registry["alpha"])
    probabilities = invalid_probabilities(
        candidates=candidates,
        requirements=requirements,
        tolerance=tolerance,
        invalid_gap=float(registry["invalid_gap"]),
        safe_risk=float(registry["safe_risk"]),
    )

    screen_counts = rng.binomial(
        audit_records,
        probabilities,
        size=(repetitions, candidates, requirements),
    )
    screen_components = exact_bernoulli_badness_pvalue(
        screen_counts,
        audit_records,
        tolerance,
    )
    screen_candidate_pvalues = screen_components.max(axis=2)
    selected = np.argmin(screen_candidate_pvalues, axis=1)
    rows = np.arange(repetitions)

    reused = screen_candidate_pvalues[rows, selected] <= alpha

    sealed_counts = rng.binomial(
        audit_records,
        probabilities,
        size=(repetitions, candidates, requirements),
    )
    sealed_components = exact_bernoulli_badness_pvalue(
        sealed_counts,
        audit_records,
        tolerance,
    )
    sealed_candidate_pvalues = sealed_components.max(axis=2)
    sealed = sealed_candidate_pvalues[rows, selected] <= alpha

    corrected = holm_rejections(screen_candidate_pvalues, alpha).any(axis=1)

    outcomes = {
        "Selected candidate, reused target": reused,
        "Selected candidate, sealed target": sealed,
        "Complete family, Holm correction": corrected,
    }
    result_rows = []
    for method, decisions in outcomes.items():
        events = int(decisions.sum())
        low, high = _wilson_interval(events, repetitions)
        result_rows.append(
            {
                "Method": method,
                "FalseValidations": events,
                "Trials": repetitions,
                "FalseValidationRate": events / repetitions,
                "Wilson95Low": low,
                "Wilson95High": high,
            }
        )
    return pd.DataFrame(result_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare target-audit reuse with a sealed audit and family correction."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_target_reuse_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_target_reuse",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    summary = run_study(registry)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registered_settings": registry,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
