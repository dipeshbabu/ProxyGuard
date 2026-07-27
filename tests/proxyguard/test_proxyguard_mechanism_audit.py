from __future__ import annotations

import pandas as pd
import pytest

from scripts.proxyguard.run_proxyguard_mechanism_audit import (
    add_mechanism_diagnostics,
    build_real_mechanism_audit,
    mechanism_name,
)


def _losses(releases: int = 30) -> pd.DataFrame:
    frames = []
    for release in range(1, releases + 1):
        for epsilon, regret in ((1, -0.4), (5, 0.4)):
            candidate = f"toy::aim_e{epsilon}::release_{release}"
            frames.append(
                pd.DataFrame(
                    {
                        "Candidate": [candidate] * 1000,
                        "brier_regret": [regret] * 1000,
                        "logloss_regret": [regret] * 1000,
                        "cost5x_regret": [regret] * 1000,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_mechanism_name_keeps_dataset_and_configuration() -> None:
    assert mechanism_name("german_credit::aim_e5::release_3") == (
        "german_credit::aim_e5"
    )
    with pytest.raises(ValueError, match="Unexpected"):
        mechanism_name("bad-name")


def test_real_mechanism_audit_uses_release_groups() -> None:
    result = build_real_mechanism_audit(
        _losses(),
        minimum_reliability=0.8,
        total_alpha=0.05,
        release_error_share=0.5,
    )
    summary = result.mechanism_summary.set_index("Mechanism")

    assert summary.loc["toy::aim_e1", "Status"] == "Mechanism validated"
    assert summary.loc["toy::aim_e5", "Status"] == "Reliability violation detected"


def test_mechanism_diagnostics_are_aggregated_by_configuration() -> None:
    result = build_real_mechanism_audit(
        _losses(),
        minimum_reliability=0.8,
        total_alpha=0.05,
        release_error_share=0.5,
    )
    diagnostics = pd.DataFrame(
        {
            "Candidate": [
                "toy::aim_e1::release_1",
                "toy::aim_e1::release_2",
                "toy::aim_e5::release_1",
            ],
            "Dataset": ["toy", "toy", "toy"],
            "Epsilon": [1.0, 1.0, 5.0],
            "AUCChange": [-0.1, -0.2, -0.3],
            "Cost5xChange": [0.01, 0.02, 0.03],
        }
    )

    merged = add_mechanism_diagnostics(result.mechanism_summary, diagnostics)
    indexed = merged.set_index("Mechanism")

    assert indexed.loc["toy::aim_e1", "MeanAUCChange"] == pytest.approx(-0.15)
    assert indexed.loc["toy::aim_e5", "MeanCost5xChange"] == pytest.approx(0.03)
