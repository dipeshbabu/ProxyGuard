from __future__ import annotations

from scripts.proxyguard.run_proxyguard_stratified_subgroup_study import (
    simulate_stratified_subgroup_study,
)


def test_stratified_subgroup_study_reports_both_designs() -> None:
    registry = {
        "subgroup_audit": {
            "alpha": 0.05,
            "repetitions": 50,
            "seed": 7,
            "candidates": 3,
            "subgroup_prevalence": 0.02,
            "risk_tolerance": 0.1,
            "valid_subgroup_risk": 0.04,
            "boundary_subgroup_risk": 0.1,
            "audit_sizes": [100],
            "stratified_subgroup_sizes": [50],
        }
    }
    summary = simulate_stratified_subgroup_study(registry)
    assert set(summary["Design"]) == {"Simple random", "Subgroup-stratified"}
    assert (summary["RareGroupNMean"] >= 0).all()
    assert summary["FalseValidationFWER"].between(0, 1).all()
    assert summary["ValidationPower"].between(0, 1).all()
