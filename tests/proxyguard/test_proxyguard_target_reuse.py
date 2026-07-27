from scripts.proxyguard.run_proxyguard_target_reuse_study import run_study


def test_target_reuse_study_returns_all_registered_comparisons() -> None:
    registry = {
        "seed": 19,
        "repetitions": 100,
        "candidates": 5,
        "requirements": 2,
        "audit_records": 100,
        "tolerance": 0.1,
        "invalid_gap": 0.01,
        "safe_risk": 0.02,
        "alpha": 0.05,
    }

    result = run_study(registry)

    assert result["Method"].tolist() == [
        "Selected candidate, reused target",
        "Selected candidate, sealed target",
        "Complete family, Holm correction",
    ]
    assert result["Trials"].eq(100).all()
    assert result["FalseValidationRate"].between(0.0, 1.0).all()
    assert (result["Wilson95Low"] <= result["FalseValidationRate"]).all()
    assert (result["Wilson95High"] >= result["FalseValidationRate"]).all()
