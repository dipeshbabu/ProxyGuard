from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.proxyguard.run_proxyguard_mechanism_revision_study import (
    run_alpha_allocation_study,
    run_near_boundary_study,
)


def _registry() -> dict:
    return {
        "near_boundary_inner_correction": {
            "total_alpha": 0.05,
            "release_error_share": 0.5,
            "minimum_reliability": 0.8,
            "valid_reliability": 0.98,
            "requirements": 2,
            "mechanisms": 2,
            "releases": 8,
            "audit_size": 100,
            "tolerance": 0.1,
            "safe_release_risk": 0.02,
            "bad_release_risks": [0.1, 0.102],
            "repetitions": 40,
            "seed": 9,
        },
        "alpha_allocation": {
            "total_alpha": 0.05,
            "release_error_shares": [0.25, 0.5, 0.75],
            "minimum_reliability": 0.8,
            "valid_reliability": 0.98,
            "requirements": 2,
            "mechanisms": 2,
            "releases": 8,
            "audit_size": 100,
            "tolerance": 0.1,
            "safe_release_risk": 0.02,
            "bad_release_risk": 0.102,
            "repetitions": 40,
            "seed": 10,
        },
    }


def test_near_boundary_study_separates_inner_correction() -> None:
    result = run_near_boundary_study(_registry())

    assert set(result["BadReleaseRisk"]) == {0.1, 0.102}
    assert set(result["Method"]) == {
        "Uncorrected release IUT",
        "Inner Holm (ProxyGuard)",
        "Oracle release labels",
    }
    indexed = result.set_index(["BadReleaseRisk", "Method"])
    for bad_risk in (0.1, 0.102):
        assert (
            indexed.loc[
                (bad_risk, "Inner Holm (ProxyGuard)"),
                "FalseReleaseRecognitionFWER",
            ]
            <= indexed.loc[
                (bad_risk, "Uncorrected release IUT"),
                "FalseReleaseRecognitionFWER",
            ]
        )
        assert (
            indexed.loc[
                (bad_risk, "Inner Holm (ProxyGuard)"),
                "FalseMechanismValidation",
            ]
            <= indexed.loc[
                (bad_risk, "Uncorrected release IUT"),
                "FalseMechanismValidation",
            ]
        )


def test_alpha_allocation_study_reports_the_registered_grid() -> None:
    result = run_alpha_allocation_study(_registry())

    assert result["ReleaseErrorShare"].tolist() == [0.25, 0.5, 0.75]
    assert (result["ReleaseAlpha"] + result["MechanismAlpha"]).round(12).eq(0.05).all()


def test_revision_registry_matches_its_frozen_digest() -> None:
    registry_path = Path("registries/proxyguard_mechanism_revision_registry.json")
    digest_path = registry_path.with_suffix(".sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(registry_path.read_bytes()).hexdigest()

    assert observed == expected


def test_revision_amendment_matches_its_frozen_digest() -> None:
    amendment_path = Path(
        "registries/proxyguard_mechanism_revision_amendment_20260726.json"
    )
    digest_path = amendment_path.with_suffix(".sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(amendment_path.read_bytes()).hexdigest()

    assert observed == expected
