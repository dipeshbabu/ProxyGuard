from __future__ import annotations

from scripts.proxyguard.run_proxyguard_prospective_aim_mechanism import (
    registered_mechanism_release_indices,
    registered_mechanism_settings,
)


def test_registered_mechanism_settings_are_explicit() -> None:
    registry = {
        "prospective_aim_mechanism": {
            "minimum_reliability": 0.8,
            "total_alpha": 0.05,
            "release_error_share": 0.4,
            "analysis_status": "prospective replication",
        }
    }

    settings = registered_mechanism_settings(registry)

    assert settings == {
        "minimum_reliability": 0.8,
        "total_alpha": 0.05,
        "release_error_share": 0.4,
        "analysis_status": "prospective replication",
    }


def test_amendment_selects_unique_mechanism_releases() -> None:
    amendment = {
        "prospective_aim_mechanism": {
            "mechanism_release_indices": [2, 3, 4],
        }
    }

    assert registered_mechanism_release_indices(amendment) == [2, 3, 4]
