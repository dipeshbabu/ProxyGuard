from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.proxyguard.build_proxyguard_mechanism_assets import (
    write_adaptive_table,
    write_absolute_risk_baseline_table,
    write_alpha_allocation_table,
    write_claim_status_table,
    write_magic_sealed_table,
    write_mechanism_calibration_table,
    write_near_boundary_table,
    write_positive_bootstrap_table,
    write_prospective_aim_table,
    write_real_mechanism_table,
    write_release_planning_table,
    write_target_lineage_table,
)


def test_mechanism_tables_are_self_contained(tmp_path: Path) -> None:
    calibration = pd.DataFrame(
        {
            "Method": ["Two-level ProxyGuard"],
            "AuditN": [500],
            "Releases": [50],
            "FalseMechanismValidation": [0.01],
            "MechanismValidationPower": [0.9],
        }
    )
    adaptive = pd.DataFrame(
        [
            {
                "Analysis": "False validation",
                "Method": method,
                "Round": round_number,
                "Rate": 0.02,
            }
            for round_number in (1, 10, 25, 50)
            for method in ("Fixed alpha each round", "Quadratic alpha spending")
        ]
        + [
            {
                "Analysis": "Valid-candidate power",
                "Method": "Quadratic alpha spending",
                "Round": round_number,
                "Rate": 0.8,
            }
            for round_number in (1, 10, 25, 50)
        ]
    )
    mechanism = pd.DataFrame(
        {
            "Dataset": ["taiwan_default"],
            "Epsilon": [1.0],
            "Releases": [5],
            "ValidatedReleases": [0],
            "DetectedReleaseViolations": [5],
            "ReliabilityLCB": [0.0],
            "ReliabilityUCB": [0.69],
            "Status": ["Reliability violation detected"],
        }
    )
    planning = pd.DataFrame(
        {
            "MinimumReliability": [0.8],
            "Mechanisms": [5],
            "AllRecognizedGoodReleasesNeeded": [24],
        }
    )
    near_boundary = pd.DataFrame(
        {
            "BadReleaseRisk": [0.102, 0.102],
            "Method": ["Uncorrected release IUT", "Inner Holm (ProxyGuard)"],
            "FalseReleaseRecognitionFWER": [0.9, 0.02],
            "FalseMechanismValidation": [0.04, 0.01],
        }
    )
    allocation = pd.DataFrame(
        {
            "ReleaseErrorShare": [0.25, 0.5],
            "ReleaseAlpha": [0.0125, 0.025],
            "MechanismAlpha": [0.0375, 0.025],
            "FalseMechanismValidation": [0.01, 0.01],
            "MechanismValidationPower": [0.7, 0.8],
        }
    )
    prospective = pd.DataFrame(
        {
            "Epsilon": [1.0],
            "Releases": [25],
            "ValidatedReleases": [0],
            "DetectedReleaseViolations": [20],
            "ReliabilityLCB": [0.0],
            "ReliabilityUCB": [0.3],
            "MeanCost5xChange": [0.04],
            "Status": ["Reliability violation detected"],
        }
    )
    positive = pd.DataFrame(
        {
            "Requirement": [
                "Brier transfer",
                "Proxy Brier risk",
                "Proxy log-loss risk",
                "Proxy Cost5x risk",
            ],
            "Estimand": [
                "relative_regret",
                "absolute_risk",
                "absolute_risk",
                "absolute_risk",
            ],
            "Tolerance": [0.04, 0.18, 0.07, 0.16],
            "MeanValue": [0.01, 0.14, 0.03, 0.11],
            "SimultaneousUCB": [0.02, 0.16, 0.04, 0.14],
        }
    )
    baselines = pd.DataFrame(
        {
            "Baseline": ["Source procedure", "Constant 0.5"],
            "BrierRisk": [0.13, 0.25],
            "ClippedLogLossRisk": [0.03, 0.05],
            "Cost5xRisk": [0.11, 0.16],
        }
    )
    lineage = pd.DataFrame(
        {
            "Study": ["AIM informed replication"],
            "ReplicationAuditN": [6000],
            "OverlapPilotTrain": [3623],
            "OverlapPilotValidation": [1169],
            "OverlapPilotAudit": [1208],
        }
    )
    claim_status = pd.DataFrame(
        {
            "Experiment": ["Sealed audit"],
            "EvidenceClass": ["Prospective"],
            "TargetLineage": ["Untouched reserve"],
            "PreAccessRegistry": ["Yes"],
            "ProjectWideUntouched": ["Yes"],
            "FormalGuarantee": ["Yes"],
        }
    )
    magic_detail = pd.DataFrame(
        {
            "Requirement": [
                "Brier transfer",
                "Proxy Brier risk",
                "Log-loss transfer",
                "Proxy log-loss risk",
                "Cost5x transfer",
                "Proxy Cost5x risk",
            ],
            "MeanValue": [0.003, 0.097, 0.001, 0.023, 0.003, 0.067],
            "SimultaneousUCB": [0.016, 0.114, 0.010, 0.030, 0.022, 0.087],
            "Tolerance": [0.04, 0.141, 0.04, 0.072, 0.04, 0.108],
        }
    )

    write_mechanism_calibration_table(calibration, tmp_path / "calibration.tex")
    write_adaptive_table(adaptive, tmp_path / "adaptive.tex")
    write_real_mechanism_table(mechanism, tmp_path / "mechanism.tex")
    write_release_planning_table(planning, tmp_path / "planning.tex")
    write_near_boundary_table(near_boundary, tmp_path / "near_boundary.tex")
    write_alpha_allocation_table(allocation, tmp_path / "allocation.tex")
    write_prospective_aim_table(prospective, tmp_path / "prospective.tex")
    write_positive_bootstrap_table(positive, tmp_path / "positive.tex")
    write_absolute_risk_baseline_table(
        baselines,
        positive,
        tmp_path / "baselines.tex",
    )
    write_target_lineage_table(lineage, tmp_path / "lineage.tex")
    write_claim_status_table(claim_status, tmp_path / "claim_status.tex")
    write_magic_sealed_table(magic_detail, tmp_path / "magic.tex")

    assert "Two-level \\method{}" in (tmp_path / "calibration.tex").read_text()
    assert "Spending power" in (tmp_path / "adaptive.tex").read_text()
    assert "Taiwan" in (tmp_path / "mechanism.tex").read_text()
    assert "80\\%" in (tmp_path / "planning.tex").read_text()
    assert "Separate release IUTs" in (tmp_path / "near_boundary.tex").read_text()
    assert "$\\lambda$" in (tmp_path / "allocation.tex").read_text()
    assert "AIM" in (tmp_path / "prospective.tex").read_text()
    assert "Absolute risk" in (tmp_path / "positive.tex").read_text()
    assert "Registered ceiling" in (tmp_path / "baselines.tex").read_text()
    assert "6{,}000" in (tmp_path / "lineage.tex").read_text()
    assert "Untouched reserve" in (tmp_path / "claim_status.tex").read_text()
    assert "Clipped log loss" in (tmp_path / "magic.tex").read_text()
