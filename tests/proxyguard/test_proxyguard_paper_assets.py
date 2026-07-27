from pathlib import Path

import pandas as pd

from scripts.proxyguard.build_proxyguard_paper_assets import (
    write_aim_audit_table,
    write_calibration_table,
    write_real_audit_table,
)


def test_proxyguard_tables_are_self_contained(tmp_path: Path) -> None:
    calibration = pd.DataFrame(
        {
            "Method": ["ProxyGuard (Holm)"],
            "AuditN": [500],
            "Alpha": [0.05],
            "FalseAcceptanceRate": [0.02],
            "ValidCandidatePower": [0.8],
        }
    )
    real = pd.DataFrame(
        {
            "Candidate": ["a_dataset/noise"],
            "AuditNMin": [100],
            "Validated": [True],
            "ViolationDetected": [False],
            "Status": ["Validated"],
        }
    )

    calibration_path = tmp_path / "calibration.tex"
    real_path = tmp_path / "real.tex"
    write_calibration_table(calibration, calibration_path)
    write_real_audit_table(real, real_path)

    assert "2.0\\%" in calibration_path.read_text(encoding="utf-8")
    assert "A Dataset" in real_path.read_text(encoding="utf-8")


def test_aim_table_is_self_contained(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "Dataset": ["heart_disease"],
            "AuditN": [54],
            "Epsilon": [1.0],
            "AUCChange": [-0.1],
            "SourceCost5x": [0.1],
            "ProxyCost5x": [0.12],
            "Status": ["Unresolved"],
        }
    )
    output = tmp_path / "aim.tex"
    write_aim_audit_table(summary, output)

    contents = output.read_text(encoding="utf-8")
    assert "Heart Disease" in contents
    assert "Unres." in contents
