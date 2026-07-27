from __future__ import annotations

import pandas as pd

from scripts.proxyguard.audit_proxyguard_target_lineage import summarize_overlap


def test_lineage_summary_counts_every_replication_audit_record() -> None:
    y = pd.Series([0, 1] * 100)
    summary, detail = summarize_overlap(
        y,
        study="test",
        pilot_seed=11,
        replication_seed=23,
        pilot_informed_choice="test choice",
    )

    assert summary["ReplicationAuditN"] == 40
    assert (
        summary["OverlapPilotTrain"]
        + summary["OverlapPilotValidation"]
        + summary["OverlapPilotAudit"]
        == summary["OverlapAnyPilotRole"]
        == 40
    )
    assert summary["RecordDisjointFromPilot"] is False
    assert summary["TheoremBackedProspectiveClaim"] is False
    assert len(detail) == 40
    assert set(detail["PilotRole"]) <= {"train", "validation", "audit"}
