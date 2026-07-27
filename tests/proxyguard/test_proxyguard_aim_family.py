from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.proxyguard.build_proxyguard_aim_family import build_family_audit


def _write_run(path: Path, candidate: str, regret: float) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "Candidate": [candidate] * 200,
            "brier_regret": [regret] * 200,
            "logloss_regret": [regret] * 200,
            "cost5x_regret": [regret] * 200,
        }
    ).to_csv(path / "paired_losses.csv", index=False)
    pd.DataFrame(
        {
            "Candidate": [candidate],
            "Dataset": ["toy"],
            "Mechanism": ["AIM"],
            "Epsilon": [1.0],
            "Delta": [1e-9],
            "AuditN": [200],
            "SourceModel": ["Logistic"],
            "ProxyModel": ["Random forest"],
            "SourceAUC": [0.8],
            "ProxyAUC": [0.79],
            "AUCChange": [-0.01],
            "SourceCost5x": [0.1],
            "ProxyCost5x": [0.11],
            "SourceThreshold": [0.4],
            "ProxyThreshold": [0.4],
            "SyntheticPositiveRate": [0.5],
        }
    ).to_csv(path / "diagnostics.csv", index=False)


def test_build_family_audit_combines_candidates(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, "toy::aim_e1", regret=-0.20)
    _write_run(second, "toy::aim_e5", regret=0.20)

    summary, detail, diagnostics, losses = build_family_audit(
        [first, second],
        alpha=0.05,
    )

    assert set(summary["Candidate"]) == {"toy::aim_e1", "toy::aim_e5"}
    assert len(detail) == 6
    assert len(diagnostics) == 2
    assert len(losses) == 400
