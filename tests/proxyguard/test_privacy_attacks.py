from __future__ import annotations

import numpy as np
import pandas as pd

from proxyguard.attacks import (
    attack_metrics,
    dcr_scores,
    domias_kde_scores,
    fit_attack_representation,
    gen_lra_scores,
)
from scripts.proxyguard.run_proxyguard_privacy_attacks import build_attack_advantage_audit


def test_attack_suite_returns_finite_scores() -> None:
    rng = np.random.default_rng(17)
    reference_frame = pd.DataFrame(rng.normal(size=(80, 5)))
    member_frame = pd.DataFrame(rng.normal(size=(40, 5)))
    nonmember_frame = pd.DataFrame(rng.normal(size=(40, 5)))
    synthetic_frame = member_frame.sample(40, replace=True, random_state=3)
    representation = fit_attack_representation(reference_frame, pca_dimensions=4)
    members = representation.transform(member_frame)
    nonmembers = representation.transform(nonmember_frame)
    reference = representation.transform(reference_frame)
    synthetic = representation.transform(synthetic_frame)
    query = np.vstack([members, nonmembers])

    domias, density = domias_kde_scores(
        query,
        synthetic,
        reference,
        seed=9,
    )
    gen_lra = gen_lra_scores(query, synthetic, reference, neighbors=5)
    dcr = dcr_scores(query, synthetic)
    for scores in (domias, density, gen_lra, dcr):
        assert scores.shape == (80,)
        assert np.isfinite(scores).all()


def test_attack_metrics_use_member_positive_direction() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    metrics = attack_metrics(labels, scores)
    assert metrics["AUC"] == 1.0
    assert metrics["TPR5FPR"] == 1.0


def test_attack_advantage_audit_returns_three_way_decision() -> None:
    rows = []
    for attack in ("DOMIAS-KDE", "Gen-LRA"):
        for membership, scores in (
            (1, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            (0, [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        ):
            rows.extend(
                {
                    "Release": "example",
                    "AttackerSeed": 911,
                    "Attack": attack,
                    "Membership": membership,
                    "Score": score,
                }
                for score in scores
            )
    setting = {
        "risk_controlled_claim": {
            "attacker_seed": 911,
            "nonmember_threshold_fraction": 0.5,
            "threshold_calibration_fpr": 0.05,
            "attack_advantage_ceiling": 0.05,
        }
    }
    summary, detail, diagnostics = build_attack_advantage_audit(
        pd.DataFrame(rows),
        setting,
        alpha=0.05,
        bound_method="empirical_bernstein",
    )
    assert summary.loc[0, "Status"] in {
        "Validated",
        "Unresolved",
        "Violation detected",
    }
    assert set(detail["Requirement"]) == {"DOMIAS-KDE", "Gen-LRA"}
    assert len(diagnostics) == 2
