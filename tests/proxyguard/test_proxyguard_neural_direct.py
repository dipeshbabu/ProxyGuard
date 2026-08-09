from __future__ import annotations

import numpy as np
import pandas as pd

from proxyguard.shared_target import StratifiedReleaseEvidence
from scripts.proxyguard.neural_direct_audit import (
    _fit_release_model,
    _losses,
    _select_configurations,
    _stratified_named_summary,
)


def test_random_feature_generator_fits_full_release_procedure() -> None:
    rng = np.random.default_rng(13)
    X = rng.integers(0, 5, size=(200, 5)).astype(float)
    y = ((X[:, 0] + X[:, 1]) > 4).astype(int)
    config = {
        "hidden_units": 8,
        "train_noise": 0.05,
        "sample_noise": 0.05,
        "label_fidelity": 0.9,
        "ridge": 0.01,
    }
    source_settings = {
        "regularization_c": 1.0,
        "maximum_iterations": 200,
        "decision_threshold": 0.5,
        "cost_threshold": 1.0 / 6.0,
    }

    model = _fit_release_model(
        X,
        y,
        config,
        source_settings,
        synthetic_rows=300,
        seed=21,
    )
    probability = model.predict_proba(X)[:, 1]
    losses = _losses(y, probability, source_settings)

    assert probability.shape == (200,)
    assert losses.shape == (200, 3)
    assert np.all((losses >= 0.0) & (losses <= 1.0))


def test_configuration_selection_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "Configuration": np.repeat(["a", "b", "c", "d"], 4),
            "MinimumMargin": [
                0.09,
                0.10,
                0.11,
                0.12,
                0.052,
                0.055,
                0.058,
                0.06,
                0.02,
                0.03,
                0.04,
                0.05,
                -0.08,
                -0.07,
                -0.06,
                -0.05,
            ],
        }
    )

    selected = _select_configurations(frame, target_margin=0.055)

    assert selected == {
        "high_fidelity": "a",
        "moderate_evidence": "b",
        "degraded": "d",
    }


def test_stratified_named_summary_uses_registered_local_budget() -> None:
    evidence = StratifiedReleaseEvidence(
        weighted_means=np.full((10, 2), 0.1),
        validation_pvalues=np.full((10, 2), 1e-8),
        violation_pvalues=np.ones((10, 2)),
        stratum_weights=(0.5, 0.5),
        stratum_sizes=(50, 50),
        concentration_method="bernoulli_kl",
    )
    registry = {
        "registered_mechanisms": 2,
        "total_alpha": 0.05,
        "named_release_error_share": 0.8,
        "minimum_reliability": 0.5,
        "requirements": [{"name": "first"}, {"name": "second"}],
    }

    summary, release_frame = _stratified_named_summary(
        evidence,
        registry=registry,
        role="moderate",
    )

    assert summary["ValidatedReleases"] == 10
    assert summary["DetectedReleaseViolations"] == 0
    assert summary["ReliabilityLCB"] > 0.5
    assert release_frame["Validated"].all()
