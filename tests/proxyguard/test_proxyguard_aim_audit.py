from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.proxyguard.run_proxyguard_aim_audit import (
    build_classifier,
    build_classifier_library,
    choose_cost_threshold,
    fit_selected_procedure,
    make_split,
)


def test_make_split_is_disjoint_and_exhaustive() -> None:
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = pd.Series([0, 1] * 50)
    split = make_split(X, y, seed=11)

    assert len(split.X_train) == 60
    assert len(split.X_validation) == 20
    assert len(split.X_audit) == 20
    assert split.y_train.mean() == split.y_validation.mean() == split.y_audit.mean()
    position_sets = [
        set(split.train_positions),
        set(split.validation_positions),
        set(split.audit_positions),
    ]
    assert not (position_sets[0] & position_sets[1])
    assert not (position_sets[0] & position_sets[2])
    assert not (position_sets[1] & position_sets[2])
    assert set.union(*position_sets) == set(range(100))


def test_cost_threshold_and_classifier_are_well_formed() -> None:
    y = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = choose_cost_threshold(y, probability, false_negative_cost=5.0)

    assert 0.0 <= threshold <= 1.0
    assert build_classifier(seed=3).named_steps["classifier"].random_state == 3
    assert set(build_classifier_library(seed=3)) == {
        "Logistic",
        "Random forest",
        "Histogram boosting",
    }


def test_selected_procedure_uses_validation_only() -> None:
    X_train = pd.DataFrame({"x": np.r_[np.linspace(-2, -0.1, 30), np.linspace(0.1, 2, 30)]})
    y_train = pd.Series([0] * 30 + [1] * 30)
    X_validation = pd.DataFrame({"x": [-1.5, -0.5, 0.5, 1.5]})
    y_validation = pd.Series([0, 0, 1, 1])

    name, model, threshold, probability = fit_selected_procedure(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=9,
    )

    assert name in build_classifier_library(seed=9)
    assert 0.0 <= threshold <= 1.0
    assert probability.shape == (4,)
    assert model.predict_proba(X_validation).shape == (4, 2)
