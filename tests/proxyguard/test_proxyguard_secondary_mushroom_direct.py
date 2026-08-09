from __future__ import annotations

import pandas as pd
import pytest

from scripts.proxyguard.run_proxyguard_secondary_mushroom_direct import (
    _encode_frame,
    _fit_classifier,
    _fit_dp_conditional_generator,
    _round_up,
    _sample_dp_conditional_generator,
)


REGISTRY = {
    "dataset": {
        "delimiter": ";",
        "feature_domains": {
            "class": ["e", "p"],
            "cap-shape": ["b", "x", "?"],
            "season": ["s", "u", "?"],
        },
    }
}


def test_secondary_mushroom_encoder_uses_registered_public_domains(tmp_path) -> None:
    path = tmp_path / "development.csv"
    path.write_text(
        "class;cap-shape;season\n"
        "e;b;s\n"
        "p;x;u\n"
        "e;;s\n",
        encoding="utf-8",
    )

    X, y = _encode_frame(path, REGISTRY)

    assert X.to_dict(orient="list") == {
        "cap-shape": [0, 1, 2],
        "season": [0, 1, 0],
    }
    assert y.tolist() == [0, 1, 0]


def test_secondary_mushroom_encoder_rejects_unregistered_value(tmp_path) -> None:
    path = tmp_path / "development.csv"
    path.write_text("class;cap-shape;season\ne;z;s\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unregistered values"):
        _encode_frame(path, REGISTRY)


def test_secondary_mushroom_encoder_can_use_registered_unknown_policy(tmp_path) -> None:
    path = tmp_path / "development.csv"
    path.write_text("class;cap-shape;season\ne;z;s\n", encoding="utf-8")
    registry = {
        "dataset": {
            **REGISTRY["dataset"],
            "unknown_category_policy": "map_to_registered_unknown",
        }
    }

    X, _ = _encode_frame(path, registry)

    assert X["cap-shape"].tolist() == [2]


def test_fixed_categorical_classifier_returns_probabilities() -> None:
    X = pd.DataFrame(
        {
            "cap-shape": [0, 1, 0, 1, 0, 1, 0, 1],
            "season": [0, 1, 1, 0, 0, 1, 1, 0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    model, threshold, probability = _fit_classifier(
        X,
        y,
        X,
        y,
        REGISTRY,
        13,
    )

    assert model.predict_proba(X).shape == (8, 2)
    assert probability.shape == (8,)
    assert 0.0 <= threshold <= 1.0


def test_round_up_is_stable_at_registered_precision() -> None:
    assert _round_up(0.12301) == 0.124
    assert _round_up(0.124) == 0.124


def test_private_conditional_generator_uses_registered_sensitivity() -> None:
    table = pd.DataFrame(
        {
            "cap-shape": [0, 1, 0, 1, 0, 1, 0, 1],
            "season": [0, 1, 1, 0, 0, 1, 1, 0],
            "class": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    registry = {
        **REGISTRY,
        "mechanism": {
            "epsilon": 3.0,
            "vector_l1_sensitivity": 3,
            "post_noise_pseudocount": 1.0,
            "synthetic_records": 40,
        },
    }

    generator = _fit_dp_conditional_generator(table, registry, seed=19)
    release, y_release = _sample_dp_conditional_generator(
        generator,
        registry,
        seed=23,
    )

    assert generator["laplace_scale"] == 1.0
    assert release.shape == (40, 2)
    assert len(y_release) == 40
    assert set(release["cap-shape"]).issubset({0, 1, 2})
    assert set(y_release).issubset({0, 1})
