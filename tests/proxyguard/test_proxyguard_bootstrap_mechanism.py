from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.proxyguard.run_proxyguard_bootstrap_mechanism import (
    _constant_policy_threshold,
    bootstrap_release,
)


def test_bootstrap_release_is_seeded_and_preserves_schema() -> None:
    X = pd.DataFrame({"a": np.arange(20), "b": np.arange(20) * 2})
    y = pd.Series([0, 1] * 10)

    first_X, first_y = bootstrap_release(X, y, seed=11)
    second_X, second_y = bootstrap_release(X, y, seed=11)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)
    assert list(first_X) == ["a", "b"]
    assert len(first_X) == len(X)
    assert set(first_y.unique()).issubset({0, 1})


def test_bootstrap_release_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same positive length"):
        bootstrap_release(pd.DataFrame({"x": [1, 2]}), pd.Series([1]), seed=3)


def test_constant_policy_threshold_is_chosen_on_validation_cost() -> None:
    assert _constant_policy_threshold(pd.Series([1, 1] + [0] * 8), 0.2) == 0.0
    assert _constant_policy_threshold(pd.Series([1] + [0] * 9), 0.2) == 1.0
