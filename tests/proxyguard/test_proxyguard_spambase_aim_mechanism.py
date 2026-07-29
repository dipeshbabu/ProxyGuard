from __future__ import annotations

import pandas as pd

from scripts.proxyguard.run_proxyguard_spambase_aim_mechanism import (
    FEATURES,
    _read_spambase,
    _rounded_ceiling,
)


def test_spambase_reader_inverts_target(tmp_path) -> None:
    row = [0.0] * len(FEATURES) + [1]
    path = tmp_path / "spambase.data"
    path.write_text(",".join(map(str, row)) + "\n", encoding="utf-8")
    X, y = _read_spambase(path)
    assert X.shape == (1, len(FEATURES))
    assert y.tolist() == [0]
    assert isinstance(X, pd.DataFrame)


def test_spambase_limit_rounds_up() -> None:
    assert _rounded_ceiling(0.1234, 0.05) == 0.174
