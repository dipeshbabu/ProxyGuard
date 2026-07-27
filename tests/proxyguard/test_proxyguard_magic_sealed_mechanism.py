from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.proxyguard.run_proxyguard_magic_sealed_mechanism import (
    _development_split,
    _read_magic,
    _rounded_ceiling,
)


def test_magic_loader_and_development_split(tmp_path: Path) -> None:
    rows = [
        ",".join([*(str(float(i + j)) for j in range(10)), "g" if i % 2 else "h"])
        for i in range(100)
    ]
    path = tmp_path / "magic.data"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    X, y = _read_magic(path)
    X_train, X_validation, y_train, y_validation = _development_split(
        X,
        y,
        seed=19,
    )

    assert X.shape == (100, 10)
    assert y.sum() == 50
    assert len(X_train) == len(y_train) == 80
    assert len(X_validation) == len(y_validation) == 20
    assert np.isclose(y_train.mean(), y_validation.mean())


def test_registered_ceiling_rounds_up() -> None:
    assert _rounded_ceiling(0.1231, 0.05) == 0.174
