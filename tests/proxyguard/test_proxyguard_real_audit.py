from pathlib import Path

import pandas as pd

from scripts.proxyguard.build_proxyguard_real_audit import (
    available_dataset_variants,
    paired_candidate_frame,
)


def write_records(path: Path, probability: list[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Dataset": ["d", "d"],
            "Model": ["m", "m"],
            "split_seed": [7, 7],
            "record_id": ["a", "b"],
            "y_true": [0, 1],
            "probability": probability,
            "cost_threshold_5x": [threshold, threshold],
        }
    ).to_csv(path, index=False)


def test_real_audit_pairs_source_and_proxy_records(tmp_path: Path) -> None:
    write_records(
        tmp_path / "proxy_transform" / "baseline" / "d" / "audit_records.csv",
        probability=[0.1, 0.9],
        threshold=0.5,
    )
    write_records(
        tmp_path / "proxy_transform" / "masked" / "d" / "audit_records.csv",
        probability=[0.8, 0.2],
        threshold=0.5,
    )

    variants = available_dataset_variants(tmp_path)
    losses = paired_candidate_frame(tmp_path, "d", "masked", split_seed=7)

    assert variants == {"d": ["masked"]}
    assert (losses["brier_regret"] > 0.0).all()
    assert (losses["cost5x_regret"] > 0.0).all()
