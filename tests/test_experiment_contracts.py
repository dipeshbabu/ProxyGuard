from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from risk_models.configs import (
    DATASET_REGISTRY,
    clone_experiment_config,
    get_default_experiment_config,
    get_dataset_config,
    get_fmsd_model_configs,
)
from risk_models.cv_runner import run_benchmark
from risk_models.dataset import load_dataset


def test_fmsd_model_scope_matches_final_paper() -> None:
    model_names = [config.name for config in get_fmsd_model_configs(include_tabpfn=True)]
    assert model_names == [
        "logreg_baseline",
        "xgb_baseline",
        "compact_xgb",
        "tabpfn_baseline",
    ]


def require_local_dataset(dataset_name: str):
    config = get_dataset_config(dataset_name)
    if not Path(config.path).exists():
        pytest.skip(f"{dataset_name} data file is not present locally: {config.path}")
    return config


def test_public_dataset_contracts() -> None:
    available_dataset_names = [
        dataset_name
        for dataset_name in DATASET_REGISTRY
        if Path(get_dataset_config(dataset_name).path).exists()
    ]
    if not available_dataset_names:
        pytest.skip("No local dataset files are present.")

    for dataset_name in available_dataset_names:
        bundle = load_dataset(require_local_dataset(dataset_name))
        X = bundle["X"]
        y = bundle["y"]
        metadata = bundle["metadata"]

        assert not X.empty
        assert len(X) == len(y)
        assert y.nunique() == 2
        assert not X.columns.duplicated().any()
        assert not np.isinf(X.select_dtypes(include=[np.number]).to_numpy()).any()
        assert "subgroup_frame" in metadata


def test_german_weak_label_leakage_proxies_are_removed() -> None:
    bundle = load_dataset(require_local_dataset("german_credit"))
    forbidden = {
        "Credit amount",
        "Duration",
        "Monthly_Revenue",
    }
    forbidden_prefixes = ("Business_Type_",)

    columns = set(bundle["X"].columns)
    assert columns.isdisjoint(forbidden)
    assert not any(column.startswith(forbidden_prefixes) for column in columns)


def test_real_experiment_outputs_include_reliability_metrics() -> None:
    output_root = Path("outputs") / f"contract_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=2,
        output_root=str(output_root),
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
    )
    model_configs = [
        config
        for config in get_fmsd_model_configs(include_tabpfn=False)
        if config.name in {"logreg_baseline", "xgb_baseline", "compact_xgb"}
    ]
    result = run_benchmark(
        require_local_dataset("german_credit"),
        model_configs,
        exp_cfg,
        mode="contract",
    )

    aggregate = result["aggregate_metrics"]
    required = {"Model", "AUC", "Brier", "ECE (10-bin)", "LogLoss", "CalibrationSlope", "n_splits"}
    assert required.issubset(aggregate.columns)
    assert set(aggregate["n_splits"]) == {2}
    assert not result["subgroup_metrics"].empty
    assert not result["feature_stability"].empty

    saved = pd.read_csv(output_root / "contract" / "german_credit" / "aggregate_metrics.csv")
    assert required.issubset(saved.columns)
