from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from risk_models.configs import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    clone_experiment_config,
    get_ablation_model_configs,
    get_default_experiment_config,
    get_dataset_config,
    get_fmsd_model_configs,
    get_midas_model_configs,
    get_spotlight_model_configs,
)
from risk_models.cv_runner import run_benchmark
from risk_models.dataset import load_dataset
from risk_models.model import RegionallyRobustRCStacker, build_predictor
from scripts.build_ruap_audit import build_exposure_table, nearest_neighbor_risk, uniqueness_rate
from scripts.run_proxy_transform_audit import sensitive_feature_columns, transform_features


def test_fmsd_model_scope_matches_final_paper() -> None:
    model_names = [config.name for config in get_fmsd_model_configs(include_tabpfn=True)]
    assert model_names == [
        "logreg_baseline",
        "xgb_baseline",
        "compact_xgb",
        "tabpfn_baseline",
    ]


def test_midas_model_scope_includes_expected_tabular_baselines() -> None:
    model_names = [config.name for config in get_midas_model_configs(include_tabpfn=True)]
    assert model_names == [
        "logreg_baseline",
        "xgb_baseline",
        "lightgbm_baseline",
        "catboost_baseline",
        "histgb_baseline",
        "compact_xgb",
        "tabpfn_baseline",
        "tabicl_baseline",
    ]


def test_spotlight_model_scope_includes_reliability_method() -> None:
    model_names = [config.name for config in get_spotlight_model_configs(include_tabpfn=True)]
    assert model_names == [
        "logreg_baseline",
        "xgb_baseline",
        "lightgbm_baseline",
        "catboost_baseline",
        "histgb_baseline",
        "compact_xgb",
        "tabpfn_baseline",
        "tabicl_baseline",
        "reliability_ensemble",
        "rc_stack",
        "rc_stack_dr",
        "rrc_stack",
    ]


def test_regionally_robust_stacker_builds_expected_region_families() -> None:
    rng = np.random.default_rng(3407)
    X = pd.DataFrame(
        {
            "x0": rng.normal(size=80),
            "x1": rng.normal(size=80),
            "x2": rng.normal(size=80),
        }
    )
    y = pd.Series((X["x0"] + 0.25 * rng.normal(size=80) > 0).astype(int))
    stacker = RegionallyRobustRCStacker(
        base_model_names=("logreg", "rf"),
        n_folds=2,
        n_reliability_regions=4,
        region_strategy="hybrid",
        min_region_size=8,
        max_train_samples=80,
        max_iter=20,
        random_state=3407,
    )
    stacker.fit(X, y)

    assert stacker.n_calibration_region_families_ >= 1
    probabilities = stacker.predict_proba(X)[:, 1]
    assert probabilities.shape == (80,)
    assert np.all(np.isfinite(probabilities))


def test_rrc_stack_config_builds_regionally_robust_predictor() -> None:
    config = [config for config in get_spotlight_model_configs(include_tabpfn=False, include_tabicl=False) if config.name == "rrc_stack"][0]
    predictor = build_predictor(config)
    assert isinstance(predictor, RegionallyRobustRCStacker)
    assert predictor.region_strategy == "hybrid"


def test_rc_stack_logloss_only_ablation_is_calibration_only_control() -> None:
    config = [config for config in get_ablation_model_configs() if config.name == "rc_stack_logloss_only"][0]
    predictor = build_predictor(config)
    assert predictor.brier_weight == 0.0
    assert predictor.ece_weight == 0.0
    assert predictor.cost_weight == 0.0
    assert predictor.balance_weight == 0.0
    assert predictor.base_model_names == MODEL_REGISTRY["rc_stack"].estimator_params["base_model_names"]


def test_proxy_transform_audit_variants_are_deterministic() -> None:
    X = pd.DataFrame(
        {
            "continuous": np.linspace(0.0, 1.0, 20),
            "binary": [0, 1] * 10,
            "nullable_bool": pd.Series([True, False] * 10, dtype="boolean"),
            "other": np.linspace(10.0, 30.0, 20),
        }
    )
    noisy_a = transform_features(X, "numeric_noise_10", seed=123)
    noisy_b = transform_features(X, "numeric_noise_10", seed=123)
    assert noisy_a.equals(noisy_b)
    assert noisy_a["binary"].equals(X["binary"])
    assert not noisy_a["continuous"].equals(X["continuous"])

    laplace_a = transform_features(X, "laplace_noise_20", seed=123)
    laplace_b = transform_features(X, "laplace_noise_20", seed=123)
    assert laplace_a.equals(laplace_b)
    assert laplace_a["binary"].equals(X["binary"])
    assert not laplace_a["continuous"].equals(X["continuous"])

    coarsened = transform_features(X, "coarsen_quartile", seed=123)
    assert coarsened["continuous"].nunique() <= 4

    swapped_a = transform_features(X, "rank_swap_10", seed=123)
    swapped_b = transform_features(X, "rank_swap_10", seed=123)
    assert swapped_a.equals(swapped_b)
    assert swapped_a.shape == X.shape
    assert swapped_a["binary"].equals(X["binary"])

    synth_a = transform_features(X, "synthetic_marginal", seed=123)
    synth_b = transform_features(X, "synthetic_marginal", seed=123)
    assert synth_a.equals(synth_b)
    assert synth_a.shape == X.shape
    assert not synth_a.equals(X)

    noisy_synth_a = transform_features(X, "noisy_synthetic_marginal", seed=123)
    noisy_synth_b = transform_features(X, "noisy_synthetic_marginal", seed=123)
    assert noisy_synth_a.equals(noisy_synth_b)
    assert noisy_synth_a.shape == X.shape
    assert not noisy_synth_a.equals(X)

    masked = transform_features(X, "feature_mask_20", seed=123)
    assert (masked == 0.0).sum().sum() > (X == 0.0).sum().sum()
    assert str(masked["nullable_bool"].dtype) == "boolean"

    sensitive = pd.DataFrame({"sex": [0, 1], "sex_Female": [1, 0], "age": [22, 51], "x": [3, 4]})
    assert sensitive_feature_columns(sensitive, ["sex", "AgeBin"]) == ["age", "sex", "sex_Female"]
    sensitive_masked = transform_features(sensitive, "sensitive_mask", seed=123, sensitive_columns=["sex", "AgeBin"])
    assert sensitive_masked[["sex", "sex_Female", "age"]].sum().sum() == 0


def test_ruap_exposure_metrics_are_bounded() -> None:
    X = pd.DataFrame(
        {
            "x0": [0, 0, 1, 1, 2, 2, 3, 3],
            "x1": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    rate = uniqueness_rate(X)
    assert 0.0 <= rate <= 1.0
    nn_risk = nearest_neighbor_risk(X, seed=3407, max_rows=8)
    assert 0.0 <= nn_risk <= 1.0


def test_ruap_exposure_table_builds_for_local_small_dataset() -> None:
    require_local_dataset("australian_credit")
    table = build_exposure_table(["australian_credit"], ["baseline", "feature_mask_20"], seed=3407)
    assert set(table["Variant"]) == {"baseline", "feature_mask_20"}
    assert table["UniquenessRate"].between(0.0, 1.0).all()
    assert table["NearestNeighborRisk"].between(0.0, 1.0).all()
    assert table["SensitivePredictability"].dropna().between(0.0, 1.0).all()


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
        if get_dataset_config(dataset_name).label_params.get("loader") == "generic_tabular":
            assert not any(any(char in str(column) for char in "[]<") for column in X.columns)
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


def test_reliability_ensemble_runs_in_contract_benchmark() -> None:
    output_root = Path("outputs") / f"contract_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=1,
        output_root=str(output_root),
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
        run_subgroups=False,
    )
    model_configs = [
        config
        for config in get_spotlight_model_configs(include_tabpfn=False, include_tabicl=False)
        if config.name == "reliability_ensemble"
    ]
    result = run_benchmark(
        require_local_dataset("australian_credit"),
        model_configs,
        exp_cfg,
        mode="contract",
    )

    aggregate = result["aggregate_metrics"]
    assert set(aggregate["Model"]) == {"reliability_ensemble"}
    assert {"AUC", "Brier", "ECE (10-bin)", "DecisionCost5x", "n_splits"}.issubset(aggregate.columns)


def test_rc_stack_runs_in_contract_benchmark() -> None:
    output_root = Path("outputs") / f"contract_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=1,
        output_root=str(output_root),
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
        run_subgroups=False,
    )
    model_configs = [
        config
        for config in get_spotlight_model_configs(include_tabpfn=False, include_tabicl=False)
        if config.name == "rc_stack"
    ]
    result = run_benchmark(
        require_local_dataset("australian_credit"),
        model_configs,
        exp_cfg,
        mode="contract",
    )

    aggregate = result["aggregate_metrics"]
    assert set(aggregate["Model"]) == {"rc_stack"}
    assert {"AUC", "Brier", "ECE (10-bin)", "DecisionCost5x", "n_splits"}.issubset(aggregate.columns)


def test_rc_stack_dr_runs_in_contract_benchmark() -> None:
    output_root = Path("outputs") / f"contract_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=1,
        output_root=str(output_root),
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
        run_subgroups=False,
    )
    model_configs = [
        config
        for config in get_spotlight_model_configs(include_tabpfn=False, include_tabicl=False)
        if config.name == "rc_stack_dr"
    ]
    result = run_benchmark(
        require_local_dataset("australian_credit"),
        model_configs,
        exp_cfg,
        mode="contract",
    )

    aggregate = result["aggregate_metrics"]
    assert set(aggregate["Model"]) == {"rc_stack_dr"}
    assert {"AUC", "Brier", "ECE (10-bin)", "DecisionCost5x", "n_splits"}.issubset(aggregate.columns)
