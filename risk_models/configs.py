import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "2")

SEED = 3407
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass
class DatasetConfig:
    name: str
    path: str
    label_type: str = "weak"
    target_column: str = "Risk"
    subgroup_columns: List[str] = field(default_factory=list)
    label_params: Dict[str, Any] = field(default_factory=dict)
    drop_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str
    selector_type: str = "none"
    use_feature_engineering: bool = True
    use_segmenter: bool = False
    predictor_type: str = "xgb"
    use_calibration: bool = True
    use_uncertainty: bool = False
    use_pseudo_labels: bool = False
    selector_k: int = 20
    feature_engineering_max_features: int = 8
    segmenter_type: str = "kmeans"
    segmenter_n_clusters: int = 4
    selector_params: Dict[str, Any] = field(default_factory=dict)
    estimator_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    seed: int = SEED
    n_repeats: int = 20
    test_size: float = 0.20
    val_size: float = 0.20
    calibration_method: str = "temperature"
    run_subgroups: bool = True
    run_weak_label_sensitivity: bool = True
    output_root: str = "outputs"
    save_split_metrics: bool = True
    save_aggregate_metrics: bool = True
    save_shap: bool = False
    save_reliability: bool = True


def clone_dataset_config(config: DatasetConfig, **updates: Any) -> DatasetConfig:
    return replace(config, **updates)


def clone_model_config(config: ModelConfig, **updates: Any) -> ModelConfig:
    return replace(config, **updates)


def clone_experiment_config(config: ExperimentConfig, **updates: Any) -> ExperimentConfig:
    return replace(config, **updates)


XGB_KW = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss",
    random_state=SEED,
)


DATASETS: Dict[str, DatasetConfig] = {
    "taiwan_default": DatasetConfig(
        name="taiwan_default",
        path=str(PROJECT_ROOT / "data" / "taiwan_default.csv"),
        label_type="real",
        target_column="default.payment.next.month",
        subgroup_columns=["SEX", "EDUCATION", "MARRIAGE", "AgeBin"],
    ),
    "give_me_some_credit": DatasetConfig(
        name="give_me_some_credit",
        path=str(PROJECT_ROOT / "data" / "give_me_some_credit.csv"),
        label_type="real",
        target_column="SeriousDlqin2yrs",
        subgroup_columns=["AgeBin"],
    ),
    "german_credit": DatasetConfig(
        name="german_credit",
        path=str(PROJECT_ROOT / "data" / "german_credit.csv"),
        label_type="weak",
        subgroup_columns=["Sex", "Housing", "Age"],
        label_params={"top_frac": 0.30},
    ),
    "australian_credit": DatasetConfig(
        name="australian_credit",
        path=str(PROJECT_ROOT / "data" / "australian_credit.dat"),
        label_type="real",
        target_column="A15",
        subgroup_columns=["A1", "A12"],
    ),
}


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "logreg_baseline": ModelConfig(
        name="logreg_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="logreg",
        use_calibration=True,
    ),
    "rf_baseline": ModelConfig(
        name="rf_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="rf",
        use_calibration=True,
    ),
    "xgb_baseline": ModelConfig(
        name="xgb_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="xgb",
        use_calibration=True,
        estimator_params=dict(XGB_KW),
    ),
    "compact_xgb": ModelConfig(
        name="compact_xgb",
        selector_type="stability",
        use_feature_engineering=True,
        use_segmenter=False,
        predictor_type="xgb",
        use_calibration=True,
        estimator_params=dict(XGB_KW),
    ),
    "compact_xgb_segmented": ModelConfig(
        name="compact_xgb_segmented",
        selector_type="stability",
        use_feature_engineering=True,
        use_segmenter=True,
        predictor_type="xgb",
        use_calibration=True,
        estimator_params=dict(XGB_KW),
    ),
}


# Backward-compatible aliases for the rest of the repo.
DATASET_REGISTRY = DATASETS


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return clone_dataset_config(DATASETS[dataset_name])


def get_default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()


def get_benchmark_model_configs() -> List[ModelConfig]:
    return [clone_model_config(model_cfg) for model_cfg in MODEL_REGISTRY.values()]


def get_tabular_foundation_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig(
            name="tabpfn_baseline",
            selector_type="none",
            use_feature_engineering=False,
            use_segmenter=False,
            predictor_type="tabpfn",
            use_calibration=True,
            estimator_params={
                "device": "auto",
                "n_estimators": 1,
                "max_train_samples": 1024,
            },
        )
    ]


def get_fmsd_model_configs(include_tabpfn: bool = False) -> List[ModelConfig]:
    paper_model_names = ["logreg_baseline", "xgb_baseline", "compact_xgb"]
    configs = [clone_model_config(MODEL_REGISTRY[name]) for name in paper_model_names]
    if include_tabpfn:
        configs.extend(get_tabular_foundation_model_configs())
    return configs

def get_ablation_model_configs() -> List[ModelConfig]:
    compact = clone_model_config(MODEL_REGISTRY["compact_xgb"])
    segmented = clone_model_config(MODEL_REGISTRY["compact_xgb_segmented"])
    return [
        compact,
        segmented,
        clone_model_config(compact, name="compact_xgb_no_selector", selector_type="none"),
        clone_model_config(compact, name="compact_xgb_mi", selector_type="mi"),
        clone_model_config(compact, name="compact_xgb_no_fe", use_feature_engineering=False),
        clone_model_config(segmented, name="compact_xgb_no_segment", use_segmenter=False),
        clone_model_config(compact, name="compact_xgb_no_cal", use_calibration=False),
        clone_model_config(compact, name="compact_xgb_no_uncertainty", use_uncertainty=False),
    ]
