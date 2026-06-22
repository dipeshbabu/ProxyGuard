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

LIGHTGBM_KW = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    objective="binary",
    random_state=SEED,
    verbose=-1,
)

CATBOOST_KW = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Logloss",
    auto_class_weights="Balanced",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
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
    "adult_income": DatasetConfig(
        name="adult_income",
        path=str(PROJECT_ROOT / "data" / "adult_income.csv"),
        label_type="real",
        target_column="income",
        subgroup_columns=["age", "sex", "race"],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            ],
            "positive_values": [">50K"],
            "read_csv_kwargs": {
                "header": None,
                "skipinitialspace": True,
                "names": [
                    "age",
                    "workclass",
                    "fnlwgt",
                    "education",
                    "education_num",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                    "native_country",
                    "income",
                ],
            },
        },
    ),
    "bank_marketing": DatasetConfig(
        name="bank_marketing",
        path=str(PROJECT_ROOT / "data" / "bank_marketing.csv"),
        label_type="real",
        target_column="y",
        subgroup_columns=["age", "job", "marital", "education"],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
            ],
            "zip_member": "bank-full.csv",
            "positive_values": ["yes"],
            "read_csv_kwargs": {"sep": ";"},
        },
    ),
    "compas_recidivism": DatasetConfig(
        name="compas_recidivism",
        path=str(PROJECT_ROOT / "data" / "compas_scores_two_years.csv"),
        label_type="real",
        target_column="two_year_recid",
        subgroup_columns=["age", "sex", "race"],
        drop_columns=[
            "id",
            "name",
            "first",
            "last",
            "dob",
            "compas_screening_date",
            "c_jail_in",
            "c_jail_out",
            "c_case_number",
            "r_case_number",
            "vr_case_number",
            "start",
            "end",
            "event",
            "is_recid",
            "decile_score",
            "score_text",
            "v_decile_score",
            "v_score_text",
        ],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            ],
            "include_columns": [
                "sex",
                "age",
                "age_cat",
                "race",
                "juv_fel_count",
                "juv_misd_count",
                "juv_other_count",
                "priors_count",
                "days_b_screening_arrest",
                "c_days_from_compas",
                "c_charge_degree",
            ],
        },
    ),
    "heart_disease": DatasetConfig(
        name="heart_disease",
        path=str(PROJECT_ROOT / "data" / "statlog_heart.csv"),
        label_type="real",
        target_column="heart_disease",
        subgroup_columns=["age", "sex"],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
            ],
            "positive_values": [2],
            "read_csv_kwargs": {
                "header": None,
                "sep": r"\s+",
                "names": [
                    "age",
                    "sex",
                    "chest_pain",
                    "resting_bp",
                    "cholesterol",
                    "fasting_blood_sugar",
                    "resting_ecg",
                    "max_heart_rate",
                    "exercise_angina",
                    "oldpeak",
                    "st_slope",
                    "major_vessels",
                    "thal",
                    "heart_disease",
                ],
            },
        },
    ),
    "mammographic_mass": DatasetConfig(
        name="mammographic_mass",
        path=str(PROJECT_ROOT / "data" / "mammographic_masses.csv"),
        label_type="real",
        target_column="severity",
        subgroup_columns=["age", "density"],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
            ],
            "read_csv_kwargs": {
                "header": None,
                "na_values": ["?"],
                "names": ["bi_rads", "age", "shape", "margin", "density", "severity"],
            },
        },
    ),
    "breast_cancer_wdbc": DatasetConfig(
        name="breast_cancer_wdbc",
        path=str(PROJECT_ROOT / "data" / "wdbc.csv"),
        label_type="real",
        target_column="diagnosis",
        subgroup_columns=[],
        drop_columns=["id"],
        label_params={
            "loader": "generic_tabular",
            "download_urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            ],
            "positive_values": ["M"],
            "read_csv_kwargs": {
                "header": None,
                "names": [
                    "id",
                    "diagnosis",
                    "radius_mean",
                    "texture_mean",
                    "perimeter_mean",
                    "area_mean",
                    "smoothness_mean",
                    "compactness_mean",
                    "concavity_mean",
                    "concave_points_mean",
                    "symmetry_mean",
                    "fractal_dimension_mean",
                    "radius_se",
                    "texture_se",
                    "perimeter_se",
                    "area_se",
                    "smoothness_se",
                    "compactness_se",
                    "concavity_se",
                    "concave_points_se",
                    "symmetry_se",
                    "fractal_dimension_se",
                    "radius_worst",
                    "texture_worst",
                    "perimeter_worst",
                    "area_worst",
                    "smoothness_worst",
                    "compactness_worst",
                    "concavity_worst",
                    "concave_points_worst",
                    "symmetry_worst",
                    "fractal_dimension_worst",
                ],
            },
        },
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
    "lightgbm_baseline": ModelConfig(
        name="lightgbm_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="lightgbm",
        use_calibration=True,
        estimator_params=dict(LIGHTGBM_KW),
    ),
    "catboost_baseline": ModelConfig(
        name="catboost_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="catboost",
        use_calibration=True,
        estimator_params=dict(CATBOOST_KW),
    ),
    "histgb_baseline": ModelConfig(
        name="histgb_baseline",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="histgb",
        use_calibration=True,
        estimator_params={
            "max_iter": 300,
            "learning_rate": 0.05,
            "l2_regularization": 0.01,
            "random_state": SEED,
        },
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
    "reliability_ensemble": ModelConfig(
        name="reliability_ensemble",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="reliability_ensemble",
        use_calibration=True,
        estimator_params={
            "base_model_names": ("logreg", "xgb", "rf"),
            "validation_size": 0.25,
            "brier_weight": 1.0,
            "ece_weight": 0.5,
            "cost_weight": 0.05,
            "balance_weight": 0.25,
            "fn_cost": 5.0,
            "max_iter": 300,
            "random_state": SEED,
        },
    ),
    "rc_stack": ModelConfig(
        name="rc_stack",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="rc_stack",
        use_calibration=True,
        estimator_params={
            "base_model_names": ("logreg", "xgb", "lightgbm", "catboost", "rf"),
            "n_folds": 3,
            "brier_weight": 1.0,
            "ece_weight": 0.5,
            "cost_weight": 0.05,
            "balance_weight": 0.25,
            "fn_cost": 5.0,
            "max_train_samples": 25000,
            "max_iter": 300,
            "random_state": SEED,
        },
    ),
    "rc_stack_dr": ModelConfig(
        name="rc_stack_dr",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="rc_stack_dr",
        use_calibration=True,
        estimator_params={
            "base_model_names": ("logreg", "xgb", "lightgbm", "catboost", "rf"),
            "n_folds": 3,
            "n_reliability_clusters": 4,
            "group_ece_weight": 0.4,
            "group_brier_weight": 0.2,
            "min_group_size": 20,
            "brier_weight": 1.0,
            "ece_weight": 0.5,
            "cost_weight": 0.05,
            "balance_weight": 0.25,
            "fn_cost": 5.0,
            "max_train_samples": 25000,
            "max_iter": 300,
            "random_state": SEED,
        },
    ),
    "rrc_stack": ModelConfig(
        name="rrc_stack",
        selector_type="none",
        use_feature_engineering=False,
        use_segmenter=False,
        predictor_type="rrc_stack",
        use_calibration=True,
        estimator_params={
            "base_model_names": ("logreg", "xgb", "lightgbm", "catboost", "rf"),
            "n_folds": 3,
            "n_reliability_regions": 4,
            "region_strategy": "hybrid",
            "region_ece_weight": 0.4,
            "region_brier_weight": 0.2,
            "min_region_size": 20,
            "brier_weight": 1.0,
            "ece_weight": 0.5,
            "cost_weight": 0.05,
            "balance_weight": 0.25,
            "fn_cost": 5.0,
            "max_train_samples": 25000,
            "max_iter": 300,
            "random_state": SEED,
        },
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


def get_tabicl_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig(
            name="tabicl_baseline",
            selector_type="none",
            use_feature_engineering=False,
            use_segmenter=False,
            predictor_type="tabicl",
            use_calibration=True,
            estimator_params={
                "device": None,
                "n_estimators": 1,
                "batch_size": 4,
                "max_train_samples": 4096,
            },
        )
    ]


def get_fmsd_model_configs(include_tabpfn: bool = False) -> List[ModelConfig]:
    paper_model_names = ["logreg_baseline", "xgb_baseline", "compact_xgb"]
    configs = [clone_model_config(MODEL_REGISTRY[name]) for name in paper_model_names]
    if include_tabpfn:
        configs.extend(get_tabular_foundation_model_configs())
    return configs


def get_midas_model_configs(include_tabpfn: bool = True, include_tabicl: bool = True) -> List[ModelConfig]:
    paper_model_names = [
        "logreg_baseline",
        "xgb_baseline",
        "lightgbm_baseline",
        "catboost_baseline",
        "histgb_baseline",
        "compact_xgb",
    ]
    configs = [clone_model_config(MODEL_REGISTRY[name]) for name in paper_model_names]
    if include_tabpfn:
        configs.extend(get_tabular_foundation_model_configs())
    if include_tabicl:
        configs.extend(get_tabicl_model_configs())
    return configs


def get_spotlight_model_configs(include_tabpfn: bool = True, include_tabicl: bool = True) -> List[ModelConfig]:
    configs = get_midas_model_configs(include_tabpfn=include_tabpfn, include_tabicl=include_tabicl)
    configs.append(clone_model_config(MODEL_REGISTRY["reliability_ensemble"]))
    configs.append(clone_model_config(MODEL_REGISTRY["rc_stack"]))
    configs.append(clone_model_config(MODEL_REGISTRY["rc_stack_dr"]))
    configs.append(clone_model_config(MODEL_REGISTRY["rrc_stack"]))
    return configs


def get_ablation_model_configs() -> List[ModelConfig]:
    compact = clone_model_config(MODEL_REGISTRY["compact_xgb"])
    segmented = clone_model_config(MODEL_REGISTRY["compact_xgb_segmented"])
    reliability = clone_model_config(MODEL_REGISTRY["reliability_ensemble"])
    return [
        compact,
        segmented,
        clone_model_config(compact, name="compact_xgb_no_selector", selector_type="none"),
        clone_model_config(compact, name="compact_xgb_mi", selector_type="mi"),
        clone_model_config(compact, name="compact_xgb_no_fe", use_feature_engineering=False),
        clone_model_config(segmented, name="compact_xgb_no_segment", use_segmenter=False),
        clone_model_config(compact, name="compact_xgb_no_cal", use_calibration=False),
        clone_model_config(compact, name="compact_xgb_no_uncertainty", use_uncertainty=False),
        reliability,
        clone_model_config(
            reliability,
            name="reliability_ensemble_logloss_only",
            estimator_params={
                **reliability.estimator_params,
                "brier_weight": 0.0,
                "ece_weight": 0.0,
                "cost_weight": 0.0,
                "balance_weight": 0.0,
            },
        ),
        clone_model_config(
            reliability,
            name="reliability_ensemble_no_cost",
            estimator_params={
                **reliability.estimator_params,
                "cost_weight": 0.0,
            },
        ),
        clone_model_config(
            reliability,
            name="reliability_ensemble_no_ece",
            estimator_params={
                **reliability.estimator_params,
                "ece_weight": 0.0,
                "balance_weight": 0.0,
            },
        ),
        clone_model_config(MODEL_REGISTRY["rc_stack"]),
        clone_model_config(
            MODEL_REGISTRY["rc_stack"],
            name="rc_stack_logloss_only",
            estimator_params={
                **MODEL_REGISTRY["rc_stack"].estimator_params,
                "brier_weight": 0.0,
                "ece_weight": 0.0,
                "cost_weight": 0.0,
                "balance_weight": 0.0,
            },
        ),
        clone_model_config(MODEL_REGISTRY["rrc_stack"]),
        clone_model_config(
            MODEL_REGISTRY["rrc_stack"],
            name="rrc_stack_kmeans_regions",
            estimator_params={
                **MODEL_REGISTRY["rrc_stack"].estimator_params,
                "region_strategy": "kmeans",
            },
        ),
        clone_model_config(
            MODEL_REGISTRY["rrc_stack"],
            name="rrc_stack_risk_regions",
            estimator_params={
                **MODEL_REGISTRY["rrc_stack"].estimator_params,
                "region_strategy": "risk",
            },
        ),
        clone_model_config(
            MODEL_REGISTRY["rrc_stack"],
            name="rrc_stack_random_regions",
            estimator_params={
                **MODEL_REGISTRY["rrc_stack"].estimator_params,
                "region_strategy": "random",
            },
        ),
    ]
