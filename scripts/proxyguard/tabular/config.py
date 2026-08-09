from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")

SEED = 3407
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DatasetConfig:
    name: str
    path: str
    label_type: str = "weak"
    target_column: str = "Risk"
    subgroup_columns: list[str] = field(default_factory=list)
    label_params: dict[str, Any] = field(default_factory=dict)
    drop_columns: list[str] = field(default_factory=list)


DATASET_REGISTRY: dict[str, DatasetConfig] = {
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
                "names": [
                    "bi_rads",
                    "age",
                    "shape",
                    "margin",
                    "density",
                    "severity",
                ],
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

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return replace(DATASET_REGISTRY[dataset_name])
