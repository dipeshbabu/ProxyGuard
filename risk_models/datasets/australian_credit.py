from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from risk_models.configs import DatasetConfig
from risk_models.datasets.utils import clip_numeric_columns, download_member_from_zip, safe_ratio

AUSTRALIAN_CREDIT_URLS = [
    "https://archive.ics.uci.edu/static/public/143/statlog%2Baustralian%2Bcredit%2Bapproval.zip",
]

AUSTRALIAN_COLUMNS = [f"A{i}" for i in range(1, 16)]
CATEGORICAL_COLUMNS = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
NUMERIC_COLUMNS = ["A2", "A3", "A7", "A10", "A13", "A14"]
TARGET_COLUMN = "A15"


def ensure_australian_credit_file(path: str) -> str:
    return str(
        download_member_from_zip(
            AUSTRALIAN_CREDIT_URLS,
            destination=path,
            member_name="australian.dat",
        )
    )


def load_raw_australian_credit(path: str) -> pd.DataFrame:
    local_path = ensure_australian_credit_file(path)
    df = pd.read_csv(local_path, sep=r"\s+", header=None, names=AUSTRALIAN_COLUMNS, engine="python")
    df = df.replace("?", np.nan)
    return df


def preprocess_australian_credit(df: pd.DataFrame, config: DatasetConfig) -> Dict[str, Any]:
    working = df.copy()
    target_column = config.target_column or TARGET_COLUMN
    for column in CATEGORICAL_COLUMNS:
        working[column] = working[column].astype("string").fillna("Missing")
    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working[column] = working[column].fillna(working[column].median())

    y = pd.to_numeric(working[target_column], errors="coerce").fillna(0).astype(int).replace({2: 0})
    X = working.drop(columns=[target_column], errors="ignore")
    if config.drop_columns:
        X = X.drop(columns=config.drop_columns, errors="ignore")

    X["A14_to_A2"] = safe_ratio(X["A14"], X["A2"] + 1.0)
    X["A10_to_A14"] = safe_ratio(X["A10"], X["A14"] + 1.0)
    X["A3_times_A7"] = X["A3"] * X["A7"]
    X["A2_times_A3"] = X["A2"] * X["A3"]
    X = clip_numeric_columns(X, NUMERIC_COLUMNS + ["A14_to_A2", "A10_to_A14", "A3_times_A7", "A2_times_A3"], min_unique=8)

    subgroup_frame = pd.DataFrame(index=working.index)
    for column in config.subgroup_columns:
        if column in working.columns:
            subgroup_frame[column] = working[column].astype(str)

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, dummy_na=False, drop_first=False)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    metadata = {
        "dataset_name": config.name,
        "label_type": config.label_type,
        "label_name": target_column,
        "label_params": dict(config.label_params),
        "numeric_cols": numeric_cols,
        "categorical_cols": [column for column in X.columns if column not in numeric_cols],
        "subgroup_frame": subgroup_frame,
        "raw_df": df,
        "download_urls": AUSTRALIAN_CREDIT_URLS,
        "citation": "Quinlan, R. (1987). Statlog (Australian Credit Approval) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59012.",
        "preprocessing_notes": [
            "coerce numeric columns and median-impute small-table missingness",
            "derive simple scale-free ratio and interaction features",
            "winsorize continuous variables before one-hot encoding categoricals",
        ],
    }
    return {"X": X, "y": y, "metadata": metadata}


def load_australian_credit(config: DatasetConfig) -> Dict[str, Any]:
    raw_df = load_raw_australian_credit(config.path)
    return preprocess_australian_credit(raw_df, config)
