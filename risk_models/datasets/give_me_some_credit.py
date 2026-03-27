from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from risk_models.configs import DatasetConfig
from risk_models.datasets.utils import (
    add_missing_indicators,
    clip_numeric_columns,
    download_to_path,
    make_age_bins,
    safe_ratio,
)

# Public mirror of the Kaggle "Give Me Some Credit" training file.
GIVE_ME_SOME_CREDIT_URLS = [
    "https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/GiveMeSomeCredit/CreditScoring.csv",
    "https://github.com/ChicagoBoothML/MLClassData/raw/master/GiveMeSomeCredit/CreditScoring.csv",
]


def ensure_give_me_some_credit_file(path: str) -> str:
    return str(download_to_path(GIVE_ME_SOME_CREDIT_URLS, path))


def load_raw_give_me_some_credit(path: str) -> pd.DataFrame:
    local_path = ensure_give_me_some_credit_file(path)
    df = pd.read_csv(local_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def preprocess_give_me_some_credit(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    numeric_columns = working.columns.tolist()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)
    working["age"] = working["age"].where(working["age"] >= 18)
    working["NumberOfDependents"] = working["NumberOfDependents"].where(working["NumberOfDependents"] >= 0)
    working["MonthlyIncome"] = working["MonthlyIncome"].where(working["MonthlyIncome"] >= 0)
    working["DebtRatio"] = working["DebtRatio"].where(working["DebtRatio"] >= 0)
    working["RevolvingUtilizationOfUnsecuredLines"] = working["RevolvingUtilizationOfUnsecuredLines"].where(
        working["RevolvingUtilizationOfUnsecuredLines"] >= 0
    )
    working = add_missing_indicators(working, ["MonthlyIncome", "NumberOfDependents"])

    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    dependents = working["NumberOfDependents"].fillna(0)
    income = working["MonthlyIncome"].fillna(working["MonthlyIncome"].median())
    debt_ratio = working["DebtRatio"].fillna(working["DebtRatio"].median())
    utilization = working["RevolvingUtilizationOfUnsecuredLines"].fillna(
        working["RevolvingUtilizationOfUnsecuredLines"].median()
    )
    late_total = working[late_cols].fillna(0).sum(axis=1)

    working["LateCountTotal"] = late_total
    working["HasSeriousDelinquency"] = (working["NumberOfTimes90DaysLate"].fillna(0) > 0).astype(int)
    working["HasAnyDelinquency"] = (late_total > 0).astype(int)
    working["IncomePerDependent"] = safe_ratio(income, dependents + 1.0)
    working["DebtPerDependent"] = safe_ratio(debt_ratio, dependents + 1.0)
    working["UtilizationToDebtRatio"] = safe_ratio(utilization, debt_ratio + 1e-6)
    working["IncomeAgeRatio"] = safe_ratio(income, working["age"].fillna(working["age"].median()) + 1.0)
    working["LogMonthlyIncome"] = np.log1p(income.clip(lower=0))
    working["LogDebtRatio"] = np.log1p(debt_ratio.clip(lower=0))
    working["LogUtilization"] = np.log1p(utilization.clip(lower=0))

    clip_columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "LateCountTotal",
        "IncomePerDependent",
        "DebtPerDependent",
        "UtilizationToDebtRatio",
        "IncomeAgeRatio",
        "LogMonthlyIncome",
        "LogDebtRatio",
        "LogUtilization",
    ] + late_cols
    working = clip_numeric_columns(working, clip_columns, lower_q=0.01, upper_q=0.99, min_unique=8)
    return working


def build_give_me_some_credit_bundle(df: pd.DataFrame, config: DatasetConfig) -> Dict[str, Any]:
    working = preprocess_give_me_some_credit(df)
    target_column = config.target_column
    y = working[target_column].astype(int)
    X = working.drop(columns=[target_column], errors="ignore")
    if config.drop_columns:
        X = X.drop(columns=config.drop_columns, errors="ignore")

    subgroup_frame = pd.DataFrame(index=working.index)
    if "age" in working.columns:
        subgroup_frame["AgeBin"] = make_age_bins(working["age"])

    metadata = {
        "dataset_name": config.name,
        "label_type": config.label_type,
        "label_name": target_column,
        "label_params": dict(config.label_params),
        "numeric_cols": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": [column for column in X.columns if column not in X.select_dtypes(include=[np.number]).columns],
        "subgroup_frame": subgroup_frame,
        "raw_df": df,
        "download_urls": GIVE_ME_SOME_CREDIT_URLS,
        "preprocessing_notes": [
            "coerce numeric credit fields and replace inf with NaN",
            "add missing indicators for income and dependents",
            "derive delinquency summary and income/debt ratio features",
            "log-transform and winsorize heavy-tailed numeric variables",
        ],
    }
    return {"X": X, "y": y, "metadata": metadata}


def load_give_me_some_credit(config: DatasetConfig):
    raw_df = load_raw_give_me_some_credit(config.path)
    return build_give_me_some_credit_bundle(raw_df, config)
