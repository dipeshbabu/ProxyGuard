from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from scripts.proxyguard.tabular.config import DatasetConfig
from scripts.proxyguard.tabular.datasets.utils import clip_numeric_columns, download_to_path, make_age_bins, safe_ratio

# Public CSV mirror of the UCI Default of Credit Card Clients dataset.
TAIWAN_DEFAULT_URLS = [
    "https://huggingface.co/datasets/scikit-learn/credit-card-clients/resolve/main/UCI_Credit_Card.csv",
]


def ensure_taiwan_default_file(path: str) -> str:
    return str(download_to_path(TAIWAN_DEFAULT_URLS, path))


def load_raw_taiwan_default(path: str) -> pd.DataFrame:
    local_path = ensure_taiwan_default_file(path)
    df = pd.read_csv(local_path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def preprocess_taiwan_default(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    sex_map = {1: "male", 2: "female"}
    education_map = {
        1: "graduate_school",
        2: "university",
        3: "high_school",
        4: "other",
    }
    marriage_map = {1: "married", 2: "single", 3: "other"}
    working["SEX_GROUP"] = working["SEX"].map(sex_map).fillna("other")
    working["EDUCATION_GROUP"] = working["EDUCATION"].map(education_map).fillna("other")
    working["MARRIAGE_GROUP"] = working["MARRIAGE"].map(marriage_map).fillna("other")

    pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_columns = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    payment_columns = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    pay_frame = working[pay_columns].fillna(0)
    bill_frame = working[bill_columns].fillna(0)
    payment_frame = working[payment_columns].fillna(0)
    limit_balance = working["LIMIT_BAL"].fillna(working["LIMIT_BAL"].median())

    working["MonthsWithDelay"] = (pay_frame > 0).sum(axis=1)
    working["MonthsWithSevereDelay"] = (pay_frame >= 2).sum(axis=1)
    working["MaxDelinquencyStatus"] = pay_frame.max(axis=1)
    working["MeanDelinquencyStatus"] = pay_frame.mean(axis=1)
    working["TotalBillAmount"] = bill_frame.sum(axis=1)
    working["TotalPaymentAmount"] = payment_frame.sum(axis=1)
    working["MeanBillAmount"] = bill_frame.mean(axis=1)
    working["MeanPaymentAmount"] = payment_frame.mean(axis=1)
    working["CurrentBalanceToLimit"] = safe_ratio(working["BILL_AMT1"].fillna(0), limit_balance)
    working["AverageBillToLimit"] = safe_ratio(working["MeanBillAmount"], limit_balance)
    working["PaymentToBillRatio"] = safe_ratio(working["TotalPaymentAmount"], working["TotalBillAmount"].abs() + 1.0)
    working["RecentBillTrend"] = working["BILL_AMT1"].fillna(0) - working["BILL_AMT6"].fillna(0)
    working["RecentPaymentTrend"] = working["PAY_AMT1"].fillna(0) - working["PAY_AMT6"].fillna(0)
    working["LogLimitBalance"] = np.log1p(limit_balance.clip(lower=0))

    clip_columns = [
        "LIMIT_BAL",
        "AGE",
        "MonthsWithDelay",
        "MonthsWithSevereDelay",
        "MaxDelinquencyStatus",
        "MeanDelinquencyStatus",
        "TotalBillAmount",
        "TotalPaymentAmount",
        "MeanBillAmount",
        "MeanPaymentAmount",
        "CurrentBalanceToLimit",
        "AverageBillToLimit",
        "PaymentToBillRatio",
        "RecentBillTrend",
        "RecentPaymentTrend",
        "LogLimitBalance",
    ] + bill_columns + payment_columns
    working = clip_numeric_columns(working, clip_columns, lower_q=0.01, upper_q=0.99, min_unique=8)

    working = pd.get_dummies(
        working,
        columns=["SEX_GROUP", "EDUCATION_GROUP", "MARRIAGE_GROUP"],
        dummy_na=False,
        drop_first=False,
    )
    return working


def build_taiwan_default_bundle(df: pd.DataFrame, config: DatasetConfig) -> Dict[str, Any]:
    subgroup_source = df.copy()
    working = preprocess_taiwan_default(df)
    target_column = config.target_column
    y = working[target_column].astype(int)
    X = working.drop(columns=[target_column], errors="ignore")
    if config.drop_columns:
        X = X.drop(columns=config.drop_columns, errors="ignore")

    subgroup_frame = pd.DataFrame(index=subgroup_source.index)
    if "SEX" in subgroup_source.columns:
        subgroup_frame["SEX"] = subgroup_source["SEX"].map({1: "male", 2: "female"}).fillna("other")
    if "EDUCATION" in subgroup_source.columns:
        subgroup_frame["EDUCATION"] = subgroup_source["EDUCATION"].map({
            1: "graduate_school",
            2: "university",
            3: "high_school",
            4: "other",
        }).fillna("other")
    if "MARRIAGE" in subgroup_source.columns:
        subgroup_frame["MARRIAGE"] = subgroup_source["MARRIAGE"].map({1: "married", 2: "single", 3: "other"}).fillna("other")
    if "AGE" in subgroup_source.columns:
        subgroup_frame["AgeBin"] = make_age_bins(subgroup_source["AGE"])

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
        "download_urls": TAIWAN_DEFAULT_URLS,
        "preprocessing_notes": [
            "coerce all columns to numeric and normalize demographic code groups",
            "derive delinquency, balance, payment, and utilization-style summary features",
            "winsorize extreme bill and payment tails",
            "one-hot encode demographic code groups for linear and tree baselines",
        ],
    }
    return {"X": X, "y": y, "metadata": metadata}


def load_taiwan_default(config: DatasetConfig):
    raw_df = load_raw_taiwan_default(config.path)
    return build_taiwan_default_bundle(raw_df, config)
