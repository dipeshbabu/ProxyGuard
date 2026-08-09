from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from scripts.proxyguard.tabular.config import SEED
from scripts.proxyguard.tabular.datasets.utils import clip_numeric_columns, download_to_path, safe_ratio

GERMAN_CREDIT_URLS = [
    "https://gist.githubusercontent.com/eyuelberga/dc4f2fab53f71731a7ad54bee42687fe/raw/f690908cced81953aeca15a84d71350e56383de7/german_credit_data.csv",
]


LEAKAGE_RAW_COLUMNS = [
    "Credit amount",
    "Duration",
    "Monthly_Revenue",
]

LEAKAGE_PREFIXES = [
    "Business_Type_",
]

ORDINAL_SPECS = {
    "Saving accounts": ["NaN", "little", "moderate", "quite rich", "rich"],
    "Checking account": ["NaN", "little", "moderate", "rich"],
}


def load_raw_german_credit(path: str) -> pd.DataFrame:
    local_path = download_to_path(GERMAN_CREDIT_URLS, path)
    return pd.read_csv(local_path, index_col=0)


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Business_Age"] = df["Age"].apply(lambda x: max(0, x - 22))
    df["Employees"] = df["Job"].map(
        {0: "1-5", 1: "1-5", 2: "5-10", 3: "10-20"}
    ).astype("category")
    df["Monthly_Revenue"] = df["Credit amount"] / df["Duration"].replace(0, 1)
    df["Business_Type"] = pd.cut(
        df["Credit amount"],
        bins=[0, 5000, 20000, np.inf],
        labels=["Micro", "Small", "Medium"],
    )
    df["JobSkillTier"] = df["Job"].map({0: "unskilled", 1: "semi_skilled", 2: "skilled", 3: "highly_skilled"}).astype("category")
    df["IsSeniorApplicant"] = (df["Age"] >= 50).astype(int)
    df["HasCheckingAccount"] = (~df["Checking account"].fillna("NaN").replace("NA", "NaN").eq("NaN")).astype(int)
    df["HasSavingsAccount"] = (~df["Saving accounts"].fillna("NaN").replace("NA", "NaN").eq("NaN")).astype(int)
    return clip_numeric_columns(df, ["Age", "Business_Age", "Monthly_Revenue"], min_unique=8)


def create_credit_ratio_label(
    df: pd.DataFrame,
    top_frac: float = 0.30,
    threshold_shift: float = 0.0,
    noise_rate: float = 0.0,
    random_state: int = SEED,
) -> pd.DataFrame:
    df = df.copy()
    credit_ratio = df["Credit amount"] / df["Duration"].replace(0, 1)
    base_threshold = credit_ratio.quantile(1.0 - top_frac)
    shifted_threshold = base_threshold + float(threshold_shift) * float(credit_ratio.std(ddof=0))
    labels = (credit_ratio >= shifted_threshold).astype(int)

    if noise_rate > 0.0:
        rng = np.random.default_rng(random_state)
        n_flip = int(round(len(labels) * float(noise_rate)))
        if n_flip > 0:
            flip_index = rng.choice(len(labels), size=n_flip, replace=False)
            labels = labels.to_numpy(copy=True)
            labels[flip_index] = 1 - labels[flip_index]
            labels = pd.Series(labels, index=df.index)

    df["Risk"] = labels.astype(int)
    return df


def encode_ordinal_finance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, categories in ORDINAL_SPECS.items():
        df[col] = df[col].fillna("NaN").replace("NA", "NaN")
        enc = OrdinalEncoder(
            categories=[categories],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        df[col] = enc.fit_transform(df[[col]])
    return df


def one_hot_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df,
        columns=["Sex", "Housing", "Purpose", "Employees", "Business_Type", "JobSkillTier"],
        dummy_na=True,
        drop_first=False,
    )


def drop_leakage_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    to_drop = [column for column in LEAKAGE_RAW_COLUMNS if column in X.columns]
    to_drop += [column for column in X.columns if any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES)]
    return X.drop(columns=to_drop, errors="ignore")


def build_age_bins(age_series: pd.Series) -> pd.Series:
    return pd.cut(
        age_series,
        bins=[18, 30, 40, 50, 65, np.inf],
        labels=["18-29", "30-39", "40-49", "50-64", "65+"],
        right=False,
    ).astype(str)


def load_german_credit(
    path: str,
    top_frac: float = 0.30,
    threshold_shift: float = 0.0,
    noise_rate: float = 0.0,
    random_state: int = SEED,
    drop_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw_df = load_raw_german_credit(path)
    df = add_business_features(raw_df)
    df = create_credit_ratio_label(
        df,
        top_frac=top_frac,
        threshold_shift=threshold_shift,
        noise_rate=noise_rate,
        random_state=random_state,
    )
    df = encode_ordinal_finance_columns(df)
    df = one_hot_business_columns(df)

    y = df["Risk"].astype(int)
    X = df.drop(columns=["Risk"], errors="ignore")
    X = drop_leakage_columns(X)
    X["AgePerJobTier"] = safe_ratio(X["Age"], X["Job"] + 1.0)
    X["AccountBufferScore"] = X["Saving accounts"].fillna(0) + X["Checking account"].fillna(0)
    X = clip_numeric_columns(X, ["Age", "Business_Age", "AgePerJobTier", "AccountBufferScore"], min_unique=8)
    if drop_columns:
        X = X.drop(columns=list(drop_columns), errors="ignore")

    subgroup_frame = pd.DataFrame(index=raw_df.index)
    subgroup_frame["Sex"] = raw_df["Sex"].astype(str)
    subgroup_frame["Housing"] = raw_df["Housing"].astype(str)
    subgroup_frame["AgeBin"] = build_age_bins(raw_df["Age"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    metadata = {
        "dataset_name": "german_credit",
        "label_type": "weak",
        "label_name": "Risk",
        "label_params": {
            "top_frac": top_frac,
            "threshold_shift": threshold_shift,
            "noise_rate": noise_rate,
            "random_state": random_state,
        },
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "subgroup_frame": subgroup_frame,
        "raw_df": raw_df,
        "download_urls": GERMAN_CREDIT_URLS,
        "preprocessing_notes": [
            "encode checking and savings account levels ordinally with explicit missing handling",
            "derive non-leakage applicant profile features from age, job, and account availability",
            "drop raw and derived leakage proxies tied to weak-label construction",
            "winsorize continuous applicant profile variables",
        ],
    }

    return {
        "X": X,
        "y": y,
        "metadata": metadata,
    }
