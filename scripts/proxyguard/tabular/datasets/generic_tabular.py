from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from scripts.proxyguard.tabular.config import DatasetConfig
from scripts.proxyguard.tabular.datasets.utils import (
    clip_numeric_columns,
    download_member_from_zip,
    download_to_path,
    make_age_bins,
    one_hot_encode,
)


def _ensure_generic_file(config: DatasetConfig) -> Path:
    path = Path(config.path)
    if path.exists():
        return path

    urls = config.label_params.get("download_urls", [])
    if not urls:
        raise FileNotFoundError(
            f"{config.name} data file is missing at {path}. "
            "Provide the file locally or add download_urls to the dataset config."
        )

    zip_member = config.label_params.get("zip_member")
    if zip_member:
        return download_member_from_zip(urls, path, zip_member)
    return download_to_path(urls, path)


def _read_generic_frame(config: DatasetConfig) -> pd.DataFrame:
    path = _ensure_generic_file(config)
    read_kwargs = dict(config.label_params.get("read_csv_kwargs", {}))
    return pd.read_csv(path, **read_kwargs)


def _coerce_target(series: pd.Series, config: DatasetConfig) -> pd.Series:
    positive_values = config.label_params.get("positive_values")
    if positive_values is not None:
        positives = {str(value).strip() for value in positive_values}
        return series.astype(str).str.strip().isin(positives).astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique_values = sorted(numeric.dropna().unique().tolist())
        if set(unique_values).issubset({0, 1}):
            return numeric.astype(int)
    raise ValueError(
        f"{config.name}: target '{config.target_column}' is not binary. "
        "Set label_params['positive_values'] for generic loading."
    )


def _sanitize_feature_columns(columns: pd.Index) -> list[str]:
    sanitized_columns: list[str] = []
    seen: dict[str, int] = {}
    for column in columns:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(column)).strip("_")
        if not sanitized:
            sanitized = "feature"
        if sanitized[0].isdigit():
            sanitized = f"f_{sanitized}"

        count = seen.get(sanitized, 0)
        seen[sanitized] = count + 1
        if count:
            sanitized = f"{sanitized}_{count}"
        sanitized_columns.append(sanitized)
    return sanitized_columns


def load_generic_tabular(config: DatasetConfig) -> Dict[str, Any]:
    raw_df = _read_generic_frame(config)
    if config.target_column not in raw_df.columns:
        available = ", ".join(map(str, raw_df.columns[:20]))
        raise KeyError(f"{config.name}: target '{config.target_column}' missing. First columns: {available}")

    working = raw_df.copy()
    y = _coerce_target(working[config.target_column], config)
    X = working.drop(columns=[config.target_column], errors="ignore")
    include_columns = config.label_params.get("include_columns")
    if include_columns:
        missing_columns = [column for column in include_columns if column not in X.columns]
        if missing_columns:
            raise KeyError(f"{config.name}: include_columns missing from data: {missing_columns}")
        X = X[list(include_columns)]
    if config.drop_columns:
        X = X.drop(columns=config.drop_columns, errors="ignore")

    categorical_columns = config.label_params.get("categorical_columns")
    if categorical_columns == "auto" or categorical_columns is None:
        categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in set(categorical_columns)]
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = clip_numeric_columns(X, numeric_columns, lower_q=0.01, upper_q=0.99, min_unique=8)
    X = one_hot_encode(X, categorical_columns)
    X.columns = _sanitize_feature_columns(X.columns)

    subgroup_frame = pd.DataFrame(index=raw_df.index)
    for column in config.subgroup_columns:
        if column in raw_df.columns:
            if column.lower() == "age":
                subgroup_frame["AgeBin"] = make_age_bins(pd.to_numeric(raw_df[column], errors="coerce"))
            else:
                subgroup_frame[column] = raw_df[column].astype(str).fillna("missing")

    metadata = {
        "dataset_name": config.name,
        "label_type": config.label_type,
        "label_name": config.target_column,
        "label_params": dict(config.label_params),
        "numeric_cols": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": [column for column in X.columns if column not in X.select_dtypes(include=[np.number]).columns],
        "subgroup_frame": subgroup_frame,
        "raw_df": raw_df,
        "preprocessing_notes": [
            "generic binary-tabular loader",
            "coerce numeric columns and winsorize heavy-tailed numeric values",
            "one-hot encode object/category/bool columns",
        ],
    }
    return {"X": X, "y": y, "metadata": metadata}
