from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


def clean_account_levels(series: pd.Series) -> pd.Series:
    return series.fillna("NaN").replace({"NA": "NaN"})


def encode_ordinal_categories(
    df: pd.DataFrame,
    column: str,
    categories: Iterable[str],
) -> pd.Series:
    mapping = {value: idx for idx, value in enumerate(categories)}
    cleaned = clean_account_levels(df[column])
    return cleaned.map(mapping).astype(float)


def make_age_bins(age: pd.Series) -> pd.Series:
    bins = [0, 30, 45, 60, np.inf]
    labels = ["18-29", "30-44", "45-59", "60+"]
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)


def one_hot_encode(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    drop_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    working = df.copy()
    if drop_columns:
        working = working.drop(columns=list(drop_columns), errors="ignore")
    return pd.get_dummies(working, columns=list(categorical_columns), dummy_na=True, drop_first=False)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0,
) -> pd.Series:
    ratio = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def add_missing_indicators(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    enriched = df.copy()
    for column in columns:
        if column in enriched.columns:
            enriched[f"{column}_missing"] = enriched[column].isna().astype(int)
    return enriched


def clip_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_unique: int = 10,
) -> pd.DataFrame:
    clipped = df.copy()
    numeric_columns = list(columns) if columns is not None else clipped.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column not in clipped.columns:
            continue
        series = pd.to_numeric(clipped[column], errors="coerce")
        if series.nunique(dropna=True) < min_unique:
            clipped[column] = series
            continue
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        clipped[column] = series.clip(lower=lower, upper=upper)
    return clipped


def ensure_data_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def download_to_path(urls: Iterable[str], destination: str | Path, timeout: int = 60) -> Path:
    destination_path = ensure_data_dir(destination)
    if destination_path.exists():
        return destination_path

    last_error: Exception | None = None
    for url in urls:
        try:
            with urlopen(url, timeout=timeout) as response:
                destination_path.write_bytes(response.read())
            return destination_path
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to download dataset to {destination_path}. Last error: {last_error}"
    )


def download_member_from_zip(
    urls: Iterable[str],
    destination: str | Path,
    member_name: str,
    timeout: int = 60,
) -> Path:
    destination_path = ensure_data_dir(destination)
    if destination_path.exists():
        return destination_path

    last_error: Exception | None = None
    for url in urls:
        try:
            with urlopen(url, timeout=timeout) as response:
                archive_bytes = response.read()
            with ZipFile(BytesIO(archive_bytes)) as archive:
                with archive.open(member_name) as member:
                    destination_path.write_bytes(member.read())
            return destination_path
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to download '{member_name}' to {destination_path}. Last error: {last_error}"
    )
