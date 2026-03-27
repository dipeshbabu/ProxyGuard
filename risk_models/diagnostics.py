from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def warn_if_suspicious_metrics(metrics: Dict[str, float], model_name: str) -> List[str]:
    warnings = []
    auc = metrics.get("AUC", np.nan)
    acc = metrics.get("Accuracy", np.nan)
    ece = metrics.get("ECE", np.nan)
    if np.isfinite(auc) and np.isfinite(acc) and ((auc >= 0.98 and acc >= 0.98) or auc >= 0.999):
        warnings.append(
            f"{model_name}: extremely high AUC/Accuracy ({auc:.3f}/{acc:.3f}); review for leakage."
        )
    if np.isfinite(ece) and np.isfinite(auc) and ece < 0.02 and auc > 0.95:
        warnings.append(
            f"{model_name}: unusually low calibration error with very high AUC ({auc:.3f}); verify label leakage."
        )
    return warnings


def run_shuffle_label_sanity(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, seed: int):
    shuffled = y_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)
    model.fit(X_train_reset, shuffled)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    return {"AUC": float(auc)}


def validate_leakage_columns(feature_columns: List[str], leakage_columns: List[str]) -> List[str]:
    present = []
    for column in feature_columns:
        if column in leakage_columns:
            present.append(column)
    return present
