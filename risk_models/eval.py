from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n_obs = len(y_true)
    for bin_id in range(n_bins):
        mask = bin_idx == bin_id
        if not mask.any():
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (mask.sum() / n_obs) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_adaptive_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if len(y_true) == 0:
        return np.nan
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    bins = np.array_split(np.arange(len(y_sorted)), min(n_bins, len(y_sorted)))
    ece = 0.0
    n_obs = len(y_sorted)
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        avg_conf = p_sorted[bin_idx].mean()
        avg_acc = y_sorted[bin_idx].mean()
        ece += (len(bin_idx) / n_obs) * abs(avg_acc - avg_conf)
    return float(ece)


def calibration_slope_intercept(y_true, y_prob) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(y_prob / (1.0 - y_prob))

    def objective(params):
        intercept, slope = params
        linear_term = np.clip(intercept + slope * logits, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear_term))
        return log_loss(y_true, np.clip(fitted, 1e-6, 1 - 1e-6))

    result = minimize(objective, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    if not result.success:
        return np.nan, np.nan
    intercept, slope = result.x
    return float(slope), float(intercept)


def best_f1_threshold_from_val(y_val, p_val):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        score = f1_score(y_val, (np.asarray(p_val) >= threshold).astype(int), zero_division=0)
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def expected_decision_cost(y_true, y_prob, threshold: float, fn_cost: float = 5.0, fp_cost: float = 1.0) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    return float((fn_cost * false_negatives + fp_cost * false_positives) / max(1, len(y_true)))


def best_cost_threshold_from_val(y_val, p_val, fn_cost: float = 5.0, fp_cost: float = 1.0):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_cost = np.inf
    for threshold in thresholds:
        cost = expected_decision_cost(y_val, p_val, threshold, fn_cost=fn_cost, fp_cost=fp_cost)
        if np.isfinite(cost) and cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)
    return best_threshold, best_cost


class TemperatureScaler:
    def __init__(self):
        self.temperature_ = 1.0

    @staticmethod
    def _temperature_from_log(log_temperature):
        return float(np.exp(np.clip(log_temperature, -5.0, 5.0)))

    @staticmethod
    def _logit(probabilities):
        clipped = np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
        return np.log(clipped / (1 - clipped))

    def fit(self, p_val, y_val):
        logits = self._logit(p_val)

        def objective(log_temperature):
            temperature = self._temperature_from_log(log_temperature[0])
            scaled_logits = np.clip(logits / temperature, -30.0, 30.0)
            calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
            return log_loss(y_val, np.clip(calibrated, 1e-6, 1 - 1e-6))

        result = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B")
        self.temperature_ = self._temperature_from_log(result.x[0]) if result.success else 1.0
        return self

    def transform(self, probabilities):
        logits = self._logit(probabilities)
        scaled_logits = np.clip(logits / self.temperature_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def fit_calibrator(p_val, y_val, method: Optional[str]):
    if method in (None, "none"):
        return None
    if method == "temperature":
        return TemperatureScaler().fit(p_val, y_val)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator, probabilities):
    if calibrator is None:
        return np.asarray(probabilities)
    return calibrator.transform(probabilities)


def evaluate_predictions(
    y_true,
    y_prob,
    threshold: float = 0.5,
    feature_count: Optional[int] = None,
    train_time: Optional[float] = None,
    inference_time: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)

    metrics = {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
        "ECE (10-bin)": compute_ece(y_true, y_prob, n_bins=10),
        "ECE (15-bin)": compute_ece(y_true, y_prob, n_bins=15),
        "ECE (20-bin)": compute_ece(y_true, y_prob, n_bins=20),
        "AdaptiveECE (10-bin)": compute_adaptive_ece(y_true, y_prob, n_bins=10),
        "ECE": compute_ece(y_true, y_prob, n_bins=10),
        "LogLoss": log_loss(y_true, y_prob),
        "CalibrationSlope": slope,
        "CalibrationIntercept": intercept,
        "Threshold": float(threshold),
        "Support": int(len(y_true)),
        "PositiveRate": float(y_true.mean()),
    }
    for fn_cost in (2.0, 5.0, 10.0, 20.0):
        cost_name = f"DecisionCost{int(fn_cost)}x"
        base_cost = max(1e-12, fn_cost * float(y_true.mean()))
        metrics[cost_name] = expected_decision_cost(y_true, y_prob, threshold, fn_cost=fn_cost, fp_cost=1.0)
        metrics[f"{cost_name}RelApproveAll"] = metrics[cost_name] / base_cost
    if feature_count is not None:
        metrics["FeatureCount"] = int(feature_count)
    if train_time is not None:
        metrics["TrainTimeSec"] = float(train_time)
    if inference_time is not None:
        metrics["InferenceTimeSec"] = float(inference_time)
    if peak_memory_mb is not None:
        metrics["PeakMemoryMB"] = float(peak_memory_mb)
    return metrics


def evaluate_subgroups(y_true, y_prob, subgroup_values, threshold: float = 0.5, subgroup_name: str = "subgroup") -> pd.DataFrame:
    y_true = pd.Series(y_true)
    y_prob = pd.Series(y_prob, index=y_true.index)
    subgroup_series = pd.Series(subgroup_values, index=y_true.index).fillna("missing")

    rows = []
    for subgroup_value, mask in subgroup_series.groupby(subgroup_series).groups.items():
        subgroup_y = y_true.loc[mask]
        subgroup_p = y_prob.loc[mask]
        if subgroup_y.nunique() < 2:
            auc = np.nan
            auprc = np.nan
            slope = np.nan
            intercept = np.nan
        else:
            auc = roc_auc_score(subgroup_y, subgroup_p)
            auprc = average_precision_score(subgroup_y, subgroup_p)
            slope, intercept = calibration_slope_intercept(subgroup_y, subgroup_p)
        subgroup_pred = (subgroup_p >= threshold).astype(int)
        rows.append(
            {
                "SubgroupName": subgroup_name,
                "SubgroupValue": subgroup_value,
                "Support": int(len(subgroup_y)),
                "PositiveRate": float(subgroup_y.mean()),
                "AUC": auc,
                "AUPRC": auprc,
                "Precision": precision_score(subgroup_y, subgroup_pred, zero_division=0),
                "Recall": recall_score(subgroup_y, subgroup_pred, zero_division=0),
                "Brier": brier_score_loss(subgroup_y, subgroup_p),
                "ECE": compute_ece(subgroup_y, subgroup_p, n_bins=10),
                "CalibrationSlope": slope,
                "CalibrationIntercept": intercept,
                "Threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_AGGREGATE_METRICS = [
    "AUC",
    "AUPRC",
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "Brier",
    "ECE (10-bin)",
    "ECE (15-bin)",
    "ECE (20-bin)",
    "AdaptiveECE (10-bin)",
    "LogLoss",
    "CalibrationSlope",
    "CalibrationIntercept",
    "DecisionCost2x",
    "DecisionCost2xRelApproveAll",
    "DecisionCost5x",
    "DecisionCost5xRelApproveAll",
    "DecisionCost10x",
    "DecisionCost10xRelApproveAll",
    "DecisionCost20x",
    "DecisionCost20xRelApproveAll",
    "FeatureCount",
    "PreprocessTimeSec",
    "ModelFitTimeSec",
    "CalibrationTimeSec",
    "TrainTimeSec",
    "InferenceTimeSec",
    "PeakMemoryMB",
]


def aggregate_metrics(
    split_metrics: pd.DataFrame,
    confidence_level: float = 0.95,
    metric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (metric_columns or DEFAULT_AGGREGATE_METRICS)
        if column in split_metrics.columns and pd.api.types.is_numeric_dtype(split_metrics[column])
    ]
    grouped_rows = []
    alpha = 1.0 - confidence_level
    z_value = norm.ppf(1 - alpha / 2.0)
    group_columns = ["Dataset", "Model"] if "Dataset" in split_metrics.columns else ["Model"]

    for group_key, group in split_metrics.groupby(group_columns):
        if len(group_columns) == 2:
            dataset_name, model_name = group_key
            row = {"Dataset": dataset_name, "Model": model_name}
        else:
            model_name = group_key[0] if isinstance(group_key, tuple) else group_key
            row = {"Model": model_name}
        for column in metric_columns:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci_half_width = z_value * std_value / np.sqrt(max(1, len(values)))
            row[column] = mean_value
            row[f"{column}_std"] = std_value
            row[f"{column}_ci95"] = ci_half_width
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def compute_feature_stability(split_feature_rankings: List[List[str]], top_k: int = 10) -> Dict[str, float]:
    if len(split_feature_rankings) < 2:
        return {
            "TopKOverlapMean": np.nan,
            "TopKOverlapStd": np.nan,
            "RankCorrelationMean": np.nan,
            "RankCorrelationStd": np.nan,
            "NSplits": len(split_feature_rankings),
        }

    overlap_scores = []
    correlation_scores = []

    for left_idx in range(len(split_feature_rankings)):
        for right_idx in range(left_idx + 1, len(split_feature_rankings)):
            left = split_feature_rankings[left_idx]
            right = split_feature_rankings[right_idx]
            left_top = left[:top_k]
            right_top = right[:top_k]
            union = set(left_top) | set(right_top)
            overlap = len(set(left_top) & set(right_top)) / max(1, len(union))
            overlap_scores.append(overlap)

            common = [feature for feature in left_top if feature in right_top]
            if len(common) >= 2:
                left_rank = [left.index(feature) for feature in common]
                right_rank = [right.index(feature) for feature in common]
                corr = spearmanr(left_rank, right_rank).correlation
                if np.isfinite(corr):
                    correlation_scores.append(float(corr))

    return {
        "TopKOverlapMean": float(np.mean(overlap_scores)) if overlap_scores else np.nan,
        "TopKOverlapStd": float(np.std(overlap_scores, ddof=1)) if len(overlap_scores) > 1 else 0.0,
        "RankCorrelationMean": float(np.mean(correlation_scores)) if correlation_scores else np.nan,
        "RankCorrelationStd": float(np.std(correlation_scores, ddof=1)) if len(correlation_scores) > 1 else 0.0,
        "NSplits": len(split_feature_rankings),
    }


@dataclass
class SplitResult:
    dataset: str
    model: str
    repeat: int
    metrics: Dict[str, float]
