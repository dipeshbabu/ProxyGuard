from __future__ import annotations

import itertools
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from risk_models.configs import ModelConfig, SEED

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


class NoOpSelector:
    def __init__(self):
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.selected_features_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self) -> List[str]:
        return list(self.selected_features_)


class MutualInfoTopKSelector:
    def __init__(self, n_features: int = 20, random_state: int = SEED):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        k = min(self.n_features, X.shape[1])
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        pairs = sorted(zip(X.columns, scores), key=lambda item: item[1], reverse=True)
        self.selected_features_ = [name for name, _ in pairs[:k]]
        self.feature_scores_ = {name: float(score) for name, score in pairs}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self) -> List[str]:
        return list(self.selected_features_)


class StabilityAwareFeatureSelector:
    def __init__(
        self,
        n_features: int = 20,
        n_bootstrap: int = 40,
        stability_threshold: float = 0.6,
        required_features: Optional[Iterable[str]] = None,
        random_state: int = SEED,
    ):
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.required_features = list(required_features or [])
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_stability_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rng = np.random.default_rng(self.random_state)
        required = [column for column in self.required_features if column in X.columns]
        optional = [column for column in X.columns if column not in required]
        k_optional = max(0, min(self.n_features - len(required), len(optional)))

        if k_optional <= 0 or not optional:
            self.selected_features_ = required or list(X.columns[: self.n_features])
            self.feature_stability_scores_ = {name: 1.0 for name in self.selected_features_}
            return self

        counts = np.zeros(len(optional), dtype=float)
        avg_mi = np.zeros(len(optional), dtype=float)
        sample_size = max(1, int(0.8 * len(y)))

        for _ in range(self.n_bootstrap):
            sample_idx = rng.choice(len(y), sample_size, replace=True)
            X_boot = X.iloc[sample_idx]
            y_boot = y.iloc[sample_idx]
            mi = mutual_info_classif(X_boot[optional], y_boot, random_state=int(rng.integers(0, 1_000_000)))
            avg_mi += mi
            top_idx = np.argsort(-mi)[:k_optional]
            counts[top_idx] += 1

        stability = counts / float(self.n_bootstrap)
        avg_mi = avg_mi / float(self.n_bootstrap)
        stable_mask = stability >= self.stability_threshold

        if stable_mask.sum() < k_optional:
            combined = stability * avg_mi
            selected_idx = np.argsort(-combined)[:k_optional]
        else:
            stable_idx = np.where(stable_mask)[0]
            selected_idx = stable_idx[np.argsort(-avg_mi[stable_idx])[:k_optional]]

        selected_optional = [optional[index] for index in selected_idx]
        self.selected_features_ = (required + selected_optional)[: self.n_features]
        self.feature_stability_scores_ = {optional[index]: float(stability[index]) for index in range(len(optional))}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self) -> List[str]:
        return list(self.selected_features_)


class CompositionalFeatureEngineer:
    def __init__(self, max_features: int = 8, min_mi_improvement: float = 0.01, max_base_features: int = 6):
        self.max_features = max_features
        self.min_mi_improvement = min_mi_improvement
        self.max_base_features = max_base_features
        self.generated_feature_names_: List[str] = []

    @staticmethod
    def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
        return left / (right + 1e-8)

    def _candidate_features(self, X: pd.DataFrame) -> Dict[str, pd.Series]:
        numeric_columns = [
            column
            for column in X.select_dtypes(include=[np.number]).columns.tolist()
            if X[column].nunique(dropna=True) >= 8 and X[column].std(ddof=0) > 1e-8
        ]
        candidates: Dict[str, pd.Series] = {}

        for left, right in itertools.combinations(numeric_columns[: self.max_base_features], 2):
            candidates[f"{left}_times_{right}"] = (X[left] * X[right]).replace([np.inf, -np.inf], np.nan)
            candidates[f"{left}_over_{right}"] = self._safe_divide(X[left], X[right]).replace([np.inf, -np.inf], np.nan)
            candidates[f"{left}_minus_{right}"] = (X[left] - X[right]).abs()
        return candidates

    def fit(self, X: pd.DataFrame, y: pd.Series):
        numeric_columns = [
            column
            for column in X.select_dtypes(include=[np.number]).columns.tolist()
            if X[column].nunique(dropna=True) >= 8 and X[column].std(ddof=0) > 1e-8
        ]
        if not numeric_columns:
            self.generated_feature_names_ = []
            return self

        base_scores = mutual_info_classif(X[numeric_columns], y, random_state=SEED)
        ranked_numeric = [name for name, _ in sorted(zip(numeric_columns, base_scores), key=lambda item: item[1], reverse=True)]
        reduced_X = X.loc[:, ranked_numeric[: self.max_base_features]].copy()
        candidates = self._candidate_features(reduced_X)
        if not candidates:
            self.generated_feature_names_ = []
            return self

        baseline = float(np.mean(base_scores)) if len(base_scores) else 0.0
        scored: List[tuple[str, float]] = []
        for name, series in candidates.items():
            cleaned = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())
            score = mutual_info_classif(cleaned.to_frame(name), y, random_state=SEED)[0]
            if score > baseline + self.min_mi_improvement:
                scored.append((name, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        self.generated_feature_names_ = [name for name, _ in scored[: self.max_features]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        candidates = self._candidate_features(X)
        for name in self.generated_feature_names_:
            if name in candidates:
                transformed[name] = candidates[name]
        return transformed


class KMeansSegmenter:
    def __init__(self, n_clusters: int = 4, random_state: int = SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_: Optional[KMeans] = None
        self.scaler_ = StandardScaler()
        self.numeric_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if not self.numeric_columns_:
            return self
        X_num = self.scaler_.fit_transform(X[self.numeric_columns_])
        self.model_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.model_.fit(X_num)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or not self.numeric_columns_:
            return X
        X_num = self.scaler_.transform(X[self.numeric_columns_])
        enriched = X.copy()
        enriched["segment_id"] = self.model_.predict(X_num).astype(float)
        return enriched


class BayesianUncertaintyEstimator:
    def __init__(self, n_models: int = 10, subsample_ratio: float = 0.8, random_state: int = SEED):
        self.n_models = n_models
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.models_: List[XGBClassifier] = []
        self.model_weights_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rng = np.random.default_rng(self.random_state)
        weights = []
        self.models_ = []
        for model_idx in range(self.n_models):
            sample_size = max(1, int(self.subsample_ratio * len(y)))
            sample_idx = rng.choice(len(y), size=sample_size, replace=True)
            oob_idx = np.setdiff1d(np.arange(len(y)), sample_idx)
            X_train = X.iloc[sample_idx]
            y_train = y.iloc[sample_idx]
            model = XGBClassifier(
                n_estimators=75,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state + model_idx,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
            self.models_.append(model)
            if len(oob_idx) > 0:
                X_val = X.iloc[oob_idx]
                y_val = y.iloc[oob_idx]
                score = model.predict_proba(X_val)[:, 1]
                weights.append(float(np.clip(score.mean(), 1e-6, 1.0)))
            else:
                weights.append(1.0)

        weight_array = np.asarray(weights, dtype=float)
        self.model_weights_ = weight_array / weight_array.sum()
        return self

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.models_ or self.model_weights_ is None:
            raise ValueError("Uncertainty estimator is not fitted.")
        all_predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models_])
        mean_prediction = np.average(all_predictions, axis=0, weights=self.model_weights_)
        epistemic = np.sqrt(np.average((all_predictions - mean_prediction) ** 2, axis=0, weights=self.model_weights_))
        return {
            "mean": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "predictive_std": all_predictions.std(axis=0),
            "ci_lower": np.percentile(all_predictions, 2.5, axis=0),
            "ci_upper": np.percentile(all_predictions, 97.5, axis=0),
        }


class DataEfficientEnsemble:
    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.base_models_: List[Any] = []
        self.final_model_: Optional[Any] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_unlabeled: Optional[pd.DataFrame] = None):
        self.base_models_ = [
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
            ),
            RandomForestClassifier(n_estimators=300, random_state=self.random_state, class_weight="balanced"),
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    random_state=self.random_state,
                    class_weight="balanced",
                ),
            ),
        ]
        for model in self.base_models_:
            model.fit(X, y)

        if X_unlabeled is not None and len(X_unlabeled) > 0:
            probs = np.mean([model.predict_proba(X_unlabeled)[:, 1] for model in self.base_models_], axis=0)
            confident_mask = np.abs(probs - 0.5) >= 0.35
            if confident_mask.any():
                pseudo_y = (probs[confident_mask] >= 0.5).astype(int)
                X_aug = pd.concat([X, X_unlabeled.loc[confident_mask]], axis=0)
                y_aug = np.concatenate([y, pseudo_y])
            else:
                X_aug, y_aug = X, y
        else:
            X_aug, y_aug = X, y

        self.final_model_ = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.final_model_.fit(X_aug, y_aug)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_model_.predict_proba(X)


class ReliabilityConstrainedEnsemble(BaseEstimator, ClassifierMixin):
    """Learns a small calibrated mixture for risk decisions, not only ranking.

    The mixture weights are fitted on an inner validation split with a proper
    scoring loss plus reliability and decision-cost penalties. Base learners are
    then refit on the full training split while the learned mixture/calibration
    parameters are kept fixed.
    """

    def __init__(
        self,
        base_model_names: tuple[str, ...] = ("logreg", "xgb", "rf"),
        validation_size: float = 0.25,
        brier_weight: float = 1.0,
        ece_weight: float = 0.5,
        cost_weight: float = 0.05,
        balance_weight: float = 0.25,
        fn_cost: float = 5.0,
        max_train_samples: int = 50000,
        max_iter: int = 300,
        random_state: int = SEED,
    ):
        self.base_model_names = base_model_names
        self.validation_size = validation_size
        self.brier_weight = brier_weight
        self.ece_weight = ece_weight
        self.cost_weight = cost_weight
        self.balance_weight = balance_weight
        self.fn_cost = fn_cost
        self.max_train_samples = max_train_samples
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _clip_prob(probabilities: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = ReliabilityConstrainedEnsemble._clip_prob(probabilities)
        return np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    @staticmethod
    def _softmax(theta: np.ndarray) -> np.ndarray:
        shifted = theta - np.max(theta)
        weights = np.exp(shifted)
        return weights / weights.sum()

    @staticmethod
    def _temperature_from_log(log_temperature: float) -> float:
        return float(np.exp(np.clip(log_temperature, -5.0, 5.0)))

    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.digitize(y_prob, bins) - 1
        score = 0.0
        for bin_id in range(n_bins):
            mask = bin_idx == bin_id
            if mask.any():
                score += (mask.sum() / len(y_true)) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
        return float(score)

    def _build_base_model(self, name: str):
        if name == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=self.random_state),
            )
        if name == "xgb":
            return XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=80,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state,
                max_samples=0.8,
                n_jobs=1,
            )
        raise ValueError(f"Unknown reliability ensemble base model: {name}")

    def _base_prob_matrix(self, models: list[Any], X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self._clip_prob(model.predict_proba(X)[:, 1]) for model in models])

    def _mixture_prob(self, base_probabilities: np.ndarray) -> np.ndarray:
        logits = self._logit(base_probabilities)
        combined = logits @ self.mixture_weights_
        return self._sigmoid((combined + self.intercept_) / self.temperature_)

    def _fit_mixture(self, base_probabilities: np.ndarray, y_val: np.ndarray) -> None:
        n_models = base_probabilities.shape[1]
        logits = self._logit(base_probabilities)
        decision_threshold = 1.0 / (1.0 + float(self.fn_cost))

        def objective(params: np.ndarray) -> float:
            weights = self._softmax(params[:n_models])
            temperature = self._temperature_from_log(float(params[n_models]))
            intercept = float(params[n_models + 1])
            probabilities = self._sigmoid(((logits @ weights) + intercept) / temperature)
            predictions = (probabilities >= decision_threshold).astype(int)
            false_negatives = ((y_val == 1) & (predictions == 0)).sum()
            false_positives = ((y_val == 0) & (predictions == 1)).sum()
            decision_cost = (self.fn_cost * false_negatives + false_positives) / max(1, len(y_val))
            balance_gap = float((probabilities.mean() - y_val.mean()) ** 2)
            return float(
                log_loss(y_val, probabilities)
                + self.brier_weight * brier_score_loss(y_val, probabilities)
                + self.ece_weight * self._ece(y_val, probabilities)
                + self.cost_weight * decision_cost
                + self.balance_weight * balance_gap
            )

        init = np.zeros(n_models + 2, dtype=float)
        result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": self.max_iter})
        params = result.x if result.success else init
        self.mixture_weights_ = self._softmax(params[:n_models])
        self.temperature_ = self._temperature_from_log(float(params[n_models]))
        self.intercept_ = float(params[n_models + 1])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_frame = pd.DataFrame(X).copy()
        y_array = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.base_model_names_ = tuple(self.base_model_names)

        if len(X_frame) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_frame), size=self.max_train_samples, replace=False)
            X_frame = X_frame.iloc[sample_idx]
            y_array = y_array[sample_idx]

        can_split = len(y_array) >= 20 and len(np.unique(y_array)) == 2 and min(np.bincount(y_array)) >= 4
        if can_split:
            X_fit, X_mix, y_fit, y_mix = train_test_split(
                X_frame,
                y_array,
                test_size=self.validation_size,
                stratify=y_array,
                random_state=self.random_state,
            )
        else:
            X_fit, X_mix, y_fit, y_mix = X_frame, X_frame, y_array, y_array

        inner_models = [self._build_base_model(name) for name in self.base_model_names_]
        for model in inner_models:
            model.fit(X_fit, y_fit)
        base_probabilities = self._base_prob_matrix(inner_models, X_mix)
        self._fit_mixture(base_probabilities, y_mix)

        self.base_models_ = [self._build_base_model(name) for name in self.base_model_names_]
        for model in self.base_models_:
            model.fit(X_frame, y_array)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self._mixture_prob(self._base_prob_matrix(self.base_models_, pd.DataFrame(X)))
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ReliabilityConstrainedStacker(BaseEstimator, ClassifierMixin):
    """Cross-fitted heterogeneous stacker with reliability-aware calibration."""

    def __init__(
        self,
        base_model_names: tuple[str, ...] = ("logreg", "xgb", "lightgbm", "catboost", "rf"),
        n_folds: int = 3,
        brier_weight: float = 1.0,
        ece_weight: float = 0.5,
        cost_weight: float = 0.05,
        balance_weight: float = 0.25,
        fn_cost: float = 5.0,
        max_train_samples: int = 25000,
        max_iter: int = 300,
        random_state: int = SEED,
    ):
        self.base_model_names = base_model_names
        self.n_folds = n_folds
        self.brier_weight = brier_weight
        self.ece_weight = ece_weight
        self.cost_weight = cost_weight
        self.balance_weight = balance_weight
        self.fn_cost = fn_cost
        self.max_train_samples = max_train_samples
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _clip_prob(probabilities: np.ndarray) -> np.ndarray:
        return ReliabilityConstrainedEnsemble._clip_prob(probabilities)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        return ReliabilityConstrainedEnsemble._logit(probabilities)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return ReliabilityConstrainedEnsemble._sigmoid(logits)

    @staticmethod
    def _temperature_from_log(log_temperature: float) -> float:
        return ReliabilityConstrainedEnsemble._temperature_from_log(log_temperature)

    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        return ReliabilityConstrainedEnsemble._ece(y_true, y_prob, n_bins=n_bins)

    def _available_base_model_names(self) -> tuple[str, ...]:
        names = []
        for name in self.base_model_names:
            if name == "lightgbm" and LGBMClassifier is None:
                continue
            if name == "catboost" and CatBoostClassifier is None:
                continue
            names.append(name)
        if not names:
            raise ValueError("RC-Stack has no available base learners.")
        return tuple(names)

    def _build_base_model(self, name: str, seed_offset: int = 0):
        seed = int(self.random_state + seed_offset)
        if name == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            )
        if name == "xgb":
            return XGBClassifier(
                n_estimators=140,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=1,
            )
        if name == "lightgbm":
            if LGBMClassifier is None:
                raise ImportError("LightGBM is not installed.")
            return LGBMClassifier(
                n_estimators=220,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                objective="binary",
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            )
        if name == "catboost":
            if CatBoostClassifier is None:
                raise ImportError("CatBoost is not installed.")
            return CatBoostClassifier(
                iterations=220,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="Logloss",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            )
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=140,
                max_depth=9,
                class_weight="balanced",
                random_state=seed,
                max_samples=0.85,
                n_jobs=1,
            )
        raise ValueError(f"Unknown RC-Stack base model: {name}")

    def _fit_model(self, model: Any, X: pd.DataFrame, y: np.ndarray) -> Any:
        if hasattr(model, "set_params"):
            params = model.get_params()
            if "scale_pos_weight" in params and params.get("scale_pos_weight", 1.0) == 1.0:
                positives = max(1, int((y == 1).sum()))
                negatives = max(1, int((y == 0).sum()))
                model.set_params(scale_pos_weight=max(1.0, negatives / positives))
        model.fit(X, y)
        return model

    def _stack_features(self, base_probabilities: np.ndarray) -> np.ndarray:
        base_probabilities = self._clip_prob(base_probabilities)
        mean_prob = base_probabilities.mean(axis=1, keepdims=True)
        std_prob = base_probabilities.std(axis=1, keepdims=True)
        spread = (base_probabilities.max(axis=1) - base_probabilities.min(axis=1)).reshape(-1, 1)
        entropy = -(mean_prob * np.log(mean_prob) + (1.0 - mean_prob) * np.log(1.0 - mean_prob))
        return np.hstack([base_probabilities, mean_prob, std_prob, spread, entropy])

    def _base_prob_matrix(self, models: list[Any], X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self._clip_prob(model.predict_proba(X)[:, 1]) for model in models])

    def _build_meta_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=5000, random_state=self.random_state)

    def _fit_calibrator(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        logits = self._logit(probabilities)
        decision_threshold = 1.0 / (1.0 + float(self.fn_cost))

        def objective(params: np.ndarray) -> float:
            temperature = self._temperature_from_log(float(params[0]))
            intercept = float(params[1])
            calibrated = self._sigmoid((logits + intercept) / temperature)
            predictions = (calibrated >= decision_threshold).astype(int)
            false_negatives = ((y_true == 1) & (predictions == 0)).sum()
            false_positives = ((y_true == 0) & (predictions == 1)).sum()
            decision_cost = (self.fn_cost * false_negatives + false_positives) / max(1, len(y_true))
            balance_gap = float((calibrated.mean() - y_true.mean()) ** 2)
            return float(
                log_loss(y_true, calibrated)
                + self.brier_weight * brier_score_loss(y_true, calibrated)
                + self.ece_weight * self._ece(y_true, calibrated)
                + self.cost_weight * decision_cost
                + self.balance_weight * balance_gap
            )

        init = np.zeros(2, dtype=float)
        result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": self.max_iter})
        params = result.x if result.success else init
        self.temperature_ = self._temperature_from_log(float(params[0]))
        self.intercept_ = float(params[1])

    def _calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        logits = self._logit(probabilities)
        return self._sigmoid((logits + self.intercept_) / self.temperature_)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_frame = pd.DataFrame(X).copy()
        y_array = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.base_model_names_ = self._available_base_model_names()

        if len(X_frame) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_frame), size=self.max_train_samples, replace=False)
            X_frame = X_frame.iloc[sample_idx].reset_index(drop=True)
            y_array = y_array[sample_idx]
        else:
            X_frame = X_frame.reset_index(drop=True)

        class_counts = np.bincount(y_array, minlength=2)
        n_splits = min(int(self.n_folds), int(class_counts.min()))
        can_cross_fit = len(y_array) >= 20 and len(np.unique(y_array)) == 2 and n_splits >= 2

        if can_cross_fit:
            oof_probabilities = np.zeros((len(y_array), len(self.base_model_names_)), dtype=float)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_frame, y_array)):
                X_train, y_train = X_frame.iloc[train_idx], y_array[train_idx]
                X_val = X_frame.iloc[val_idx]
                for model_idx, name in enumerate(self.base_model_names_):
                    model = self._build_base_model(name, seed_offset=100 * fold_idx + model_idx)
                    self._fit_model(model, X_train, y_train)
                    oof_probabilities[val_idx, model_idx] = self._clip_prob(model.predict_proba(X_val)[:, 1])
        else:
            fitted_models = []
            for model_idx, name in enumerate(self.base_model_names_):
                model = self._build_base_model(name, seed_offset=model_idx)
                fitted_models.append(self._fit_model(model, X_frame, y_array))
            oof_probabilities = self._base_prob_matrix(fitted_models, X_frame)

        stack_features = self._stack_features(oof_probabilities)
        self.meta_model_ = self._build_meta_model()
        self.meta_model_.fit(stack_features, y_array)
        meta_probabilities = self._clip_prob(self.meta_model_.predict_proba(stack_features)[:, 1])
        self._prepare_calibration_context(X_frame, y_array, meta_probabilities)
        self._fit_calibrator(meta_probabilities, y_array)

        self.base_models_ = []
        for model_idx, name in enumerate(self.base_model_names_):
            model = self._build_base_model(name, seed_offset=10_000 + model_idx)
            self.base_models_.append(self._fit_model(model, X_frame, y_array))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_frame = pd.DataFrame(X).copy()
        base_probabilities = self._base_prob_matrix(self.base_models_, X_frame)
        stack_features = self._stack_features(base_probabilities)
        meta_probabilities = self._clip_prob(self.meta_model_.predict_proba(stack_features)[:, 1])
        probabilities = self._calibrate(meta_probabilities)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _prepare_calibration_context(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        return None


class RegionallyRobustRCStacker(ReliabilityConstrainedStacker):
    """RC-Stack with worst-region calibration penalties over learned regions.

    Region families are built only on the outer-training fold and are used to
    fit the final temperature/intercept calibrator. They do not use the test
    fold and do not change the prediction-time feature path.
    """

    def __init__(
        self,
        base_model_names: tuple[str, ...] = ("logreg", "xgb", "lightgbm", "catboost", "rf"),
        n_folds: int = 3,
        n_reliability_regions: int = 4,
        region_strategy: str = "hybrid",
        region_ece_weight: float = 0.4,
        region_brier_weight: float = 0.2,
        min_region_size: int = 20,
        brier_weight: float = 1.0,
        ece_weight: float = 0.5,
        cost_weight: float = 0.05,
        balance_weight: float = 0.25,
        fn_cost: float = 5.0,
        max_train_samples: int = 25000,
        max_iter: int = 300,
        random_state: int = SEED,
    ):
        super().__init__(
            base_model_names=base_model_names,
            n_folds=n_folds,
            brier_weight=brier_weight,
            ece_weight=ece_weight,
            cost_weight=cost_weight,
            balance_weight=balance_weight,
            fn_cost=fn_cost,
            max_train_samples=max_train_samples,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.n_reliability_regions = n_reliability_regions
        self.region_strategy = region_strategy
        self.region_ece_weight = region_ece_weight
        self.region_brier_weight = region_brier_weight
        self.min_region_size = min_region_size

    def _quantile_regions(self, values: np.ndarray, n_regions: int) -> Optional[np.ndarray]:
        values = np.asarray(values, dtype=float)
        if len(values) < max(2 * self.min_region_size, n_regions * self.min_region_size):
            return None
        quantiles = np.linspace(0.0, 1.0, n_regions + 1)
        edges = np.unique(np.quantile(values, quantiles))
        if len(edges) <= 2:
            return None
        edges[0] = -np.inf
        edges[-1] = np.inf
        return np.digitize(values, edges[1:-1], right=False).astype(int)

    def _kmeans_regions(self, X: pd.DataFrame, y: np.ndarray) -> Optional[np.ndarray]:
        if len(y) < max(2 * self.min_region_size, self.n_reliability_regions * self.min_region_size):
            return None
        n_regions = min(
            int(self.n_reliability_regions),
            max(2, len(y) // max(1, self.min_region_size)),
        )
        numeric = pd.DataFrame(X).select_dtypes(include=[np.number])
        if numeric.empty:
            return None
        self.reliability_cluster_scaler_ = StandardScaler()
        X_scaled = self.reliability_cluster_scaler_.fit_transform(numeric)
        self.reliability_clusterer_ = KMeans(n_clusters=n_regions, random_state=self.random_state, n_init=10)
        return self.reliability_clusterer_.fit_predict(X_scaled).astype(int)

    def _random_regions(self, y: np.ndarray) -> Optional[np.ndarray]:
        if len(y) < max(2 * self.min_region_size, self.n_reliability_regions * self.min_region_size):
            return None
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, int(self.n_reliability_regions), size=len(y))

    @staticmethod
    def _valid_region_family(regions: Optional[np.ndarray], y: np.ndarray, min_region_size: int) -> Optional[np.ndarray]:
        if regions is None or len(regions) != len(y):
            return None
        regions = np.asarray(regions, dtype=int)
        valid_ids = []
        for region_id in np.unique(regions):
            mask = regions == region_id
            if mask.sum() >= min_region_size and len(np.unique(y[mask])) >= 2:
                valid_ids.append(region_id)
        if len(valid_ids) < 2:
            return None
        return regions

    def _prepare_calibration_context(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        self.calibration_region_families_: list[np.ndarray] = []
        self.reliability_clusterer_ = None
        self.reliability_cluster_scaler_ = None
        y = np.asarray(y).astype(int)
        probabilities = self._clip_prob(np.asarray(probabilities)) if probabilities is not None else None
        strategy = str(self.region_strategy).lower()

        candidates: list[Optional[np.ndarray]] = []
        if strategy in {"kmeans", "hybrid", "all"}:
            candidates.append(self._kmeans_regions(X, y))
        if probabilities is not None and strategy in {"risk", "hybrid", "all"}:
            candidates.append(self._quantile_regions(probabilities, int(self.n_reliability_regions)))
        if probabilities is not None and strategy in {"error", "all"}:
            residual = np.abs(y.astype(float) - probabilities)
            candidates.append(self._quantile_regions(residual, int(self.n_reliability_regions)))
        if strategy == "random":
            candidates.append(self._random_regions(y))
        if not candidates and strategy not in {"none", ""}:
            raise ValueError(f"Unknown region_strategy: {self.region_strategy}")

        for regions in candidates:
            valid = self._valid_region_family(regions, y, int(self.min_region_size))
            if valid is not None:
                self.calibration_region_families_.append(valid)
        self.n_calibration_region_families_ = len(self.calibration_region_families_)

    def _worst_region_penalty(self, y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
        region_families = getattr(self, "calibration_region_families_", None)
        if not region_families:
            return 0.0, 0.0
        region_ece: list[float] = []
        region_brier: list[float] = []
        for regions in region_families:
            if len(regions) != len(y_true):
                continue
            for region_id in np.unique(regions):
                mask = regions == region_id
                if mask.sum() < self.min_region_size or len(np.unique(y_true[mask])) < 2:
                    continue
                region_ece.append(self._ece(y_true[mask], probabilities[mask]))
                region_brier.append(float(brier_score_loss(y_true[mask], probabilities[mask])))
        if not region_ece:
            return 0.0, 0.0
        return float(max(region_ece)), float(max(region_brier))

    def _fit_calibrator(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        logits = self._logit(probabilities)
        decision_threshold = 1.0 / (1.0 + float(self.fn_cost))

        def objective(params: np.ndarray) -> float:
            temperature = self._temperature_from_log(float(params[0]))
            intercept = float(params[1])
            calibrated = self._sigmoid((logits + intercept) / temperature)
            predictions = (calibrated >= decision_threshold).astype(int)
            false_negatives = ((y_true == 1) & (predictions == 0)).sum()
            false_positives = ((y_true == 0) & (predictions == 1)).sum()
            decision_cost = (self.fn_cost * false_negatives + false_positives) / max(1, len(y_true))
            balance_gap = float((calibrated.mean() - y_true.mean()) ** 2)
            worst_ece, worst_brier = self._worst_region_penalty(y_true, calibrated)
            return float(
                log_loss(y_true, calibrated)
                + self.brier_weight * brier_score_loss(y_true, calibrated)
                + self.ece_weight * self._ece(y_true, calibrated)
                + self.cost_weight * decision_cost
                + self.balance_weight * balance_gap
                + self.region_ece_weight * worst_ece
                + self.region_brier_weight * worst_brier
            )

        init = np.zeros(2, dtype=float)
        result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": self.max_iter})
        params = result.x if result.success else init
        self.temperature_ = self._temperature_from_log(float(params[0]))
        self.intercept_ = float(params[1])


class DistributionallyRobustRCStacker(RegionallyRobustRCStacker):
    """Backward-compatible k-means RC-Stack robust-calibration variant."""

    def __init__(
        self,
        base_model_names: tuple[str, ...] = ("logreg", "xgb", "lightgbm", "catboost", "rf"),
        n_folds: int = 3,
        n_reliability_clusters: int = 4,
        group_ece_weight: float = 0.4,
        group_brier_weight: float = 0.2,
        min_group_size: int = 20,
        brier_weight: float = 1.0,
        ece_weight: float = 0.5,
        cost_weight: float = 0.05,
        balance_weight: float = 0.25,
        fn_cost: float = 5.0,
        max_train_samples: int = 25000,
        max_iter: int = 300,
        random_state: int = SEED,
    ):
        super().__init__(
            base_model_names=base_model_names,
            n_folds=n_folds,
            n_reliability_regions=n_reliability_clusters,
            region_strategy="kmeans",
            region_ece_weight=group_ece_weight,
            region_brier_weight=group_brier_weight,
            min_region_size=min_group_size,
            brier_weight=brier_weight,
            ece_weight=ece_weight,
            cost_weight=cost_weight,
            balance_weight=balance_weight,
            fn_cost=fn_cost,
            max_train_samples=max_train_samples,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.n_reliability_clusters = n_reliability_clusters
        self.group_ece_weight = group_ece_weight
        self.group_brier_weight = group_brier_weight
        self.min_group_size = min_group_size


class TabPFNAdapter(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        device: str = "auto",
        n_estimators: int = 1,
        max_train_samples: int = 10000,
        model_cache_dir: str = "outputs/tabpfn_cache",
        random_state: int = SEED,
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.max_train_samples = max_train_samples
        self.model_cache_dir = model_cache_dir
        self.random_state = random_state

    @staticmethod
    def _as_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                frame[column] = frame[column].fillna(frame[column].median())
        return frame.astype(float)

    def _build_classifier(self):
        cache_dir = os.path.abspath(self.model_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", cache_dir)
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "The TabPFN baseline requires `tabpfn`. Install dependencies with "
                "`pip install -r requirements.txt` or run without `--include-tabpfn`."
            ) from exc

        try:
            return TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        except TypeError:
            return TabPFNClassifier()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_frame = self._as_numeric_frame(X)
        y_array = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_array)
        self.feature_names_in_ = np.asarray(X_frame.columns)

        if len(X_frame) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_frame), size=self.max_train_samples, replace=False)
            X_fit = X_frame.iloc[sample_idx]
            y_fit = y_array[sample_idx]
        else:
            X_fit = X_frame
            y_fit = y_array

        self.model_ = self._build_classifier()
        self.model_.fit(X_fit.to_numpy(dtype=float), y_fit)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_frame = self._as_numeric_frame(X)
        probabilities = self.model_.predict_proba(X_frame.to_numpy(dtype=float))
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabICLAdapter(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        device: Optional[str] = None,
        n_estimators: int = 1,
        batch_size: int = 4,
        max_train_samples: int = 4096,
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        random_state: int = SEED,
        verbose: bool = False,
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples
        self.checkpoint_version = checkpoint_version
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _as_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                frame[column] = frame[column].fillna(frame[column].median())
        return frame.astype(float)

    def _build_classifier(self):
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:
            raise ImportError("The TabICL baseline requires `tabicl`.") from exc
        return TabICLClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            checkpoint_version=self.checkpoint_version,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_frame = self._as_numeric_frame(X)
        y_array = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_array)
        self.feature_names_in_ = np.asarray(X_frame.columns)

        if len(X_frame) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_frame), size=self.max_train_samples, replace=False)
            X_fit = X_frame.iloc[sample_idx]
            y_fit = y_array[sample_idx]
        else:
            X_fit = X_frame
            y_fit = y_array

        self.model_ = self._build_classifier()
        self.model_.fit(X_fit.to_numpy(dtype=float), y_fit)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_frame = self._as_numeric_frame(X)
        probabilities = self.model_.predict_proba(X_frame.to_numpy(dtype=float))
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CompactCreditPipeline:
    def __init__(
        self,
        model_config: ModelConfig,
        selector: Any,
        feature_engineer: Optional[CompositionalFeatureEngineer],
        segmenter: Optional[KMeansSegmenter],
        predictor: Any,
        uncertainty_estimator: Optional[BayesianUncertaintyEstimator] = None,
    ):
        self.model_config = model_config
        self.selector = selector
        self.feature_engineer = feature_engineer
        self.segmenter = segmenter
        self.predictor = predictor
        self.uncertainty_estimator = uncertainty_estimator
        self.scaler = StandardScaler()
        self.apply_scaling = model_config.predictor_type in {
            "logreg",
            "reliability_ensemble",
            "rc_stack",
            "rc_stack_dr",
            "rrc_stack",
        }
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.fill_values_: Dict[str, Any] = {}
        self.explainer_ = None
        self.fit_time_: Optional[float] = None
        self.preprocess_time_: Optional[float] = None
        self.predictor_fit_time_: Optional[float] = None
        self.feature_importances_: Dict[str, float] = {}

    def _impute_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        self.fill_values_ = {}
        for column in transformed.columns:
            series = transformed[column]
            if is_numeric_dtype(series):
                fill_value = float(series.median()) if series.notna().any() else 0.0
            else:
                fill_value = "Missing"
            transformed[column] = series.fillna(fill_value)
            self.fill_values_[column] = fill_value
        return transformed

    def _impute_apply(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        for column in transformed.columns:
            fill_value = self.fill_values_.get(column, 0.0 if is_numeric_dtype(transformed[column]) else "Missing")
            transformed[column] = transformed[column].fillna(fill_value)
        return transformed

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineer.transform(X) if self.feature_engineer else X.copy()

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.selector.transform(X)

    def _segment(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.segmenter.transform(X) if self.segmenter else X

    def _scale_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        self.numeric_columns_ = transformed.select_dtypes(include=[np.number]).columns.tolist()
        if self.apply_scaling and self.numeric_columns_:
            transformed[self.numeric_columns_] = self.scaler.fit_transform(transformed[self.numeric_columns_])
        return transformed

    def _scale_apply(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        if self.apply_scaling and self.numeric_columns_:
            transformed[self.numeric_columns_] = self.scaler.transform(transformed[self.numeric_columns_])
        return transformed

    def _prepare_for_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        transformed = self._impute_fit(X)
        transformed = self._engineer(transformed)
        self.selector.fit(transformed, y)
        transformed = self._select(transformed)
        if self.segmenter:
            self.segmenter.fit(transformed, y)
            transformed = self._segment(transformed)
        transformed = self._scale_fit(transformed)
        self.feature_names_ = list(transformed.columns)
        return transformed

    def _prepare_for_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._impute_apply(X)
        transformed = self._engineer(transformed)
        transformed = self._select(transformed)
        transformed = self._segment(transformed)
        transformed = self._scale_apply(transformed)
        for feature_name in self.feature_names_:
            if feature_name not in transformed.columns:
                transformed[feature_name] = 0.0
        return transformed.loc[:, self.feature_names_]

    def _collect_feature_importances(self):
        if hasattr(self.predictor, "feature_importances_"):
            values = np.asarray(self.predictor.feature_importances_, dtype=float)
            self.feature_importances_ = {name: float(value) for name, value in zip(self.feature_names_, values)}
        elif hasattr(self.predictor, "coef_"):
            values = np.asarray(self.predictor.coef_).ravel()
            self.feature_importances_ = {name: float(abs(value)) for name, value in zip(self.feature_names_, values)}
        else:
            self.feature_importances_ = {name: 0.0 for name in self.feature_names_}

    def fit(self, X: pd.DataFrame, y: pd.Series, X_unlabeled: Optional[pd.DataFrame] = None):
        start = time.perf_counter()
        preprocess_start = time.perf_counter()
        X_train = self._prepare_for_fit(X, y)
        self.preprocess_time_ = time.perf_counter() - preprocess_start

        predictor_start = time.perf_counter()
        if isinstance(self.predictor, DataEfficientEnsemble):
            X_unlabeled_prepared = self._prepare_for_inference(X_unlabeled) if X_unlabeled is not None else None
            self.predictor.fit(X_train, y, X_unlabeled_prepared)
        else:
            predictor = clone(self.predictor)
            if hasattr(predictor, "set_params"):
                params = predictor.get_params()
                if "scale_pos_weight" in params and params.get("scale_pos_weight", 1.0) == 1.0:
                    positives = max(1, int((y == 1).sum()))
                    negatives = max(1, int((y == 0).sum()))
                    predictor.set_params(scale_pos_weight=max(1.0, negatives / positives))
            predictor.fit(X_train, y)
            self.predictor = predictor
        self.predictor_fit_time_ = time.perf_counter() - predictor_start

        if self.uncertainty_estimator is not None:
            self.uncertainty_estimator.fit(X_train, y)

        if shap is not None and hasattr(self.predictor, "predict_proba") and hasattr(self.predictor, "feature_importances_"):
            try:
                self.explainer_ = shap.TreeExplainer(self.predictor)
            except Exception:
                self.explainer_ = None

        self.fit_time_ = time.perf_counter() - start
        self._collect_feature_importances()
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict(self._prepare_for_inference(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict_proba(self._prepare_for_inference(X))

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.uncertainty_estimator is None:
            raise ValueError("Uncertainty estimation is disabled for this model.")
        return self.uncertainty_estimator.predict_with_uncertainty(self._prepare_for_inference(X))

    def get_feature_count(self) -> int:
        return len(self.feature_names_)

    def get_feature_ranking(self) -> List[str]:
        ranked = sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked]

    def get_shap_values(self, X: pd.DataFrame):
        if self.explainer_ is None:
            raise ValueError("SHAP explainer is not available for this model.")
        X_prepared = self._prepare_for_inference(X)
        try:
            return self.explainer_.shap_values(X_prepared)
        except Exception:
            return self.explainer_(X_prepared).values

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_for_inference(X)


def build_selector(config: ModelConfig):
    if config.selector_type == "none":
        return NoOpSelector()
    if config.selector_type == "mi":
        return MutualInfoTopKSelector(n_features=config.selector_k, random_state=SEED)
    if config.selector_type == "stability":
        selector_params = {"n_features": config.selector_k, "random_state": SEED}
        selector_params.update(getattr(config, "selector_params", {}))
        return StabilityAwareFeatureSelector(**selector_params)
    raise ValueError(f"Unknown selector type: {config.selector_type}")


def build_feature_engineer(config: ModelConfig):
    if not config.use_feature_engineering:
        return None
    return CompositionalFeatureEngineer(max_features=config.feature_engineering_max_features)


def build_segmenter(config: ModelConfig):
    if not config.use_segmenter:
        return None
    if config.segmenter_type != "kmeans":
        raise ValueError(f"Unknown segmenter type: {config.segmenter_type}")
    return KMeansSegmenter(n_clusters=config.segmenter_n_clusters, random_state=SEED)


def build_predictor(config: ModelConfig):
    params = dict(config.estimator_params)
    params.setdefault("random_state", SEED)

    if config.use_pseudo_labels or config.predictor_type == "pseudo_label_ensemble":
        return DataEfficientEnsemble(random_state=params["random_state"])
    if config.predictor_type == "logreg":
        params.setdefault("max_iter", 5000)
        params.setdefault("class_weight", "balanced")
        return LogisticRegression(**params)
    if config.predictor_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    if config.predictor_type == "histgb":
        params.setdefault("random_state", SEED)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("max_iter", 300)
        params.setdefault("l2_regularization", 0.01)
        return HistGradientBoostingClassifier(**params)
    if config.predictor_type == "xgb":
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params)
    if config.predictor_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("The LightGBM baseline requires `lightgbm`.")
        params.setdefault("verbose", -1)
        return LGBMClassifier(**params)
    if config.predictor_type == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("The CatBoost baseline requires `catboost`.")
        random_state = params.pop("random_state", SEED)
        params.setdefault("random_seed", random_state)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostClassifier(**params)
    if config.predictor_type == "reliability_ensemble":
        params.setdefault("random_state", SEED)
        return ReliabilityConstrainedEnsemble(**params)
    if config.predictor_type == "rc_stack":
        params.setdefault("random_state", SEED)
        return ReliabilityConstrainedStacker(**params)
    if config.predictor_type == "rc_stack_dr":
        params.setdefault("random_state", SEED)
        return DistributionallyRobustRCStacker(**params)
    if config.predictor_type == "rrc_stack":
        params.setdefault("random_state", SEED)
        return RegionallyRobustRCStacker(**params)
    if config.predictor_type == "tabpfn":
        params.setdefault("random_state", SEED)
        return TabPFNAdapter(**params)
    if config.predictor_type == "tabicl":
        params.setdefault("random_state", SEED)
        return TabICLAdapter(**params)
    raise ValueError(f"Unknown predictor type: {config.predictor_type}")


def build_pipeline(config: ModelConfig) -> CompactCreditPipeline:
    selector = build_selector(config)
    feature_engineer = build_feature_engineer(config)
    segmenter = build_segmenter(config)
    predictor = build_predictor(config)
    uncertainty_estimator = BayesianUncertaintyEstimator(random_state=SEED) if config.use_uncertainty else None
    return CompactCreditPipeline(
        model_config=config,
        selector=selector,
        feature_engineer=feature_engineer,
        segmenter=segmenter,
        predictor=predictor,
        uncertainty_estimator=uncertainty_estimator,
    )


def build_model(model_config: ModelConfig):
    return build_pipeline(model_config)


def build_model_registry(model_registry: Dict[str, ModelConfig]) -> Dict[str, CompactCreditPipeline]:
    return {model_name: build_pipeline(model_config) for model_name, model_config in model_registry.items()}


def get_models_smalldata() -> Dict[str, CompactCreditPipeline]:
    from risk_models.configs import get_benchmark_model_configs

    return {config.name: build_pipeline(config) for config in get_benchmark_model_configs()}


class SmallDataCreditPipeline(CompactCreditPipeline):
    """
    Backward-compatible wrapper for legacy callers that still construct the old pipeline directly.
    """

    def __init__(self, use_all_techniques: bool = True):
        config = ModelConfig(
            name="legacy_small_data_pipeline",
            selector_type="stability",
            selector_k=20,
            use_feature_engineering=use_all_techniques,
            use_segmenter=use_all_techniques,
            predictor_type="xgb",
            use_calibration=use_all_techniques,
            use_uncertainty=use_all_techniques,
        )
        super().__init__(
            model_config=config,
            selector=build_selector(config),
            feature_engineer=build_feature_engineer(config),
            segmenter=build_segmenter(config),
            predictor=build_predictor(config),
            uncertainty_estimator=BayesianUncertaintyEstimator(random_state=SEED) if config.use_uncertainty else None,
        )
