from __future__ import annotations

import itertools
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from risk_models.configs import ModelConfig, SEED

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
            LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced"),
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
        self.apply_scaling = model_config.predictor_type == "logreg"
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.fill_values_: Dict[str, Any] = {}
        self.explainer_ = None
        self.fit_time_: Optional[float] = None
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
        X_train = self._prepare_for_fit(X, y)

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
        params.setdefault("max_iter", 2000)
        params.setdefault("class_weight", "balanced")
        return LogisticRegression(**params)
    if config.predictor_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    if config.predictor_type == "xgb":
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params)
    if config.predictor_type == "tabpfn":
        params.setdefault("random_state", SEED)
        return TabPFNAdapter(**params)
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
