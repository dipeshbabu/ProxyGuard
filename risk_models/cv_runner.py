from pathlib import Path
import tracemalloc
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from risk_models.configs import DatasetConfig, ExperimentConfig, ModelConfig, get_ablation_model_configs
from risk_models.dataset import load_dataset
from risk_models.eval import (
    DEFAULT_AGGREGATE_METRICS,
    TemperatureScaler,
    aggregate_metrics,
    best_cost_threshold_from_val,
    best_f1_threshold_from_val,
    compute_feature_stability,
    evaluate_predictions,
    evaluate_subgroups,
)
from risk_models.model import CompactCreditPipeline, build_model
from risk_models.reporting import plot_reliability_diagram, save_run_artifacts


def split_train_val_test(X, y, seed: int, test_size: float, val_size: float):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    inner_val_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=inner_val_size,
        stratify=y_train_full,
        random_state=seed + 1,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_optional_calibrator(p_val, y_val, exp_cfg: ExperimentConfig, enabled: bool):
    if not enabled or exp_cfg.calibration_method == "none":
        return None
    if exp_cfg.calibration_method == "temperature":
        return TemperatureScaler().fit(p_val, y_val)
    raise ValueError(f"Unknown calibration method: {exp_cfg.calibration_method}")


def apply_optional_calibrator(p, calibrator):
    if calibrator is None:
        return p
    return calibrator.transform(p)


def _clone_or_build_model(model_or_config: Any):
    if isinstance(model_or_config, ModelConfig):
        model = build_model(model_or_config)
        setattr(model, "calibration_enabled", model_or_config.use_calibration)
        return model
    if isinstance(model_or_config, CompactCreditPipeline):
        model = build_model(model_or_config.model_config)
        setattr(model, "calibration_enabled", model_or_config.model_config.use_calibration)
        return model

    try:
        cloned = clone(model_or_config)
    except Exception as exc:
        raise TypeError(
            f"Unsupported model object for repeated benchmarking: {type(model_or_config).__name__}. "
            "Pass a ModelConfig, a sklearn-cloneable estimator, or a CompactCreditPipeline."
        ) from exc
    setattr(cloned, "calibration_enabled", getattr(cloned, "calibration_enabled", True))
    return cloned


def run_single_split(model_name: str, model, X, y, exp_cfg: ExperimentConfig, split_seed: int):
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X,
        y,
        seed=split_seed,
        test_size=exp_cfg.test_size,
        val_size=exp_cfg.val_size,
    )

    fitted_model = _clone_or_build_model(model)
    tracemalloc.start()
    fit_start = time.perf_counter()
    fitted_model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - fit_start
    _, peak_fit_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    preprocess_time_sec = getattr(fitted_model, "preprocess_time_", None)
    model_fit_time_sec = getattr(fitted_model, "predictor_fit_time_", None)

    p_val = fitted_model.predict_proba(X_val)[:, 1]
    calibration_enabled = getattr(fitted_model, "calibration_enabled", True)
    calibration_start = time.perf_counter()
    calibrator = fit_optional_calibrator(
        p_val=p_val,
        y_val=y_val,
        exp_cfg=exp_cfg,
        enabled=calibration_enabled,
    )
    calibration_time_sec = time.perf_counter() - calibration_start
    p_val_calibrated = apply_optional_calibrator(p_val, calibrator)
    threshold, f1_val = best_f1_threshold_from_val(y_val, p_val_calibrated)
    cost_thresholds: dict[float, tuple[float, float]] = {}
    for fn_cost in (2.0, 5.0, 10.0, 20.0):
        cost_thresholds[fn_cost] = best_cost_threshold_from_val(
            y_val,
            p_val_calibrated,
            fn_cost=fn_cost,
            fp_cost=1.0,
        )

    tracemalloc.start()
    inference_start = time.perf_counter()
    p_test_raw = fitted_model.predict_proba(X_test)[:, 1]
    inference_time_sec = time.perf_counter() - inference_start
    _, peak_inference_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p_test = apply_optional_calibrator(p_test_raw, calibrator)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=p_test,
        threshold=threshold,
        train_time=train_time_sec,
        inference_time=inference_time_sec,
        peak_memory_mb=max(peak_fit_bytes, peak_inference_bytes) / (1024.0 * 1024.0),
    )
    for fn_cost, (cost_threshold, _) in cost_thresholds.items():
        cost_metrics = evaluate_predictions(y_true=y_test, y_prob=p_test, threshold=cost_threshold)
        cost_name = f"DecisionCost{int(fn_cost)}x"
        metrics[cost_name] = cost_metrics[cost_name]
        metrics[f"{cost_name}RelApproveAll"] = cost_metrics[f"{cost_name}RelApproveAll"]

    metrics["Model"] = model_name
    metrics["split_seed"] = split_seed
    metrics["val_F1_best"] = f1_val
    for fn_cost, (cost_threshold, cost_val) in cost_thresholds.items():
        metrics[f"val_cost_{int(fn_cost)}x_best"] = cost_val
        metrics[f"cost_threshold_{int(fn_cost)}x"] = cost_threshold
    metrics["threshold"] = threshold
    metrics["calibration_applied"] = int(calibrator is not None)
    metrics["PreprocessTimeSec"] = float(preprocess_time_sec) if preprocess_time_sec is not None else np.nan
    metrics["ModelFitTimeSec"] = float(model_fit_time_sec) if model_fit_time_sec is not None else np.nan
    metrics["CalibrationTimeSec"] = float(calibration_time_sec)

    feature_count = getattr(fitted_model, "get_feature_count", None)
    if callable(feature_count):
        metrics["FeatureCount"] = feature_count()

    feature_ranking = getattr(fitted_model, "get_feature_ranking", None)

    return metrics, {
        "Model": model_name,
        "split_seed": split_seed,
        "threshold": threshold,
        "val_index": X_val.index,
        "y_val": y_val,
        "test_index": X_test.index,
        "X_test": X_test,
        "y_test": y_test,
        "p_val": p_val_calibrated,
        "p_test": p_test,
        "cost_thresholds": {
            float(fn_cost): float(cost_threshold)
            for fn_cost, (cost_threshold, _) in cost_thresholds.items()
        },
        "feature_names": getattr(fitted_model, "feature_names_", None),
        "feature_ranking": feature_ranking() if callable(feature_ranking) else getattr(fitted_model, "feature_names_", None),
    }


def aggregate_split_metrics(df: pd.DataFrame) -> pd.DataFrame:
    aggregate_df = aggregate_metrics(
        df,
        confidence_level=0.95,
        metric_columns=[metric for metric in DEFAULT_AGGREGATE_METRICS if metric != "ECE"],
    )
    split_counts = df.groupby("Model").size().rename("n_splits").reset_index()
    return aggregate_df.merge(split_counts, on="Model", how="left")


def run_repeated_benchmark(models: Dict[str, Any], X, y, exp_cfg: ExperimentConfig):
    split_rows = []
    artifacts = []

    for k in range(exp_cfg.n_repeats):
        split_seed = exp_cfg.seed + k

        for model_name, model in models.items():
            metrics, artifact = run_single_split(
                model_name=model_name,
                model=model,
                X=X,
                y=y,
                exp_cfg=exp_cfg,
                split_seed=split_seed,
            )
            split_rows.append(metrics)
            artifacts.append(artifact)

    split_df = pd.DataFrame(split_rows)
    agg_df = aggregate_split_metrics(split_df)
    return split_df, agg_df, artifacts


def _model_configs_to_registry(model_configs: List[ModelConfig]) -> Dict[str, ModelConfig]:
    return {model_config.name: model_config for model_config in model_configs}


def _build_subgroup_metrics(
    dataset_name: str,
    artifacts: List[Dict[str, Any]],
    subgroup_frame: pd.DataFrame | None,
    exp_cfg: ExperimentConfig,
) -> pd.DataFrame:
    if not exp_cfg.run_subgroups or subgroup_frame is None or subgroup_frame.empty:
        return pd.DataFrame()

    subgroup_rows = []
    for artifact in artifacts:
        subgroup_test = subgroup_frame.loc[artifact["test_index"]]
        for subgroup_name in subgroup_test.columns:
            subgroup_metrics = evaluate_subgroups(
                y_true=artifact["y_test"],
                y_prob=artifact["p_test"],
                subgroup_values=subgroup_test[subgroup_name],
                threshold=float(artifact["threshold"]),
                subgroup_name=subgroup_name,
            )
            subgroup_metrics.insert(0, "split_seed", artifact["split_seed"])
            subgroup_metrics.insert(0, "Model", artifact["Model"])
            subgroup_metrics.insert(0, "Dataset", dataset_name)
            subgroup_rows.append(subgroup_metrics)

    if not subgroup_rows:
        return pd.DataFrame()
    return pd.concat(subgroup_rows, ignore_index=True)


def _build_feature_stability_table(dataset_name: str, artifacts: List[Dict[str, Any]]) -> pd.DataFrame:
    ranking_map: Dict[str, List[List[str]]] = {}
    for artifact in artifacts:
        ranking = artifact.get("feature_ranking") or artifact.get("feature_names") or []
        if ranking:
            ranking_map.setdefault(artifact["Model"], []).append(list(ranking))

    rows = []
    for model_name, rankings in ranking_map.items():
        stability = compute_feature_stability(rankings, top_k=10)
        stability["Dataset"] = dataset_name
        stability["Model"] = model_name
        rows.append(stability)
    return pd.DataFrame(rows)


def _build_audit_records(
    dataset_name: str,
    artifacts: List[Dict[str, Any]],
    subgroup_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for artifact in artifacts:
        test_index = pd.Index(artifact["test_index"])
        frame = pd.DataFrame(
            {
                "Dataset": dataset_name,
                "Model": artifact["Model"],
                "split_seed": artifact["split_seed"],
                "record_id": test_index.astype(str),
                "y_true": np.asarray(artifact["y_test"], dtype=int),
                "probability": np.asarray(artifact["p_test"], dtype=float),
                "f1_threshold": float(artifact["threshold"]),
            }
        )
        for fn_cost, threshold in sorted(artifact.get("cost_thresholds", {}).items()):
            frame[f"cost_threshold_{int(fn_cost)}x"] = float(threshold)
        if subgroup_frame is not None and not subgroup_frame.empty:
            subgroup_test = subgroup_frame.loc[test_index].reset_index(drop=True)
            for subgroup_name in subgroup_test.columns:
                frame[f"subgroup__{subgroup_name}"] = subgroup_test[subgroup_name].to_numpy()
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _save_reliability_outputs(run_dir: Path, artifacts: List[Dict[str, Any]], exp_cfg: ExperimentConfig):
    if not exp_cfg.save_reliability:
        return

    reliability_dir = run_dir / "reliability"
    seen_models = set()
    for artifact in artifacts:
        model_name = artifact["Model"]
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        output_path = reliability_dir / f"{model_name}.png"
        plot_reliability_diagram(
            y_true=artifact["y_test"],
            y_prob=artifact["p_test"],
            output_path=output_path,
            title=f"{model_name} reliability",
        )


def _save_benchmark_outputs(
    dataset_name: str,
    mode: str,
    exp_cfg: ExperimentConfig,
    split_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    feature_stability_df: pd.DataFrame,
    artifacts: List[Dict[str, Any]],
    audit_records_df: pd.DataFrame | None = None,
):
    run_dir = Path(exp_cfg.output_root) / mode / dataset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_artifacts(
        run_dir=run_dir,
        split_metrics=split_df if exp_cfg.save_split_metrics else None,
        aggregate_metrics=agg_df if exp_cfg.save_aggregate_metrics else None,
        subgroup_metrics=subgroup_df,
        feature_stability=feature_stability_df,
        audit_records=audit_records_df if exp_cfg.save_audit_records else None,
    )
    _save_reliability_outputs(run_dir, artifacts, exp_cfg)
    return run_dir


def run_benchmark(dataset_config: DatasetConfig, model_configs: List[ModelConfig], exp_cfg: ExperimentConfig, mode: str = "benchmark"):
    dataset_bundle = load_dataset(dataset_config)
    X = dataset_bundle["X"]
    y = dataset_bundle["y"]
    subgroup_frame = dataset_bundle["metadata"].get("subgroup_frame")
    model_registry = _model_configs_to_registry(model_configs)
    split_df, agg_df, artifacts = run_repeated_benchmark(model_registry, X, y, exp_cfg)
    subgroup_df = _build_subgroup_metrics(dataset_config.name, artifacts, subgroup_frame, exp_cfg)
    feature_stability_df = _build_feature_stability_table(dataset_config.name, artifacts)
    audit_records_df = _build_audit_records(dataset_config.name, artifacts, subgroup_frame)
    run_dir = _save_benchmark_outputs(
        dataset_config.name,
        mode,
        exp_cfg,
        split_df,
        agg_df,
        subgroup_df,
        feature_stability_df,
        artifacts,
        audit_records_df,
    )
    return {
        "split_metrics": split_df,
        "aggregate_metrics": agg_df,
        "subgroup_metrics": subgroup_df,
        "feature_stability": feature_stability_df,
        "audit_records": audit_records_df,
        "artifacts": artifacts,
        "metadata": dataset_bundle["metadata"],
        "output_dir": str(run_dir),
    }


def run_ablation_suite(dataset_config: DatasetConfig, exp_cfg: ExperimentConfig):
    return run_benchmark(dataset_config, get_ablation_model_configs(), exp_cfg, mode="ablations")
