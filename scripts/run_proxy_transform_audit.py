from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import (
    clone_experiment_config,
    get_dataset_config,
    get_default_experiment_config,
    get_spotlight_model_configs,
)
from risk_models.cv_runner import (
    _build_audit_records,
    _build_feature_stability_table,
    _build_subgroup_metrics,
    _model_configs_to_registry,
    _save_benchmark_outputs,
    run_repeated_benchmark,
)
from risk_models.dataset import load_dataset


DEFAULT_MODELS = [
    "xgb_baseline",
    "compact_xgb",
    "tabpfn_baseline",
    "tabicl_baseline",
    "rc_stack",
    "rrc_stack",
]


CSV_KEYS = {
    "aggregate_metrics.csv": ["Model"],
    "split_metrics.csv": ["Model", "split_seed"],
    "subgroup_metrics.csv": ["Dataset", "Model", "split_seed", "SubgroupName", "SubgroupValue"],
    "feature_stability.csv": ["Dataset", "Model"],
    "audit_records.csv": ["Dataset", "Model", "split_seed", "record_id"],
}


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def binary_columns(X: pd.DataFrame) -> set[str]:
    binary: set[str] = set()
    for column in X.columns:
        values = pd.Series(X[column]).dropna().unique()
        if len(values) <= 2 and set(values).issubset({0, 1, 0.0, 1.0, False, True}):
            binary.add(column)
    return binary


def sensitive_feature_columns(X: pd.DataFrame, sensitive_columns: list[str] | None) -> list[str]:
    if not sensitive_columns:
        return []
    matches: list[str] = []
    for sensitive in sensitive_columns:
        sensitive_norm = str(sensitive).strip()
        if not sensitive_norm:
            continue
        candidate_names = {sensitive_norm}
        if "age" in sensitive_norm.lower():
            candidate_names.add("age")
        for column in X.columns:
            column_name = str(column)
            column_lower = column_name.lower()
            if any(
                column_lower == candidate.lower() or column_lower.startswith(f"{candidate.lower()}_")
                for candidate in candidate_names
            ):
                matches.append(column_name)
    return sorted(set(matches))


def mask_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    transformed = X.copy()
    for column in columns:
        if column not in transformed.columns:
            continue
        if pd.api.types.is_bool_dtype(transformed[column]):
            transformed.loc[:, column] = False
        else:
            transformed.loc[:, column] = 0.0
    return transformed


def _sample_column(
    X: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
    noisy: bool,
) -> pd.Series:
    source = X[column]
    values = source.dropna()
    if values.empty:
        return pd.Series(np.nan, index=X.index, dtype=source.dtype)
    if noisy:
        counts = values.value_counts(dropna=False)
        weights = counts.to_numpy(dtype=float) + rng.laplace(0.0, 1.0, size=len(counts))
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        sampled = rng.choice(counts.index.to_numpy(), size=len(X), replace=True, p=weights)
    else:
        sampled = rng.choice(values.to_numpy(), size=len(X), replace=True)
    return pd.Series(sampled, index=X.index)


def marginal_synthesis(
    X: pd.DataFrame,
    seed: int,
    noisy: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns: dict[str, pd.Series] = {}
    binary = binary_columns(X)
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    continuous = [column for column in numeric_columns if column not in binary]

    for column in X.columns:
        sampled = _sample_column(X, column, rng, noisy=noisy)
        if column in continuous:
            std = float(pd.to_numeric(X[column], errors="coerce").std(ddof=0))
            if np.isfinite(std) and std > 0:
                scale = 0.03 * std if not noisy else 0.10 * std
                sampled = pd.to_numeric(sampled, errors="coerce") + rng.normal(0.0, scale, size=len(sampled))
        columns[column] = sampled
    return pd.DataFrame(columns, index=X.index)


def dp_marginal_synthesis(
    X: pd.DataFrame,
    seed: int,
    epsilon: float = 1.0,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Independent marginal synthesizer with Laplace-noised counts.

    Numeric features use fixed-width bins over the current feature bounds, and
    categorical/binary features use observed categories. In a restricted-data
    release those bounds/categories must be public or estimated privately; here
    the baseline is used as a simple DP-style stress test on public proxy tasks.
    """
    rng = np.random.default_rng(seed)
    epsilon = max(float(epsilon), 1e-9)
    columns = list(X.columns)
    epsilon_per_column = epsilon / max(len(columns), 1)
    binary = binary_columns(X)
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    continuous = [column for column in numeric_columns if column not in binary]
    synthesized: dict[str, pd.Series] = {}

    for column in columns:
        source = X[column]
        values = source.dropna()
        if values.empty:
            synthesized[column] = pd.Series(np.nan, index=X.index, dtype=source.dtype)
            continue

        if column in continuous:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                synthesized[column] = _sample_column(X, column, rng, noisy=True)
                continue
            lower = float(numeric.min())
            upper = float(numeric.max())
            if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
                synthesized[column] = pd.Series(np.full(len(X), lower), index=X.index)
                continue
            bins = np.linspace(lower, upper, n_bins + 1)
            counts, _ = np.histogram(pd.to_numeric(source, errors="coerce").dropna(), bins=bins)
            noisy_counts = counts.astype(float) + rng.laplace(0.0, 1.0 / epsilon_per_column, size=len(counts))
            noisy_counts = np.maximum(noisy_counts, 0.0)
            if noisy_counts.sum() <= 0:
                noisy_counts = np.ones_like(noisy_counts)
            probabilities = noisy_counts / noisy_counts.sum()
            sampled_bins = rng.choice(np.arange(n_bins), size=len(X), replace=True, p=probabilities)
            sampled = rng.uniform(bins[sampled_bins], bins[sampled_bins + 1])
            synthesized[column] = pd.Series(sampled, index=X.index)
            continue

        counts = values.value_counts(dropna=False)
        noisy_counts = counts.to_numpy(dtype=float) + rng.laplace(0.0, 1.0 / epsilon_per_column, size=len(counts))
        noisy_counts = np.maximum(noisy_counts, 0.0)
        if noisy_counts.sum() <= 0:
            noisy_counts = np.ones_like(noisy_counts)
        probabilities = noisy_counts / noisy_counts.sum()
        sampled = rng.choice(counts.index.to_numpy(), size=len(X), replace=True, p=probabilities)
        synthesized[column] = pd.Series(sampled, index=X.index)

    return pd.DataFrame(synthesized, index=X.index)


def transform_features(
    X: pd.DataFrame,
    variant: str,
    seed: int,
    sensitive_columns: list[str] | None = None,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    transformed = X.copy()
    binary = binary_columns(transformed)
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    continuous = [column for column in numeric_columns if column not in binary]

    if variant == "baseline":
        return transformed

    if variant == "numeric_noise_10":
        for column in continuous:
            std = float(transformed[column].std(ddof=0))
            if np.isfinite(std) and std > 0:
                transformed[column] = transformed[column] + rng.normal(0.0, 0.10 * std, size=len(transformed))
        return transformed

    if variant == "numeric_noise_20":
        for column in continuous:
            std = float(transformed[column].std(ddof=0))
            if np.isfinite(std) and std > 0:
                transformed[column] = transformed[column] + rng.normal(0.0, 0.20 * std, size=len(transformed))
        return transformed

    if variant == "laplace_noise_20":
        for column in continuous:
            q_low = float(transformed[column].quantile(0.05))
            q_high = float(transformed[column].quantile(0.95))
            scale = 0.20 * max(q_high - q_low, 0.0)
            if np.isfinite(scale) and scale > 0:
                transformed[column] = transformed[column] + rng.laplace(0.0, scale, size=len(transformed))
        return transformed

    if variant == "coarsen_quartile":
        for column in continuous:
            series = transformed[column]
            if series.nunique(dropna=True) >= 8:
                transformed[column] = pd.qcut(series.rank(method="first"), q=4, labels=False, duplicates="drop").astype(float)
        return transformed

    if variant == "rank_swap_10":
        for column in continuous:
            series = transformed[column]
            if series.nunique(dropna=True) < 8:
                continue
            sorted_index = series.sort_values(kind="mergesort").index.to_numpy()
            window = max(2, int(round(0.10 * len(sorted_index))))
            values = transformed.loc[sorted_index, column].to_numpy(copy=True)
            swapped = values.copy()
            for start in range(0, len(values), window):
                stop = min(start + window, len(values))
                if stop - start > 1:
                    swapped[start:stop] = rng.permutation(swapped[start:stop])
            transformed.loc[sorted_index, column] = swapped
        return transformed

    if variant == "feature_mask_20":
        candidate_columns = list(transformed.columns)
        n_mask = max(1, int(round(0.20 * len(candidate_columns))))
        columns_to_mask = rng.choice(candidate_columns, size=n_mask, replace=False)
        return mask_columns(transformed, list(columns_to_mask))

    if variant == "sensitive_mask":
        columns = sensitive_feature_columns(transformed, sensitive_columns)
        if not columns:
            return transformed
        return mask_columns(transformed, columns)

    if variant in {"synthetic_marginal", "synthetic_marginal_y"}:
        return marginal_synthesis(transformed, seed=seed, noisy=False)

    if variant in {"noisy_synthetic_marginal", "noisy_synthetic_marginal_y"}:
        return marginal_synthesis(transformed, seed=seed, noisy=True)

    if variant in {"dp_marginal_e1", "dp_marginal_e1_y"}:
        return dp_marginal_synthesis(transformed, seed=seed, epsilon=1.0)

    raise ValueError(f"Unknown proxy transform variant: {variant}")


def merge_saved_csvs(run_dir: Path, previous: dict[str, pd.DataFrame]) -> None:
    for filename, key_cols in CSV_KEYS.items():
        path = run_dir / filename
        if not path.exists():
            continue
        current = pd.read_csv(path)
        earlier = previous.get(filename, pd.DataFrame())
        if earlier.empty:
            merged = current
        elif all(column in earlier.columns and column in current.columns for column in key_cols):
            current_keys = set(map(tuple, current[key_cols].to_numpy()))
            keep_earlier = earlier[~earlier[key_cols].apply(tuple, axis=1).isin(current_keys)]
            merged = pd.concat([keep_earlier, current], ignore_index=True, sort=False)
        else:
            merged = current
        if "Model" in merged.columns:
            sort_cols = [column for column in ["Model", "split_seed", "SubgroupName", "SubgroupValue"] if column in merged.columns]
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(path, index=False)


def read_existing_outputs(run_dir: Path) -> dict[str, pd.DataFrame]:
    existing = {}
    for filename in CSV_KEYS:
        path = run_dir / filename
        if path.exists():
            existing[filename] = pd.read_csv(path)
    return existing


def run_one_proxy_model(dataset_name: str, variant: str, model_config, X: pd.DataFrame, y: pd.Series, subgroup_frame, exp_cfg) -> pd.DataFrame:
    model_registry = _model_configs_to_registry([model_config])
    split_df, agg_df, artifacts = run_repeated_benchmark(model_registry, X, y, exp_cfg)
    subgroup_df = _build_subgroup_metrics(dataset_name, artifacts, subgroup_frame, exp_cfg)
    feature_stability_df = _build_feature_stability_table(dataset_name, artifacts)
    audit_records_df = _build_audit_records(dataset_name, artifacts, subgroup_frame)
    run_dir = Path(exp_cfg.output_root) / "proxy_transform" / variant / dataset_name
    previous = read_existing_outputs(run_dir)
    _save_benchmark_outputs(
        dataset_name=dataset_name,
        mode=f"proxy_transform/{variant}",
        exp_cfg=exp_cfg,
        split_df=split_df,
        agg_df=agg_df,
        subgroup_df=subgroup_df,
        feature_stability_df=feature_stability_df,
        artifacts=artifacts,
        audit_records_df=audit_records_df,
    )
    merge_saved_csvs(run_dir, previous)
    return agg_df


def run_proxy_variant(dataset_name: str, variant: str, model_names: set[str], exp_cfg) -> pd.DataFrame:
    dataset_config = get_dataset_config(dataset_name)
    bundle = load_dataset(dataset_config)
    subgroup_frame = bundle["metadata"].get("subgroup_frame")
    sensitive_columns = list(subgroup_frame.columns) if subgroup_frame is not None else []
    y = bundle["y"]
    X = transform_features(bundle["X"], variant=variant, seed=exp_cfg.seed, sensitive_columns=sensitive_columns, y=y)
    model_configs = [
        config
        for config in get_spotlight_model_configs(include_tabpfn=True, include_tabicl=True)
        if config.name in model_names
    ]
    missing = model_names - {config.name for config in model_configs}
    if missing:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(missing))}")

    run_dir = Path(exp_cfg.output_root) / "proxy_transform" / variant / dataset_name
    rows = []
    existing = read_existing_outputs(run_dir).get("aggregate_metrics.csv", pd.DataFrame())
    for model_config in model_configs:
        if (
            not existing.empty
            and model_config.name in set(existing["Model"])
            and (existing.loc[existing["Model"] == model_config.name, "n_splits"] >= exp_cfg.n_repeats).all()
        ):
            print(f"proxy transform complete, skipping {dataset_name} [{variant}] {model_config.name}")
            rows.append(existing[existing["Model"] == model_config.name].copy())
            continue
        print(f"running {dataset_name} [{variant}] {model_config.name}")
        rows.append(run_one_proxy_model(dataset_name, variant, model_config, X, y, subgroup_frame, exp_cfg))
        existing = read_existing_outputs(run_dir).get("aggregate_metrics.csv", pd.DataFrame())
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary.insert(0, "Variant", variant)
    summary.insert(0, "Dataset", dataset_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RUA proxy-transform audits on preprocessed tabular datasets.")
    parser.add_argument("--datasets", default="australian_credit,german_credit")
    parser.add_argument("--variants", default="baseline,numeric_noise_10,coarsen_quartile,feature_mask_20")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/proxy_transform_audit")
    parser.add_argument("--no-subgroups", action="store_true")
    args = parser.parse_args()

    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=args.repeats,
        output_root=args.output_root,
        calibration_method="temperature",
        run_subgroups=not args.no_subgroups,
        save_reliability=False,
        save_shap=False,
        save_audit_records=True,
    )
    model_names = set(parse_csv(args.models))
    rows = []
    for dataset_name in parse_csv(args.datasets):
        for variant in parse_csv(args.variants):
            output_path = Path(args.output_root) / "proxy_transform" / variant / dataset_name / "aggregate_metrics.csv"
            if output_path.exists():
                existing = pd.read_csv(output_path)
                if (
                    not existing.empty
                    and model_names.issubset(set(existing["Model"]))
                    and (existing.loc[existing["Model"].isin(model_names), "n_splits"] >= args.repeats).all()
                ):
                    existing = existing[existing["Model"].isin(model_names)].copy()
                    existing.insert(0, "Variant", variant)
                    existing.insert(0, "Dataset", dataset_name)
                    rows.append(existing)
                    print(f"proxy transform complete, skipping {dataset_name} [{variant}]")
                    continue
            print(f"running {dataset_name} [{variant}]")
            rows.append(run_proxy_variant(dataset_name, variant, model_names, exp_cfg))

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary_path = Path(args.output_root) / "proxy_transform_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"proxy transform summary written to {summary_path}")


if __name__ == "__main__":
    main()
