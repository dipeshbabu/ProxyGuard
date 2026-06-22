from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import uuid

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    clone_experiment_config,
    clone_model_config,
    get_default_experiment_config,
    get_dataset_config,
)
from risk_models.cv_runner import run_benchmark
from risk_models.reporting import plot_metric_summary, save_latex_table


def resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return sorted(DATASET_REGISTRY)
    return [name.strip() for name in dataset_arg.split(",") if name.strip()]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_model_rows(existing: pd.DataFrame, new: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing
    present_keys = [column for column in key_columns if column in existing.columns and column in new.columns]
    if not present_keys:
        present_keys = ["Model"]
    new_keys = set(map(tuple, new[present_keys].to_numpy()))
    keep_existing = existing[~existing[present_keys].apply(tuple, axis=1).isin(new_keys)]
    merged = pd.concat([keep_existing, new], ignore_index=True)
    sort_columns = [column for column in ["Model", "split_seed", "SubgroupName", "SubgroupValue"] if column in merged.columns]
    return merged.sort_values(sort_columns).reset_index(drop=True) if sort_columns else merged.reset_index(drop=True)


def write_merged_outputs(target_dir: Path, source_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    merge_specs = {
        "aggregate_metrics.csv": ["Model"],
        "split_metrics.csv": ["Model", "split_seed"],
        "subgroup_metrics.csv": ["Model", "split_seed", "SubgroupName", "SubgroupValue"],
        "feature_stability.csv": ["Model"],
    }
    for filename, keys in merge_specs.items():
        merged = merge_model_rows(read_csv(target_dir / filename), read_csv(source_dir / filename), keys)
        if not merged.empty:
            merged.to_csv(target_dir / filename, index=False)

    aggregate = read_csv(target_dir / "aggregate_metrics.csv")
    if not aggregate.empty:
        save_latex_table(
            aggregate,
            target_dir / "aggregate_metrics.tex",
            caption="Aggregate benchmark metrics across repeated splits.",
            label="tab:aggregate_metrics",
        )
        plot_metric_summary(aggregate, target_dir / "auc_summary.png", metric="AUC")
        plot_metric_summary(aggregate, target_dir / "ece_summary.png", metric="ECE (10-bin)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selected registry models and merge them into spotlight artifacts.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/spotlight_final")
    parser.add_argument("--mode", default="benchmark_calibrated")
    parser.add_argument("--calibration-method", choices=["temperature", "none"], default="temperature")
    parser.add_argument(
        "--models",
        default="reliability_ensemble",
        help="Comma-separated MODEL_REGISTRY names to run and merge.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    temp_root = Path("outputs") / f"model_merge_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=args.repeats,
        output_root=str(temp_root),
        calibration_method=args.calibration_method,
        save_reliability=False,
        save_shap=False,
    )
    model_configs = [clone_model_config(MODEL_REGISTRY[name]) for name in model_names]
    try:
        for dataset_name in resolve_datasets(args.datasets):
            run_benchmark(get_dataset_config(dataset_name), model_configs, exp_cfg, mode=args.mode)
            source_dir = temp_root / args.mode / dataset_name
            target_dir = Path(args.output_root) / args.mode / dataset_name
            write_merged_outputs(target_dir, source_dir)
            print(f"merged {', '.join(model_names)} into {target_dir}")
    finally:
        if not args.keep_temp and temp_root.exists():
            shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
