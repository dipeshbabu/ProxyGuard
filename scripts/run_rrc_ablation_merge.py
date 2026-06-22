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

from risk_models.configs import clone_experiment_config, get_ablation_model_configs, get_dataset_config, get_default_experiment_config
from risk_models.configs import DATASET_REGISTRY
from risk_models.cv_runner import run_benchmark


DEFAULT_MODELS = {
    "rc_stack",
    "rc_stack_logloss_only",
    "rrc_stack",
    "rrc_stack_kmeans_regions",
    "rrc_stack_risk_regions",
    "rrc_stack_random_regions",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def merge_rows(existing: pd.DataFrame, new: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
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


def merge_artifacts(target_dir: Path, source_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    merge_specs = {
        "aggregate_metrics.csv": ["Model"],
        "split_metrics.csv": ["Model", "split_seed"],
        "subgroup_metrics.csv": ["Model", "split_seed", "SubgroupName", "SubgroupValue"],
        "feature_stability.csv": ["Model"],
    }
    for filename, keys in merge_specs.items():
        merged = merge_rows(read_csv(target_dir / filename), read_csv(source_dir / filename), keys)
        if not merged.empty:
            merged.to_csv(target_dir / filename, index=False)


def parse_csv(value: str) -> list[str]:
    if value == "all":
        return sorted(DATASET_REGISTRY)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused RRC ablations and merge them into paper artifacts.")
    parser.add_argument("--datasets", default="german_credit,australian_credit")
    parser.add_argument("--models", default=",".join(sorted(DEFAULT_MODELS)))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/spotlight_final")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    model_names = set(parse_csv(args.models))
    configs = [config for config in get_ablation_model_configs() if config.name in model_names]
    missing = model_names - {config.name for config in configs}
    if missing:
        raise ValueError(f"Unknown ablation model(s): {', '.join(sorted(missing))}")

    temp_root = Path("outputs") / f"rrc_ablation_merge_{uuid.uuid4().hex}"
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=args.repeats,
        output_root=str(temp_root),
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
    )

    try:
        for dataset_name in parse_csv(args.datasets):
            run_benchmark(get_dataset_config(dataset_name), configs, exp_cfg, mode="ablations")
            source_dir = temp_root / "ablations" / dataset_name
            target_dir = Path(args.output_root) / "ablations" / dataset_name
            merge_artifacts(target_dir, source_dir)
            print(f"merged {', '.join(config.name for config in configs)} into {target_dir}")
    finally:
        if not args.keep_temp and temp_root.exists():
            shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
