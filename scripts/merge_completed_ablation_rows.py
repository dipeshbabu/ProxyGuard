from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MERGE_KEYS = {
    "aggregate_metrics.csv": ["Model"],
    "split_metrics.csv": ["Model", "split_seed"],
    "subgroup_metrics.csv": ["Model", "split_seed", "SubgroupName", "SubgroupValue"],
    "feature_stability.csv": ["Model"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed ablation artifacts from a temp root.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", default="outputs/spotlight_final")
    parser.add_argument("--mode", default="ablations")
    args = parser.parse_args()

    source_mode = Path(args.source_root) / args.mode
    target_mode = Path(args.target_root) / args.mode
    for source_dataset_dir in sorted(path for path in source_mode.iterdir() if path.is_dir()):
        target_dataset_dir = target_mode / source_dataset_dir.name
        target_dataset_dir.mkdir(parents=True, exist_ok=True)
        for filename, keys in MERGE_KEYS.items():
            merged = merge_rows(read_csv(target_dataset_dir / filename), read_csv(source_dataset_dir / filename), keys)
            if not merged.empty:
                merged.to_csv(target_dataset_dir / filename, index=False)
        print(f"merged {source_dataset_dir.name}")


if __name__ == "__main__":
    main()
