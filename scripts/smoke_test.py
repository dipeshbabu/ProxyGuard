from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import (
    DATASET_REGISTRY,
    clone_experiment_config,
    get_benchmark_model_configs,
    get_dataset_config,
    get_default_experiment_config,
)
from risk_models.cv_runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight smoke test for credit benchmark pipelines.")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument("--output-root", default="outputs/smoke", help="Output directory for smoke artifacts.")
    return parser


def resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return sorted(DATASET_REGISTRY)
    return [dataset_arg]


def main() -> None:
    args = build_parser().parse_args()
    model_configs = get_benchmark_model_configs()[:3]
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=1,
        output_root=args.output_root,
        save_reliability=False,
        save_shap=False,
    )

    for dataset_name in resolve_datasets(args.dataset):
        result = run_benchmark(
            get_dataset_config(dataset_name),
            model_configs,
            exp_cfg,
            mode="smoke",
        )
        aggregate = result["aggregate_metrics"].sort_values("AUC", ascending=False)
        top = aggregate.iloc[0]
        print(f"{dataset_name}: best_model={top['Model']} auc={top['AUC']:.4f}")

    print(f"Smoke test outputs written to {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
