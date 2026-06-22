from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import (
    DATASET_REGISTRY,
    clone_experiment_config,
    get_default_experiment_config,
    get_dataset_config,
    get_fmsd_model_configs,
    get_midas_model_configs,
    get_spotlight_model_configs,
    get_tabular_foundation_model_configs,
)
from risk_models.cv_runner import run_ablation_suite, run_benchmark


def resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return sorted(DATASET_REGISTRY)
    return [name.strip() for name in dataset_arg.split(",") if name.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-grade FMSD credit-risk experiments.")
    parser.add_argument("--datasets", default="all", help="Dataset name, comma list, or 'all'.")
    parser.add_argument("--repeats", type=int, default=20, help="Repeated train/validation/test splits.")
    parser.add_argument("--output-root", default="outputs/fmsd", help="Root directory for FMSD artifacts.")
    parser.add_argument(
        "--include-tabpfn",
        action="store_true",
        help="Include TabPFN baseline. Requires license acceptance and TABPFN_TOKEN.",
    )
    parser.add_argument(
        "--model-set",
        choices=["classical", "tabpfn", "all", "spotlight"],
        default="classical",
        help=(
            "Model group to run. Use `tabpfn` to resume only the TabPFN baseline, "
            "or `spotlight` for the full reviewer-response model set."
        ),
    )
    parser.add_argument("--skip-no-calibration", action="store_true", help="Skip the uncalibrated comparison run.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip German-credit ablations.")
    parser.add_argument("--skip-weak-label", action="store_true", help="Skip German-credit weak-label sensitivity.")
    parser.add_argument("--no-reliability", action="store_true", help="Skip reliability diagrams.")
    parser.add_argument("--force", action="store_true", help="Rerun complete artifacts instead of resuming.")
    return parser


def print_topline(dataset_name: str, result: dict, label: str) -> None:
    aggregate = result["aggregate_metrics"].sort_values("AUC", ascending=False)
    top = aggregate.iloc[0]
    ece_col = "ECE (10-bin)" if "ECE (10-bin)" in aggregate.columns else "ECE"
    print(
        f"{label}/{dataset_name}: "
        f"best={top['Model']} auc={top['AUC']:.4f} "
        f"brier={top['Brier']:.4f} ece={top[ece_col]:.4f}"
    )


def has_complete_run(output_root: str, mode: str, dataset_name: str, model_names: set[str], repeats: int) -> bool:
    aggregate_path = Path(output_root) / mode / dataset_name / "aggregate_metrics.csv"
    if not aggregate_path.exists():
        return False
    aggregate = pd.read_csv(aggregate_path)
    if aggregate.empty or not {"Model", "n_splits"}.issubset(aggregate.columns):
        return False
    if not model_names.issubset(set(aggregate["Model"])):
        return False
    return bool((aggregate.loc[aggregate["Model"].isin(model_names), "n_splits"] >= repeats).all())


def run_main_benchmarks(args, model_configs, dataset_names: list[str]) -> None:
    base_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=args.repeats,
        output_root=args.output_root,
        calibration_method="temperature",
        save_reliability=not args.no_reliability,
        save_shap=False,
    )
    no_cal_cfg = clone_experiment_config(base_cfg, calibration_method="none")

    for dataset_name in dataset_names:
        dataset_config = get_dataset_config(dataset_name)
        model_names = {config.name for config in model_configs}
        if not args.force and has_complete_run(args.output_root, "benchmark_calibrated", dataset_name, model_names, args.repeats):
            print(f"calibrated/{dataset_name}: complete, skipping")
        else:
            calibrated = run_benchmark(dataset_config, model_configs, base_cfg, mode="benchmark_calibrated")
            print_topline(dataset_name, calibrated, "calibrated")

        if not args.skip_no_calibration:
            if not args.force and has_complete_run(args.output_root, "benchmark_uncalibrated", dataset_name, model_names, args.repeats):
                print(f"uncalibrated/{dataset_name}: complete, skipping")
            else:
                uncalibrated = run_benchmark(dataset_config, model_configs, no_cal_cfg, mode="benchmark_uncalibrated")
                print_topline(dataset_name, uncalibrated, "uncalibrated")


def run_sensitivity_suites(args) -> None:
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=args.repeats,
        output_root=args.output_root,
        calibration_method="temperature",
        save_reliability=not args.no_reliability,
        save_shap=False,
    )

    german_config = get_dataset_config("german_credit")
    if not args.skip_ablation:
        result = run_ablation_suite(german_config, exp_cfg)
        print_topline("german_credit", result, "ablation")

    if not args.skip_weak_label:
        from main import run_weak_label_mode

        weak_args = argparse.Namespace(
            dataset="german_credit",
            repeats=args.repeats,
            output_root=args.output_root,
            calibration_method="temperature",
            save_shap=False,
            no_reliability=args.no_reliability,
            no_subgroups=False,
            mode="weak_label",
        )
        run_weak_label_mode(weak_args)


def main() -> None:
    args = build_parser().parse_args()
    dataset_names = resolve_datasets(args.datasets)
    unknown = [name for name in dataset_names if name not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")

    if args.model_set == "tabpfn":
        model_configs = get_tabular_foundation_model_configs()
    elif args.model_set == "spotlight":
        model_configs = get_spotlight_model_configs(include_tabpfn=True, include_tabicl=True)
    else:
        model_configs = get_fmsd_model_configs(include_tabpfn=args.include_tabpfn or args.model_set == "all")
    run_main_benchmarks(args, model_configs, dataset_names)
    if "german_credit" in dataset_names or args.datasets == "all":
        run_sensitivity_suites(args)

    print(f"FMSD artifacts written under {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
