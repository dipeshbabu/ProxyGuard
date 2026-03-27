import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from risk_models.configs import (
    DATASET_REGISTRY,
    ExperimentConfig,
    clone_dataset_config,
    clone_experiment_config,
    get_ablation_model_configs,
    get_benchmark_model_configs,
    get_dataset_config,
    get_default_experiment_config,
)
from risk_models.cv_runner import run_ablation_suite, run_benchmark
from risk_models.diagnostics import warn_if_suspicious_metrics


def _resolve_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "all":
        return sorted(DATASET_REGISTRY)
    return [dataset_arg]


def _build_experiment_config(args) -> ExperimentConfig:
    base = get_default_experiment_config()
    return clone_experiment_config(
        base,
        n_repeats=args.repeats,
        output_root=args.output_root,
        calibration_method=args.calibration_method,
        save_shap=args.save_shap,
        save_reliability=not args.no_reliability,
        run_subgroups=not args.no_subgroups,
        run_weak_label_sensitivity=args.mode == "weak_label",
    )


def _print_summary(result_bundle):
    aggregate = result_bundle["aggregate_metrics"]
    if aggregate.empty:
        print("No aggregate metrics were produced.")
        return
    summary_cols = [col for col in ["Model", "AUC", "ECE (10-bin)", "FeatureCount"] if col in aggregate.columns]
    ranked = aggregate.sort_values("AUC", ascending=False)
    print(ranked[summary_cols].to_string(index=False))
    for _, row in ranked.iterrows():
        warnings = warn_if_suspicious_metrics(
            {
                "AUC": row.get("AUC"),
                "ECE": row.get("ECE (10-bin)"),
                "Accuracy": row.get("Accuracy"),
            },
            row["Model"],
        )
        for warning in warnings:
            print(f"[warning] {warning}")
    output_dir = result_bundle.get("output_dir")
    if output_dir:
        print(f"[artifacts] {output_dir}")
    subgroup_metrics = result_bundle.get("subgroup_metrics")
    if isinstance(subgroup_metrics, pd.DataFrame) and not subgroup_metrics.empty:
        print(f"[subgroups] rows={len(subgroup_metrics)}")
    feature_stability = result_bundle.get("feature_stability")
    if isinstance(feature_stability, pd.DataFrame) and not feature_stability.empty:
        print(f"[stability] rows={len(feature_stability)}")


def run_benchmark_mode(args):
    exp_config = _build_experiment_config(args)
    model_configs = get_benchmark_model_configs()
    for dataset_name in _resolve_datasets(args.dataset):
        dataset_config = get_dataset_config(dataset_name)
        result_bundle = run_benchmark(dataset_config, model_configs, exp_config, mode="benchmark")
        print(f"\nBenchmark results for {dataset_name}:")
        _print_summary(result_bundle)


def run_ablation_mode(args):
    exp_config = _build_experiment_config(args)
    for dataset_name in _resolve_datasets(args.dataset):
        dataset_config = get_dataset_config(dataset_name)
        result_bundle = run_ablation_suite(dataset_config, exp_config)
        print(f"\nAblation results for {dataset_name}:")
        _print_summary(result_bundle)


def _weak_label_variants(dataset_config):
    label_params = dict(dataset_config.label_params)
    return [
        ("baseline", dataset_config),
        (
            "top_frac_25",
            clone_dataset_config(dataset_config, label_params={**label_params, "top_frac": 0.25}),
        ),
        (
            "top_frac_35",
            clone_dataset_config(dataset_config, label_params={**label_params, "top_frac": 0.35}),
        ),
        (
            "threshold_shift_up",
            clone_dataset_config(dataset_config, label_params={**label_params, "threshold_shift": 0.05}),
        ),
        (
            "threshold_shift_down",
            clone_dataset_config(dataset_config, label_params={**label_params, "threshold_shift": -0.05}),
        ),
        (
            "label_noise_05",
            clone_dataset_config(dataset_config, label_params={**label_params, "noise_rate": 0.05}),
        ),
        (
            "label_noise_10",
            clone_dataset_config(dataset_config, label_params={**label_params, "noise_rate": 0.10}),
        ),
        (
            "proxy_drop",
            clone_dataset_config(
                dataset_config,
                drop_columns=["Saving accounts", "Checking account", "Job", "Business_Age"],
            ),
        ),
    ]


def run_weak_label_mode(args):
    selected = _resolve_datasets(args.dataset)
    invalid = [dataset_name for dataset_name in selected if dataset_name != "german_credit"]
    if invalid:
        invalid_list = ", ".join(invalid)
        raise ValueError(
            f"Weak-label mode is only supported for 'german_credit'. Received: {invalid_list}"
        )

    exp_config = _build_experiment_config(args)
    model_configs = get_benchmark_model_configs()
    all_results = []
    for dataset_name in selected:
        base_config = get_dataset_config(dataset_name)
        for variant_name, variant_config in _weak_label_variants(base_config):
            result_bundle = run_benchmark(
                variant_config,
                model_configs,
                exp_config,
                mode=f"weak_label/{variant_name}",
            )
            aggregate = result_bundle["aggregate_metrics"].copy()
            aggregate.insert(0, "Variant", variant_name)
            all_results.append(aggregate)
            print(f"\nWeak-label results for {dataset_name} [{variant_name}]:")
            _print_summary(result_bundle)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = Path(exp_config.output_root) / "weak_label" / "weak_label_summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"\nSaved weak-label summary to {output_path}")


def run_debug_mode(args):
    exp_config = _build_experiment_config(args)
    exp_config = clone_experiment_config(exp_config, n_repeats=1, save_shap=False)
    model_configs = get_benchmark_model_configs()[:3]
    dataset_name = _resolve_datasets(args.dataset)[0]
    dataset_config = get_dataset_config(dataset_name)
    result_bundle = run_benchmark(dataset_config, model_configs, exp_config, mode="debug")
    print(f"\nDebug results for {dataset_name}:")
    _print_summary(result_bundle)


def build_parser():
    parser = argparse.ArgumentParser(description="Small-data credit risk benchmark runner.")
    parser.add_argument("--mode", choices=["benchmark", "ablation", "weak_label", "debug"], default="benchmark")
    parser.add_argument("--dataset", default="german_credit", help="Dataset name or 'all'.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated outer splits.")
    parser.add_argument("--output-root", default=str(Path("outputs")), help="Root directory for experiment outputs.")
    parser.add_argument("--calibration-method", default="temperature", choices=["temperature", "none"])
    parser.add_argument(
        "--save-shap",
        action="store_true",
        help="Reserved for future SHAP artifact export; model-level SHAP hooks exist but runner export is not enabled yet.",
    )
    parser.add_argument("--no-reliability", action="store_true", help="Skip reliability diagram generation.")
    parser.add_argument("--no-subgroups", action="store_true", help="Skip subgroup metrics.")
    return parser


def main():
    args = build_parser().parse_args()
    mode_handlers = {
        "benchmark": run_benchmark_mode,
        "ablation": run_ablation_mode,
        "weak_label": run_weak_label_mode,
        "debug": run_debug_mode,
    }
    mode_handlers[args.mode](args)


if __name__ == "__main__":
    main()
