from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import DATASET_REGISTRY, get_spotlight_model_configs


REQUIRED_MAIN_METRICS = {
    "AUC",
    "AUPRC",
    "Brier",
    "ECE (10-bin)",
    "ECE (15-bin)",
    "ECE (20-bin)",
    "AdaptiveECE (10-bin)",
    "LogLoss",
    "CalibrationSlope",
    "DecisionCost5x",
    "DecisionCost10x",
    "PreprocessTimeSec",
    "ModelFitTimeSec",
    "CalibrationTimeSec",
    "InferenceTimeSec",
    "PeakMemoryMB",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_mode(output_root: Path, mode: str, filename: str) -> pd.DataFrame:
    rows = []
    mode_dir = output_root / mode
    if not mode_dir.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        df = read_csv(dataset_dir / filename)
        if df.empty:
            continue
        if "Dataset" not in df.columns:
            df.insert(0, "Dataset", dataset_dir.name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def status(ok: bool) -> str:
    return "PASS" if ok else "GAP"


def add_check(rows: list[dict[str, str]], name: str, ok: bool, evidence: str, fix: str) -> None:
    rows.append(
        {
            "Check": name,
            "Status": status(ok),
            "Evidence": evidence,
            "Fix if GAP": fix,
        }
    )


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# Reviewer Readiness Audit",
        "",
        "This file is generated from local experiment artifacts. It is intended to catch reviewer-style rejection points before submission.",
        "",
        "| Check | Status | Evidence | Fix if GAP |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Check']} | {row['Status']} | {row['Evidence']} | {row['Fix if GAP']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-readiness checklist from benchmark artifacts.")
    parser.add_argument("--output-root", default="outputs/spotlight_final")
    parser.add_argument("--asset-root", default="paper_assets/spotlight_readiness")
    parser.add_argument("--min-datasets", type=int, default=10)
    parser.add_argument("--min-splits", type=int, default=20)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    asset_root = Path(args.asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    calibrated = collect_mode(output_root, "benchmark_calibrated", "aggregate_metrics.csv")
    splits = collect_mode(output_root, "benchmark_calibrated", "split_metrics.csv")
    subgroups = collect_mode(output_root, "benchmark_calibrated", "subgroup_metrics.csv")
    ablations = collect_mode(output_root, "ablations", "aggregate_metrics.csv")
    expected_models = {config.name for config in get_spotlight_model_configs(include_tabpfn=True, include_tabicl=True)}
    observed_models = set(calibrated["Model"]) if "Model" in calibrated else set()
    observed_datasets = set(calibrated["Dataset"]) if "Dataset" in calibrated else set()
    rows: list[dict[str, str]] = []

    add_check(
        rows,
        "Dataset breadth",
        len(observed_datasets) >= args.min_datasets,
        f"{len(observed_datasets)} datasets observed; target is at least {args.min_datasets}.",
        "Add more public risk-style tabular tasks or narrow the venue claim away from spotlight/top-15%.",
    )
    add_check(
        rows,
        "Real-label coverage",
        sum(DATASET_REGISTRY[name].label_type == "real" for name in observed_datasets if name in DATASET_REGISTRY) >= 3,
        "Real-label datasets are counted from DATASET_REGISTRY.",
        "Avoid generalizing from weak-label German; add real-label tasks or make German a stress-test only.",
    )
    add_check(
        rows,
        "Spotlight method present",
        {"reliability_ensemble", "rc_stack", "rc_stack_dr", "rrc_stack"}.issubset(observed_models),
        f"Observed models: {', '.join(sorted(observed_models)) or 'none'}.",
        "Run `scripts/run_fmsd_experiments.py --model-set spotlight` or merge the missing proposed method rows.",
    )
    add_check(
        rows,
        "Baseline scope",
        expected_models.issubset(observed_models),
        f"Missing: {', '.join(sorted(expected_models - observed_models)) or 'none'}.",
        "Include XGB, LightGBM, CatBoost, HistGB, compact XGB, TabPFN, TabICL, RCE, RC-Stack, RC-Stack-DR, and RRC-Stack in the main run.",
    )
    add_check(
        rows,
        "Split count consistency",
        (not calibrated.empty) and "n_splits" in calibrated and int(calibrated["n_splits"].min()) >= args.min_splits,
        f"Minimum observed split count: {int(calibrated['n_splits'].min()) if 'n_splits' in calibrated and not calibrated.empty else 'none'}.",
        "Complete missing split rows or explicitly mark compute-limited comparisons as secondary.",
    )
    add_check(
        rows,
        "Calibration sensitivity metrics",
        REQUIRED_MAIN_METRICS.issubset(set(calibrated.columns)),
        f"Missing metrics: {', '.join(sorted(REQUIRED_MAIN_METRICS - set(calibrated.columns))) or 'none'}.",
        "Regenerate outputs with the current evaluator so ECE variants, adaptive ECE, cost, and timing are present.",
    )
    add_check(
        rows,
        "Paired split artifacts",
        (not splits.empty) and {"Dataset", "Model", "split_seed", "AUC", "Brier", "ECE (10-bin)"}.issubset(splits.columns),
        f"{len(splits)} split rows observed.",
        "Keep split_metrics.csv for every dataset so paired wins and significance tests are possible.",
    )
    add_check(
        rows,
        "Subgroup calibration artifacts",
        (not subgroups.empty) and {"SubgroupName", "SubgroupValue", "ECE", "Brier"}.issubset(subgroups.columns),
        f"{len(subgroups)} subgroup rows observed.",
        "Enable subgroup evaluation and report worst-group or spread metrics in the appendix.",
    )
    add_check(
        rows,
        "Method ablations",
        (not ablations.empty)
        and {
            "reliability_ensemble",
            "reliability_ensemble_logloss_only",
            "reliability_ensemble_no_cost",
            "reliability_ensemble_no_ece",
            "rrc_stack",
            "rrc_stack_kmeans_regions",
            "rrc_stack_risk_regions",
            "rrc_stack_random_regions",
        }.issubset(set(ablations["Model"])),
        f"Ablation models observed: {', '.join(sorted(set(ablations['Model']))) if 'Model' in ablations else 'none'}.",
        "Run German and at least one real-label dataset ablations for the new method.",
    )

    audit = pd.DataFrame(rows)
    audit.to_csv(asset_root / "reviewer_readiness_audit.csv", index=False)
    write_markdown(rows, asset_root / "reviewer_readiness_audit.md")
    print(f"Reviewer readiness audit written under {asset_root.resolve()}")


if __name__ == "__main__":
    main()
