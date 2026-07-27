from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
    paired_prediction_losses,
)

DEFAULT_REQUIREMENTS = (
    RiskRequirement("brier", tolerance=0.01),
    RiskRequirement("logloss", tolerance=0.01),
    RiskRequirement("cost5x", tolerance=0.01),
)


def available_dataset_variants(output_root: Path) -> dict[str, list[str]]:
    proxy_root = output_root / "proxy_transform"
    if not proxy_root.exists():
        raise FileNotFoundError(f"Proxy-transform output does not exist: {proxy_root}")
    baseline_root = proxy_root / "baseline"
    if not baseline_root.exists():
        raise FileNotFoundError(f"Baseline audit records do not exist: {baseline_root}")

    variants: dict[str, list[str]] = {}
    for dataset_dir in sorted(path for path in baseline_root.iterdir() if path.is_dir()):
        dataset_variants = []
        for variant_dir in sorted(path for path in proxy_root.iterdir() if path.is_dir()):
            if variant_dir.name == "baseline":
                continue
            record_path = variant_dir / dataset_dir.name / "audit_records.csv"
            if record_path.exists():
                dataset_variants.append(variant_dir.name)
        if dataset_variants:
            variants[dataset_dir.name] = dataset_variants
    return variants


def read_audit_records(output_root: Path, dataset: str, variant: str) -> pd.DataFrame:
    path = output_root / "proxy_transform" / variant / dataset / "audit_records.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing audit record file: {path}")
    frame = pd.read_csv(path, dtype={"record_id": str})
    required = {
        "Dataset",
        "Model",
        "split_seed",
        "record_id",
        "y_true",
        "probability",
        "cost_threshold_5x",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def paired_candidate_frame(
    output_root: Path,
    dataset: str,
    variant: str,
    split_seed: int,
) -> pd.DataFrame:
    source = read_audit_records(output_root, dataset, "baseline")
    proxy = read_audit_records(output_root, dataset, variant)
    source = source[source["split_seed"].eq(split_seed)].copy()
    proxy = proxy[proxy["split_seed"].eq(split_seed)].copy()
    if source.empty or proxy.empty:
        raise ValueError(f"No records for {dataset}/{variant} at split seed {split_seed}.")

    keys = ["Dataset", "Model", "split_seed", "record_id"]
    paired = source.merge(
        proxy,
        on=keys,
        how="inner",
        suffixes=("_source", "_proxy"),
        validate="one_to_one",
    )
    if len(paired) != len(source) or len(paired) != len(proxy):
        raise ValueError(f"Source and proxy audit records do not align for {dataset}/{variant}.")
    if not paired["y_true_source"].equals(paired["y_true_proxy"]):
        raise ValueError(f"Source and proxy labels differ for {dataset}/{variant}.")

    losses = paired_prediction_losses(
        y_true=paired["y_true_source"],
        source_probability=paired["probability_source"],
        proxy_probability=paired["probability_proxy"],
        source_thresholds={5.0: float(paired["cost_threshold_5x_source"].iloc[0])},
        proxy_thresholds={5.0: float(paired["cost_threshold_5x_proxy"].iloc[0])},
        record_ids=paired["record_id"],
    )
    losses.insert(0, "split_seed", split_seed)
    losses.insert(0, "Variant", variant)
    losses.insert(0, "Dataset", dataset)
    return losses


def build_real_audit(
    output_root: Path,
    split_seed: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frames = []
    candidate_regrets: dict[str, dict[str, pd.Series]] = {}
    for dataset, variants in available_dataset_variants(output_root).items():
        for variant in variants:
            frame = paired_candidate_frame(output_root, dataset, variant, split_seed)
            candidate = f"{dataset}/{variant}"
            candidate_frames.append(frame)
            candidate_regrets[candidate] = {
                "brier": frame["brier_regret"],
                "logloss": frame["logloss_regret"],
                "cost5x": frame["cost5x_regret"],
            }

    if not candidate_frames:
        raise ValueError("No paired proxy candidates were found.")
    audit = audit_proxy_candidates(
        candidate_regrets,
        DEFAULT_REQUIREMENTS,
        alpha=alpha,
    )
    paired_losses = pd.concat(candidate_frames, ignore_index=True)
    return audit.candidate_summary, audit.requirement_detail, paired_losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply ProxyGuard to saved source/proxy per-record predictions."
    )
    parser.add_argument("--input-root", default="outputs/proxyguard_proxy_audit")
    parser.add_argument("--output-root", default="outputs/proxyguard_real_audit")
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    summary, detail, losses = build_real_audit(
        output_root=Path(args.input_root),
        split_seed=args.split_seed,
        alpha=args.alpha,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "candidate_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)

    status_counts = summary["Status"].value_counts().rename_axis("Status").reset_index(name="Candidates")
    print(status_counts.to_string(index=False))
    print()
    print(
        summary[
            [
                "Candidate",
                "AuditNMin",
                "HolmAdjustedPValue",
                "Status",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
