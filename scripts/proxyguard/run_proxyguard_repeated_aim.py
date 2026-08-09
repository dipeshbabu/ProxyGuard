from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.tabular.config import get_dataset_config  # noqa: E402
from scripts.proxyguard.tabular.data import load_dataset  # noqa: E402
from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
    paired_prediction_losses,
)
from scripts.proxyguard.run_proxyguard_aim_audit import (  # noqa: E402
    _mean_normalized_cost,
    fit_aim_release,
    fit_selected_procedure,
    make_split,
)

REQUIREMENTS = [
    RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
]


def _write_split(
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    frame = X.reset_index(drop=True).copy()
    frame["__target__"] = y.reset_index(drop=True).to_numpy(dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _regret_map(losses: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "Brier": losses["brier_regret"].to_numpy(),
        "Clipped log loss": losses["logloss_regret"].to_numpy(),
        "Cost5x": losses["cost5x_regret"].to_numpy(),
    }


def run_repeated_aim(
    registry: dict,
    output_root: Path,
    dataset_subset: list[str] | None = None,
    epsilon_subset: list[float] | None = None,
    release_subset: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aim = registry["modern_generators"]["aim"]
    alpha = float(registry["risk_control"]["alpha"])
    registered_datasets = list(aim["datasets"])
    registered_epsilons = [float(value) for value in aim["epsilons"]]
    datasets = dataset_subset or registered_datasets
    epsilons = epsilon_subset or registered_epsilons
    if not set(datasets).issubset(registered_datasets):
        raise ValueError("dataset_subset contains an unregistered dataset.")
    if not set(epsilons).issubset(registered_epsilons):
        raise ValueError("epsilon_subset contains an unregistered value.")
    base_release_seeds = [int(value) for value in aim["release_seeds"]]
    expected_releases = int(aim["releases_per_cell"])
    if len(base_release_seeds) != expected_releases:
        raise ValueError("The registry release seed count does not match releases_per_cell.")
    release_indices = release_subset or list(range(1, expected_releases + 1))
    if not set(release_indices).issubset(range(1, expected_releases + 1)):
        raise ValueError("release_subset contains an unregistered release index.")

    all_regrets: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    loss_frames: list[pd.DataFrame] = []

    for dataset_name in datasets:
        dataset_offset = registered_datasets.index(dataset_name)
        bundle = load_dataset(get_dataset_config(dataset_name))
        X = bundle["X"].reset_index(drop=True).astype(float)
        y = pd.Series(bundle["y"]).reset_index(drop=True).astype(int)
        split_seed = base_release_seeds[0] + dataset_offset * 101
        split = make_split(X, y, split_seed)

        (
            source_model_name,
            source_model,
            source_threshold,
            _source_validation_probability,
        ) = fit_selected_procedure(
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
            seed=split_seed,
        )
        source_audit_probability = source_model.predict_proba(split.X_audit)[:, 1]
        train_table = split.X_train.copy()
        train_table["__target__"] = split.y_train.to_numpy()
        public_schema = X.copy()
        public_schema["__target__"] = y.to_numpy()

        split_root = output_root / "splits" / dataset_name
        _write_split(split.X_train, split.y_train, split_root / "members.csv")
        _write_split(
            split.X_validation,
            split.y_validation,
            split_root / "reference.csv",
        )
        _write_split(split.X_audit, split.y_audit, split_root / "nonmembers.csv")

        for epsilon in epsilons:
            epsilon_index = registered_epsilons.index(epsilon)
            for release_number in release_indices:
                release_index = release_number - 1
                base_seed = base_release_seeds[release_index]
                release_seed = (
                    base_seed
                    + dataset_offset * 10_000
                    + epsilon_index * 1_000
                )
                candidate = (
                    f"{dataset_name}::aim_e{epsilon:g}::release_{release_index + 1}"
                )
                release_path = (
                    output_root
                    / "releases"
                    / dataset_name
                    / f"aim_e{epsilon:g}_release_{release_index + 1}.csv"
                )
                if release_path.exists():
                    release_frame = pd.read_csv(release_path)
                    synthetic = release_frame.drop(columns=["__target__"])
                    synthetic["__target__"] = release_frame["__target__"]
                else:
                    synthetic = fit_aim_release(
                        train_table=train_table,
                        public_schema=public_schema,
                        epsilon=epsilon,
                        delta=float(aim["delta"]),
                        seed=release_seed,
                        bins=int(aim["bins"]),
                        max_model_size=int(aim["max_model_size"]),
                        num_marginals=int(aim["num_marginals"]),
                    )
                synthetic_y = (
                    pd.to_numeric(synthetic.pop("__target__"), errors="coerce")
                    .fillna(0)
                    .round()
                    .clip(0, 1)
                    .astype(int)
                )
                if synthetic_y.nunique() < 2:
                    raise RuntimeError(f"{candidate} generated only one target class.")
                synthetic_X = synthetic.reindex(columns=X.columns).astype(float)
                release_frame = synthetic_X.copy()
                release_frame["__target__"] = synthetic_y.to_numpy()
                release_path.parent.mkdir(parents=True, exist_ok=True)
                release_frame.to_csv(release_path, index=False)
                print(f"saved {release_path}", flush=True)

                (
                    proxy_model_name,
                    proxy_model,
                    proxy_threshold,
                    _proxy_validation_probability,
                ) = fit_selected_procedure(
                    synthetic_X,
                    synthetic_y,
                    split.X_validation,
                    split.y_validation,
                    seed=release_seed,
                )
                proxy_audit_probability = proxy_model.predict_proba(split.X_audit)[:, 1]
                losses = paired_prediction_losses(
                    y_true=split.y_audit,
                    source_probability=source_audit_probability,
                    proxy_probability=proxy_audit_probability,
                    source_thresholds={5.0: source_threshold},
                    proxy_thresholds={5.0: proxy_threshold},
                    record_ids=split.audit_positions,
                )
                losses.insert(0, "Release", release_index + 1)
                losses.insert(0, "Epsilon", epsilon)
                losses.insert(0, "Dataset", dataset_name)
                losses.insert(0, "Candidate", candidate)
                loss_frames.append(losses)
                all_regrets[candidate] = _regret_map(losses)
                diagnostics.append(
                    {
                        "Candidate": candidate,
                        "Dataset": dataset_name,
                        "Mechanism": "AIM",
                        "Epsilon": epsilon,
                        "Release": release_index + 1,
                        "ReleaseSeed": release_seed,
                        "AuditN": len(split.y_audit),
                        "SourceModel": source_model_name,
                        "ProxyModel": proxy_model_name,
                        "SourceAUC": roc_auc_score(
                            split.y_audit,
                            source_audit_probability,
                        ),
                        "ProxyAUC": roc_auc_score(
                            split.y_audit,
                            proxy_audit_probability,
                        ),
                        "AUCChange": roc_auc_score(
                            split.y_audit,
                            proxy_audit_probability,
                        )
                        - roc_auc_score(split.y_audit, source_audit_probability),
                        "SourceCost5x": _mean_normalized_cost(
                            split.y_audit,
                            source_audit_probability,
                            source_threshold,
                            5.0,
                        ),
                        "ProxyCost5x": _mean_normalized_cost(
                            split.y_audit,
                            proxy_audit_probability,
                            proxy_threshold,
                            5.0,
                        ),
                        "SourceThreshold": source_threshold,
                        "ProxyThreshold": proxy_threshold,
                        "SyntheticPositiveRate": float(synthetic_y.mean()),
                    }
                )
                del (
                    proxy_model,
                    proxy_audit_probability,
                    release_frame,
                    synthetic_X,
                    synthetic_y,
                )
                gc.collect()

    joint = audit_proxy_candidates(
        all_regrets,
        requirements=REQUIREMENTS,
        alpha=alpha,
        bound_method=str(registry["risk_control"]["bound_method"]),
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["Cost5xChange"] = (
        diagnostics_frame["ProxyCost5x"] - diagnostics_frame["SourceCost5x"]
    )
    joint_summary = joint.candidate_summary.rename(
        columns={"Status": "JointFamilyStatus"}
    )
    joint_summary = joint_summary.merge(
        diagnostics_frame,
        on="Candidate",
        how="left",
        validate="one_to_one",
    )

    individual_rows: list[pd.DataFrame] = []
    for candidate, regrets in all_regrets.items():
        result = audit_proxy_candidates(
            {candidate: regrets},
            requirements=REQUIREMENTS,
            alpha=alpha,
            bound_method=str(registry["risk_control"]["bound_method"]),
        )
        individual_rows.append(
            result.candidate_summary[
                ["Candidate", "Status", "WorstUpperRegretBound"]
            ].rename(
                columns={
                    "Status": "IndividualReleaseStatus",
                    "WorstUpperRegretBound": "IndividualWorstUpperRegretBound",
                }
            )
        )
    joint_summary = joint_summary.merge(
        pd.concat(individual_rows, ignore_index=True),
        on="Candidate",
        how="left",
        validate="one_to_one",
    )

    frequency = (
        joint_summary.groupby(["Dataset", "Epsilon"], as_index=False)
        .agg(
            Releases=("Release", "size"),
            JointValidated=(
                "JointFamilyStatus",
                lambda values: int((values == "Validated").sum()),
            ),
            JointViolations=(
                "JointFamilyStatus",
                lambda values: int((values == "Violation detected").sum()),
            ),
            IndividualValidated=(
                "IndividualReleaseStatus",
                lambda values: int((values == "Validated").sum()),
            ),
            IndividualViolations=(
                "IndividualReleaseStatus",
                lambda values: int((values == "Violation detected").sum()),
            ),
            MeanAUCChange=("AUCChange", "mean"),
            SDAUCChange=("AUCChange", "std"),
            MeanCost5xChange=("Cost5xChange", "mean"),
            SDCost5xChange=("Cost5xChange", "std"),
        )
        .sort_values(["Dataset", "Epsilon"], ignore_index=True)
    )
    return (
        joint_summary,
        joint.requirement_detail,
        frequency,
        pd.concat(loss_frames, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prospectively registered repeated-release AIM audit."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_frontier_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_repeated_aim",
    )
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated registered dataset subset.",
    )
    parser.add_argument(
        "--epsilons",
        help="Optional comma-separated registered epsilon subset.",
    )
    parser.add_argument(
        "--releases",
        help="Optional comma-separated registered one-based release indices.",
    )
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_subset = (
        [value.strip() for value in args.datasets.split(",") if value.strip()]
        if args.datasets
        else None
    )
    epsilon_subset = (
        [float(value) for value in args.epsilons.split(",") if value.strip()]
        if args.epsilons
        else None
    )
    release_subset = (
        [int(value) for value in args.releases.split(",") if value.strip()]
        if args.releases
        else None
    )
    summary, detail, frequency, losses = run_repeated_aim(
        registry,
        output_root,
        dataset_subset=dataset_subset,
        epsilon_subset=epsilon_subset,
        release_subset=release_subset,
    )
    summary.to_csv(output_root / "release_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    frequency.to_csv(output_root / "mechanism_variability.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registered_aim": registry["modern_generators"]["aim"],
                "candidate_count": len(summary),
                "interpretation": (
                    "JointFamilyStatus controls the registered family of realized "
                    "releases. Mechanism variability is descriptive because five "
                    "releases do not identify every future release from AIM."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        frequency.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
