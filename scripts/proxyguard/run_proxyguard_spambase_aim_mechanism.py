from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from proxyguard.core import (
    RiskRequirement,
    audit_proxy_mechanisms,
    paired_prediction_losses,
)
from scripts.proxyguard.run_proxyguard_aim_audit import (
    _mean_normalized_cost,
    fit_aim_release,
    fit_selected_procedure,
)

FEATURES = [f"feature_{index:02d}" for index in range(57)]
TARGET = "__target__"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_spambase(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, names=[*FEATURES, "spam"])
    if frame.shape[1] != 58 or frame.empty:
        raise ValueError("Spambase must contain 57 features and one target.")
    if not set(frame["spam"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected Spambase target value.")
    X = frame[FEATURES].astype(float)
    # UCI documents false positives as the costly error. Inverting the target
    # makes a false negative a legitimate message classified as spam.
    y = 1 - frame["spam"].astype(int)
    return X, y


def _split_development(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train, validation = train_test_split(
        np.arange(len(y)),
        test_size=validation_fraction,
        random_state=seed,
        stratify=y,
    )
    return (
        X.iloc[train].reset_index(drop=True),
        X.iloc[validation].reset_index(drop=True),
        y.iloc[train].reset_index(drop=True),
        y.iloc[validation].reset_index(drop=True),
    )


def _rounded_ceiling(value: float, margin: float) -> float:
    return min(1.0, math.ceil(1000.0 * (value + margin)) / 1000.0)


def _write_rows(lines: list[str], positions: np.ndarray, path: Path) -> None:
    path.write_text(
        "".join(lines[int(position)] for position in positions),
        encoding="utf-8",
    )


def _requirements(registry: dict) -> list[RiskRequirement]:
    return [
        RiskRequirement(
            name=str(requirement["name"]),
            tolerance=float(requirement["tolerance"]),
            lower=float(requirement["lower"]),
            upper=float(requirement["upper"]),
            estimand=str(requirement["estimand"]),
        )
        for requirement in registry["requirements"]
    ]


def _selected_features(registry: dict) -> list[str]:
    indices = registry["dataset"].get(
        "selected_feature_indices",
        list(range(len(FEATURES))),
    )
    return [FEATURES[int(index)] for index in indices]


def _audit_values(
    losses: pd.DataFrame,
    registry: dict,
) -> dict[str, np.ndarray]:
    return {
        str(requirement["name"]): losses[
            str(requirement["value_column"])
        ].to_numpy()
        for requirement in registry["requirements"]
    }


def prepare_registry(
    *,
    partition_registry_path: Path,
    archive_path: Path,
    sealed_root: Path,
    audit_registry_path: Path,
) -> dict:
    if audit_registry_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen registry: {audit_registry_path}"
        )
    partition_registry = json.loads(
        partition_registry_path.read_text(encoding="utf-8")
    )
    expected_partition_hash = (
        partition_registry_path.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if _sha256(partition_registry_path) != expected_partition_hash:
        raise ValueError("Partition registry hash mismatch.")
    dataset = partition_registry["dataset"]
    if _sha256(archive_path) != str(dataset["archive_sha256"]):
        raise ValueError("Official UCI archive hash mismatch.")

    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read("spambase.data").decode("utf-8")
    lines = raw.splitlines(keepends=True)
    if len(lines) != int(dataset["expected_rows"]):
        raise ValueError(
            f"Expected {dataset['expected_rows']} rows, found {len(lines)}."
        )

    partition = partition_registry["partition"]
    rng = np.random.default_rng(int(partition["reserve_seed"]))
    audit_positions = np.sort(
        rng.choice(
            len(lines),
            size=int(partition["sealed_audit_rows"]),
            replace=False,
        )
    )
    audit_set = set(audit_positions.tolist())
    development_positions = np.asarray(
        [
            position
            for position in range(len(lines))
            if position not in audit_set
        ],
        dtype=int,
    )

    sealed_root.mkdir(parents=True, exist_ok=False)
    development_path = sealed_root / "development.data"
    audit_path = sealed_root / "sealed_audit.data"
    development_positions_path = sealed_root / "development_positions.csv"
    audit_positions_path = sealed_root / "audit_positions.csv"
    _write_rows(lines, development_positions, development_path)
    _write_rows(lines, audit_positions, audit_path)
    pd.DataFrame({"RecordPosition": development_positions}).to_csv(
        development_positions_path,
        index=False,
    )
    pd.DataFrame({"RecordPosition": audit_positions}).to_csv(
        audit_positions_path,
        index=False,
    )

    X_development, y_development = _read_spambase(development_path)
    X_train, X_validation, y_train, y_validation = _split_development(
        X_development,
        y_development,
        seed=int(partition["development_seed"]),
        validation_fraction=float(partition["development_validation_fraction"]),
    )
    (
        source_model_name,
        _source_model,
        source_threshold,
        source_validation_probability,
    ) = fit_selected_procedure(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=int(partition["development_seed"]),
    )
    validation_losses = paired_prediction_losses(
        y_true=y_validation,
        source_probability=source_validation_probability,
        proxy_probability=source_validation_probability,
        source_thresholds={5.0: source_threshold},
        proxy_thresholds={5.0: source_threshold},
    )
    relative_tolerance = float(
        partition_registry["claims"]["relative_risk_tolerance"]
    )
    value_columns = {
        "Brier": ("brier_regret", "brier_proxy"),
        "Log loss": ("logloss_regret", "logloss_proxy"),
        "Cost5x": ("cost5x_regret", "cost5x_proxy"),
    }
    requirements: list[dict[str, float | str]] = []
    for loss_name, (regret_column, risk_column) in value_columns.items():
        requirements.append(
            {
                "name": f"{loss_name} transfer",
                "estimand": "relative_regret",
                "value_column": regret_column,
                "tolerance": relative_tolerance,
                "lower": -1.0,
                "upper": 1.0,
            }
        )
        requirements.append(
            {
                "name": f"Proxy {loss_name} risk",
                "estimand": "absolute_risk",
                "value_column": risk_column,
                "tolerance": _rounded_ceiling(
                    float(validation_losses[risk_column].mean()),
                    0.05,
                ),
                "lower": 0.0,
                "upper": 1.0,
            }
        )

    audit_registry = {
        "registry_version": "1.0",
        "frozen_on": "2026-07-28",
        "analysis_status": (
            "Prospective AIM mechanism audit. Raw rows were partitioned before "
            "any target value was parsed. Requirements, release seeds, and the "
            "mechanism configuration were fixed before the sealed file opens."
        ),
        "partition_registry": str(partition_registry_path),
        "partition_registry_sha256": expected_partition_hash,
        "dataset": {
            **dataset,
            "archive_path": str(archive_path),
        },
        "partition": {
            "rule": "Uniform row sample without replacement before parsing.",
            "development_records": len(development_positions),
            "audit_records": len(audit_positions),
            "development_path": str(development_path),
            "development_sha256": _sha256(development_path),
            "development_positions_path": str(development_positions_path),
            "development_positions_sha256": _sha256(
                development_positions_path
            ),
            "sealed_audit_path": str(audit_path),
            "sealed_audit_sha256": _sha256(audit_path),
            "audit_positions_path": str(audit_positions_path),
            "audit_positions_sha256": _sha256(audit_positions_path),
            "audit_outcomes_inspected_during_preparation": False,
        },
        "development": {
            "split_seed": int(partition["development_seed"]),
            "validation_fraction": float(
                partition["development_validation_fraction"]
            ),
            "source_model": source_model_name,
            "source_threshold": source_threshold,
            "source_validation_risks": {
                "Brier": float(validation_losses["brier_proxy"].mean()),
                "NormalizedClippedLogLoss": float(
                    validation_losses["logloss_proxy"].mean()
                ),
                "NormalizedCost5x": float(
                    validation_losses["cost5x_proxy"].mean()
                ),
            },
        },
        "mechanism": partition_registry["mechanism"],
        "requirements": requirements,
        "minimum_reliability": float(
            partition_registry["claims"]["minimum_reliability"]
        ),
        "total_alpha": float(
            partition_registry["claims"]["total_false_validation_alpha"]
        ),
        "release_error_share": float(
            partition_registry["claims"]["release_error_share"]
        ),
        "violation_total_alpha": float(
            partition_registry["claims"]["violation_total_alpha"]
        ),
        "violation_release_error_share": float(
            partition_registry["claims"]["violation_release_error_share"]
        ),
        "bound_method": "empirical_bernstein",
        "mechanism_count_mode": "holm",
    }
    audit_registry_path.write_text(
        json.dumps(audit_registry, indent=2),
        encoding="utf-8",
    )
    registry_hash = _sha256(audit_registry_path)
    audit_registry_path.with_suffix(".sha256").write_text(
        f"{registry_hash}  {audit_registry_path.name}\n",
        encoding="utf-8",
    )
    return audit_registry


def prepare_amended_registry(
    *,
    parent_registry_path: Path,
    amendment_path: Path,
    output_registry_path: Path,
) -> dict:
    if output_registry_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen registry: {output_registry_path}"
        )
    parent = json.loads(parent_registry_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    parent_hash = (
        parent_registry_path.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    amendment_hash = (
        amendment_path.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if _sha256(parent_registry_path) != parent_hash:
        raise ValueError("Parent audit registry hash mismatch.")
    if _sha256(amendment_path) != amendment_hash:
        raise ValueError("Compute amendment hash mismatch.")
    if parent_hash != str(amendment["parent_registry_sha256"]):
        raise ValueError("Amendment does not name the parent registry hash.")

    revised = json.loads(json.dumps(parent))
    changes = amendment["changes"]
    feature_indices = [int(index) for index in changes["selected_feature_indices"]]
    revised["registry_version"] = "1.1"
    revised["analysis_status"] = (
        "Prospective amended AIM audit. A computational amendment restricted "
        "the feature schema before the sealed target was parsed."
    )
    revised["parent_registry"] = str(parent_registry_path)
    revised["parent_registry_sha256"] = parent_hash
    revised["compute_amendment"] = str(amendment_path)
    revised["compute_amendment_sha256"] = amendment_hash
    revised["dataset"]["selected_feature_indices"] = feature_indices
    revised["dataset"]["selected_features"] = [
        FEATURES[index] for index in feature_indices
    ]
    for key in ("bins", "max_model_size", "num_marginals"):
        revised["mechanism"][key] = int(changes[key])

    X_development, y_development = _read_spambase(
        Path(revised["partition"]["development_path"])
    )
    feature_names = _selected_features(revised)
    X_development = X_development[feature_names]
    X_train, X_validation, y_train, y_validation = _split_development(
        X_development,
        y_development,
        seed=int(revised["development"]["split_seed"]),
        validation_fraction=float(
            revised["development"]["validation_fraction"]
        ),
    )
    (
        source_model_name,
        _source_model,
        source_threshold,
        source_validation_probability,
    ) = fit_selected_procedure(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=int(revised["development"]["split_seed"]),
    )
    validation_losses = paired_prediction_losses(
        y_true=y_validation,
        source_probability=source_validation_probability,
        proxy_probability=source_validation_probability,
        source_thresholds={5.0: source_threshold},
        proxy_thresholds={5.0: source_threshold},
    )
    risk_columns = {
        "Proxy Brier risk": "brier_proxy",
        "Proxy Log loss risk": "logloss_proxy",
        "Proxy Cost5x risk": "cost5x_proxy",
    }
    for requirement in revised["requirements"]:
        name = str(requirement["name"])
        if name in risk_columns:
            requirement["tolerance"] = _rounded_ceiling(
                float(validation_losses[risk_columns[name]].mean()),
                0.05,
            )
    revised["development"]["source_model"] = source_model_name
    revised["development"]["source_threshold"] = source_threshold
    revised["development"]["source_validation_risks"] = {
        "Brier": float(validation_losses["brier_proxy"].mean()),
        "NormalizedClippedLogLoss": float(
            validation_losses["logloss_proxy"].mean()
        ),
        "NormalizedCost5x": float(
            validation_losses["cost5x_proxy"].mean()
        ),
    }
    output_registry_path.write_text(
        json.dumps(revised, indent=2),
        encoding="utf-8",
    )
    revised_hash = _sha256(output_registry_path)
    output_registry_path.with_suffix(".sha256").write_text(
        f"{revised_hash}  {output_registry_path.name}\n",
        encoding="utf-8",
    )
    return revised


def _fit_proxy_release(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    release_seed: int,
    mechanism: dict,
    feature_names: list[str],
) -> tuple[str, object, float]:
    train_table = X_train.copy()
    train_table[TARGET] = y_train.to_numpy()
    release = fit_aim_release(
        train_table=train_table,
        public_schema=train_table,
        epsilon=float(mechanism["epsilon"]),
        delta=float(mechanism["delta"]),
        seed=release_seed,
        bins=int(mechanism["bins"]),
        max_model_size=int(mechanism["max_model_size"]),
        num_marginals=int(mechanism["num_marginals"]),
    )
    release = release.apply(pd.to_numeric, errors="raise")
    release[TARGET] = release[TARGET].round().clip(0, 1).astype(int)
    proxy_train, proxy_validation = train_test_split(
        np.arange(len(release)),
        test_size=0.2,
        random_state=release_seed,
        stratify=release[TARGET],
    )
    proxy_model_name, proxy_model, proxy_threshold, _ = fit_selected_procedure(
        release.iloc[proxy_train][feature_names].reset_index(drop=True),
        release.iloc[proxy_train][TARGET].reset_index(drop=True),
        release.iloc[proxy_validation][feature_names].reset_index(drop=True),
        release.iloc[proxy_validation][TARGET].reset_index(drop=True),
        seed=release_seed,
    )
    return proxy_model_name, proxy_model, proxy_threshold


def run_audit(
    *,
    registry_path: Path,
    output_root: Path,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_registry_hash = (
        registry_path.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if _sha256(registry_path) != expected_registry_hash:
        raise ValueError("Audit registry hash mismatch.")
    partition = registry["partition"]
    for path_key, hash_key in (
        ("development_path", "development_sha256"),
        ("development_positions_path", "development_positions_sha256"),
        ("sealed_audit_path", "sealed_audit_sha256"),
        ("audit_positions_path", "audit_positions_sha256"),
    ):
        if _sha256(Path(partition[path_key])) != str(partition[hash_key]):
            raise ValueError(f"Partition hash mismatch for {path_key}.")

    X_development, y_development = _read_spambase(
        Path(partition["development_path"])
    )
    feature_names = _selected_features(registry)
    X_development = X_development[feature_names]
    X_train, X_validation, y_train, y_validation = _split_development(
        X_development,
        y_development,
        seed=int(registry["development"]["split_seed"]),
        validation_fraction=float(
            registry["development"]["validation_fraction"]
        ),
    )
    (
        source_model_name,
        source_model,
        source_threshold,
        _,
    ) = fit_selected_procedure(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=int(registry["development"]["split_seed"]),
    )

    release_seeds = [
        int(seed) for seed in registry["mechanism"]["release_seeds"]
    ]
    if jobs < 1:
        raise ValueError("jobs must be positive.")

    def fit_one(release_seed: int) -> tuple[int, str, object, float]:
        proxy_name, proxy_model, proxy_threshold = _fit_proxy_release(
            X_train=X_train,
            y_train=y_train,
            release_seed=release_seed,
            mechanism=registry["mechanism"],
            feature_names=feature_names,
        )
        return release_seed, proxy_name, proxy_model, proxy_threshold

    fitted_releases = Parallel(n_jobs=jobs, verbose=10)(
        delayed(fit_one)(release_seed) for release_seed in release_seeds
    )
    for release_index, _ in enumerate(fitted_releases, start=1):
        print(
            f"fitted AIM release {release_index}/{len(release_seeds)}",
            flush=True,
        )

    # This is the first parse of the sealed audit file. Every source and proxy
    # procedure above is already fixed.
    X_audit, y_audit = _read_spambase(Path(partition["sealed_audit_path"]))
    X_audit = X_audit[feature_names]
    audit_positions = pd.read_csv(partition["audit_positions_path"])[
        "RecordPosition"
    ].to_numpy(dtype=int)
    source_probability = source_model.predict_proba(X_audit)[:, 1]
    source_auc = roc_auc_score(y_audit, source_probability)
    mechanism_name = str(registry["mechanism"]["name"])
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    release_to_mechanism: dict[str, str] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    loss_frames: list[pd.DataFrame] = []

    for release_index, (
        release_seed,
        proxy_name,
        proxy_model,
        proxy_threshold,
    ) in enumerate(fitted_releases, start=1):
        candidate = f"{mechanism_name}::release_{release_index:02d}"
        proxy_probability = proxy_model.predict_proba(X_audit)[:, 1]
        losses = paired_prediction_losses(
            y_true=y_audit,
            source_probability=source_probability,
            proxy_probability=proxy_probability,
            source_thresholds={5.0: source_threshold},
            proxy_thresholds={5.0: proxy_threshold},
            record_ids=audit_positions,
        )
        losses.insert(0, "Release", release_index)
        losses.insert(0, "Candidate", candidate)
        loss_frames.append(losses)
        candidate_values[candidate] = _audit_values(losses, registry)
        release_to_mechanism[candidate] = mechanism_name
        diagnostics.append(
            {
                "Candidate": candidate,
                "Mechanism": mechanism_name,
                "Release": release_index,
                "ReleaseSeed": release_seed,
                "AuditN": len(y_audit),
                "SourceModel": source_model_name,
                "ProxyModel": proxy_name,
                "SourceAUC": source_auc,
                "ProxyAUC": roc_auc_score(y_audit, proxy_probability),
                "SourceCost5x": _mean_normalized_cost(
                    y_audit,
                    source_probability,
                    source_threshold,
                    5.0,
                ),
                "ProxyCost5x": _mean_normalized_cost(
                    y_audit,
                    proxy_probability,
                    proxy_threshold,
                    5.0,
                ),
            }
        )

    audit = audit_proxy_mechanisms(
        candidate_values,
        release_to_mechanism,
        requirements=_requirements(registry),
        minimum_reliability=float(registry["minimum_reliability"]),
        total_alpha=float(registry["total_alpha"]),
        release_error_share=float(registry["release_error_share"]),
        violation_total_alpha=float(registry["violation_total_alpha"]),
        violation_release_error_share=float(
            registry["violation_release_error_share"]
        ),
        mechanism_count_mode="holm",
        bound_method=str(registry["bound_method"]),
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    release_summary = audit.release_audit.candidate_summary.merge(
        diagnostics_frame,
        on=["Candidate", "Mechanism"],
        how="left",
        validate="one_to_one",
    )
    mechanism_summary = audit.mechanism_summary.copy()
    mechanism_summary["Dataset"] = str(registry["dataset"]["name"])
    mechanism_summary["MeanAUCChange"] = (
        diagnostics_frame["ProxyAUC"] - diagnostics_frame["SourceAUC"]
    ).mean()
    mechanism_summary["MeanCost5xChange"] = (
        diagnostics_frame["ProxyCost5x"]
        - diagnostics_frame["SourceCost5x"]
    ).mean()

    output_root.mkdir(parents=True, exist_ok=False)
    release_summary.to_csv(output_root / "release_summary.csv", index=False)
    mechanism_summary.to_csv(
        output_root / "mechanism_summary.csv",
        index=False,
    )
    audit.release_audit.requirement_detail.to_csv(
        output_root / "requirement_detail.csv",
        index=False,
    )
    pd.concat(loss_frames, ignore_index=True).to_csv(
        output_root / "paired_losses.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256": expected_registry_hash,
                "audit_opened_after_all_procedures_were_fixed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return release_summary, mechanism_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or run the sealed UCI Spambase AIM audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--partition-registry",
        default="registries/proxyguard_spambase_aim_partition.json",
    )
    prepare.add_argument(
        "--archive",
        default="data/uci_spambase/spambase.zip",
    )
    prepare.add_argument(
        "--sealed-root",
        default="data/uci_spambase/sealed_v1",
    )
    prepare.add_argument(
        "--audit-registry",
        default="registries/proxyguard_spambase_aim_audit.json",
    )
    amend = subparsers.add_parser("prepare-amendment")
    amend.add_argument(
        "--parent-registry",
        default="registries/proxyguard_spambase_aim_audit.json",
    )
    amend.add_argument(
        "--amendment",
        default="registries/proxyguard_spambase_aim_compute_amendment.json",
    )
    amend.add_argument(
        "--audit-registry",
        default="registries/proxyguard_spambase_aim_audit_v2.json",
    )
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--registry",
        default="registries/proxyguard_spambase_aim_audit.json",
    )
    audit.add_argument(
        "--output-root",
        default="outputs/proxyguard_spambase_aim",
    )
    audit.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    if args.command == "prepare":
        registry = prepare_registry(
            partition_registry_path=Path(args.partition_registry),
            archive_path=Path(args.archive),
            sealed_root=Path(args.sealed_root),
            audit_registry_path=Path(args.audit_registry),
        )
        print(json.dumps(registry, indent=2))
        return
    if args.command == "prepare-amendment":
        registry = prepare_amended_registry(
            parent_registry_path=Path(args.parent_registry),
            amendment_path=Path(args.amendment),
            output_registry_path=Path(args.audit_registry),
        )
        print(json.dumps(registry, indent=2))
        return

    _, mechanism_summary = run_audit(
        registry_path=Path(args.registry),
        output_root=Path(args.output_root),
        jobs=int(args.jobs),
    )
    print(mechanism_summary.to_string(index=False))


if __name__ == "__main__":
    main()
