from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_mechanisms,
    paired_prediction_losses,
)
from scripts.proxyguard.run_proxyguard_aim_audit import (  # noqa: E402
    _mean_normalized_cost,
    fit_selected_procedure,
)
from scripts.proxyguard.run_proxyguard_bootstrap_mechanism import (  # noqa: E402
    bootstrap_release,
)

FEATURES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]
TARGET = "class"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_magic(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, names=[*FEATURES, TARGET])
    if frame.shape[1] != 11 or frame.empty:
        raise ValueError("MAGIC data must contain ten features and one target.")
    if not set(frame[TARGET].unique()).issubset({"g", "h"}):
        raise ValueError("Unexpected MAGIC class label.")
    X = frame[FEATURES].astype(float)
    # A false negative now means accepting a hadron background event as signal,
    # the cost-sensitive error emphasized in the UCI dataset documentation.
    y = frame[TARGET].eq("h").astype(int)
    return X, y


def _development_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    positions = np.arange(len(y))
    train, validation = train_test_split(
        positions,
        test_size=0.20,
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


def _write_partition(
    lines: list[str],
    positions: np.ndarray,
    path: Path,
) -> None:
    path.write_text(
        "".join(lines[int(position)] for position in positions),
        encoding="utf-8",
    )


def prepare_registry(
    *,
    raw_path: Path,
    archive_path: Path,
    sealed_root: Path,
    registry_path: Path,
    audit_fraction: float,
    reserve_seed: int,
    development_seed: int,
    release_seed_start: int,
    releases: int,
) -> dict:
    if registry_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen registry: {registry_path}"
        )
    lines = raw_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != 19_020:
        raise ValueError(f"Expected 19,020 MAGIC rows, found {len(lines)}.")

    rng = np.random.default_rng(reserve_seed)
    audit_n = int(round(audit_fraction * len(lines)))
    audit_positions = np.sort(
        rng.choice(len(lines), size=audit_n, replace=False)
    )
    audit_set = set(audit_positions.tolist())
    development_positions = np.asarray(
        [position for position in range(len(lines)) if position not in audit_set],
        dtype=int,
    )

    sealed_root.mkdir(parents=True, exist_ok=False)
    development_path = sealed_root / "development.data"
    audit_path = sealed_root / "sealed_audit.data"
    development_positions_path = sealed_root / "development_positions.csv"
    audit_positions_path = sealed_root / "audit_positions.csv"
    _write_partition(lines, development_positions, development_path)
    _write_partition(lines, audit_positions, audit_path)
    pd.DataFrame({"RecordPosition": development_positions}).to_csv(
        development_positions_path,
        index=False,
    )
    pd.DataFrame({"RecordPosition": audit_positions}).to_csv(
        audit_positions_path,
        index=False,
    )

    X_development, y_development = _read_magic(development_path)
    X_train, X_validation, y_train, y_validation = _development_split(
        X_development,
        y_development,
        development_seed,
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
        seed=development_seed,
    )
    validation_losses = paired_prediction_losses(
        y_true=y_validation,
        source_probability=source_validation_probability,
        proxy_probability=source_validation_probability,
        source_thresholds={5.0: source_threshold},
        proxy_thresholds={5.0: source_threshold},
    )
    absolute_margin = 0.05
    absolute_limits = {
        "Proxy Brier risk": _rounded_ceiling(
            float(validation_losses["brier_proxy"].mean()),
            absolute_margin,
        ),
        "Proxy log-loss risk": _rounded_ceiling(
            float(validation_losses["logloss_proxy"].mean()),
            absolute_margin,
        ),
        "Proxy Cost5x risk": _rounded_ceiling(
            float(validation_losses["cost5x_proxy"].mean()),
            absolute_margin,
        ),
    }
    requirements: list[dict[str, float | str]] = [
        {
            "name": "Brier transfer",
            "estimand": "relative_regret",
            "value_column": "brier_regret",
            "tolerance": 0.04,
            "lower": -1.0,
            "upper": 1.0,
        },
        {
            "name": "Log-loss transfer",
            "estimand": "relative_regret",
            "value_column": "logloss_regret",
            "tolerance": 0.04,
            "lower": -1.0,
            "upper": 1.0,
        },
        {
            "name": "Cost5x transfer",
            "estimand": "relative_regret",
            "value_column": "cost5x_regret",
            "tolerance": 0.04,
            "lower": -1.0,
            "upper": 1.0,
        },
    ]
    requirements.extend(
        {
            "name": name,
            "estimand": "absolute_risk",
            "value_column": {
                "Proxy Brier risk": "brier_proxy",
                "Proxy log-loss risk": "logloss_proxy",
                "Proxy Cost5x risk": "cost5x_proxy",
            }[name],
            "tolerance": tolerance,
            "lower": 0.0,
            "upper": 1.0,
        }
        for name, tolerance in absolute_limits.items()
    )
    registry = {
        "registry_version": "1.0",
        "frozen_on": "2026-07-27",
        "analysis_status": (
            "Prospective sealed-target mechanism audit. The audit reserve was "
            "sampled uniformly without labels from a dataset not used in earlier "
            "project experiments. Requirements and release seeds were frozen "
            "before the sealed audit file was parsed."
        ),
        "dataset": {
            "name": "MAGIC Gamma Telescope",
            "uci_id": 159,
            "doi": "10.24432/C52C8B",
            "archive_path": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "raw_path": str(raw_path),
            "raw_sha256": _sha256(raw_path),
            "records": len(lines),
            "positive_class": "h",
        },
        "partition": {
            "rule": "Uniform record sample without replacement; labels not read.",
            "reserve_seed": reserve_seed,
            "audit_fraction": audit_fraction,
            "development_records": len(development_positions),
            "audit_records": len(audit_positions),
            "development_path": str(development_path),
            "development_sha256": _sha256(development_path),
            "development_positions_path": str(development_positions_path),
            "development_positions_sha256": _sha256(development_positions_path),
            "sealed_audit_path": str(audit_path),
            "sealed_audit_sha256": _sha256(audit_path),
            "audit_positions_path": str(audit_positions_path),
            "audit_positions_sha256": _sha256(audit_positions_path),
            "audit_outcomes_inspected_during_preparation": False,
        },
        "development": {
            "split_seed": development_seed,
            "validation_fraction": 0.20,
            "source_model": source_model_name,
            "source_threshold": source_threshold,
            "absolute_limit_rule": (
                "Ceiling equals source validation risk plus 0.05, rounded upward "
                "to three decimals."
            ),
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
        "mechanism": {
            "name": "Nonparametric row bootstrap",
            "description": (
                "Sample source-training rows with replacement at the original "
                "training size; each release selects its own registered pipeline "
                "on the shared development validation set."
            ),
            "releases": releases,
            "release_seeds": [
                release_seed_start + index
                for index in range(releases)
            ],
        },
        "requirements": requirements,
        "minimum_reliability": 0.8,
        "total_alpha": 0.05,
        "release_error_share": 0.5,
        "bound_method": "empirical_bernstein",
        "privacy_scope": (
            "Nonprivate high-fidelity control; no privacy or exposure claim."
        ),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    registry_hash = _sha256(registry_path)
    registry_path.with_suffix(".sha256").write_text(
        f"{registry_hash}  {registry_path.name}\n",
        encoding="utf-8",
    )
    return registry


def _requirements(registry: dict) -> list[RiskRequirement]:
    return [
        RiskRequirement(
            str(requirement["name"]),
            tolerance=float(requirement["tolerance"]),
            lower=float(requirement["lower"]),
            upper=float(requirement["upper"]),
            estimand=str(requirement["estimand"]),
        )
        for requirement in registry["requirements"]
    ]


def _audit_values(
    losses: pd.DataFrame,
    registry: dict,
) -> dict[str, np.ndarray]:
    return {
        str(requirement["name"]): losses[str(requirement["value_column"])].to_numpy()
        for requirement in registry["requirements"]
    }


def run_sealed_audit(
    *,
    registry_path: Path,
    output_root: Path,
    mechanism_count_mode: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_registry_hash = (
        registry_path.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if _sha256(registry_path) != expected_registry_hash:
        raise ValueError("Registry hash does not match the frozen SHA-256 file.")

    partition = registry["partition"]
    for path_key, hash_key in [
        ("development_path", "development_sha256"),
        ("development_positions_path", "development_positions_sha256"),
        ("sealed_audit_path", "sealed_audit_sha256"),
        ("audit_positions_path", "audit_positions_sha256"),
    ]:
        path = Path(partition[path_key])
        if _sha256(path) != str(partition[hash_key]):
            raise ValueError(f"Hash mismatch for {path}.")

    X_development, y_development = _read_magic(Path(partition["development_path"]))
    X_train, X_validation, y_train, y_validation = _development_split(
        X_development,
        y_development,
        int(registry["development"]["split_seed"]),
    )
    X_audit, y_audit = _read_magic(Path(partition["sealed_audit_path"]))
    audit_positions = pd.read_csv(partition["audit_positions_path"])[
        "RecordPosition"
    ].to_numpy(dtype=int)
    if len(X_audit) != len(audit_positions):
        raise ValueError("Audit positions do not match sealed audit rows.")

    development_seed = int(registry["development"]["split_seed"])
    (
        source_model_name,
        source_model,
        source_threshold,
        _source_validation_probability,
    ) = fit_selected_procedure(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=development_seed,
    )
    source_audit_probability = source_model.predict_proba(X_audit)[:, 1]

    mechanism_name = "magic_gamma::bootstrap"
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    release_to_mechanism: dict[str, str] = {}
    loss_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, float | int | str]] = []
    release_seeds = [int(seed) for seed in registry["mechanism"]["release_seeds"]]
    for release_index, release_seed in enumerate(release_seeds, start=1):
        candidate = f"{mechanism_name}::release_{release_index:02d}"
        bootstrap_X, bootstrap_y = bootstrap_release(
            X_train,
            y_train,
            release_seed,
        )
        (
            proxy_model_name,
            proxy_model,
            proxy_threshold,
            _proxy_validation_probability,
        ) = fit_selected_procedure(
            bootstrap_X,
            bootstrap_y,
            X_validation,
            y_validation,
            seed=release_seed,
        )
        proxy_audit_probability = proxy_model.predict_proba(X_audit)[:, 1]
        losses = paired_prediction_losses(
            y_true=y_audit,
            source_probability=source_audit_probability,
            proxy_probability=proxy_audit_probability,
            source_thresholds={5.0: source_threshold},
            proxy_thresholds={5.0: proxy_threshold},
            record_ids=audit_positions,
        )
        losses.insert(0, "Release", release_index)
        losses.insert(0, "Candidate", candidate)
        loss_frames.append(losses)
        candidate_values[candidate] = _audit_values(losses, registry)
        release_to_mechanism[candidate] = mechanism_name
        source_auc = roc_auc_score(y_audit, source_audit_probability)
        proxy_auc = roc_auc_score(y_audit, proxy_audit_probability)
        diagnostics.append(
            {
                "Candidate": candidate,
                "Mechanism": mechanism_name,
                "Release": release_index,
                "ReleaseSeed": release_seed,
                "AuditN": len(y_audit),
                "SourceModel": source_model_name,
                "ProxyModel": proxy_model_name,
                "SourceAUC": source_auc,
                "ProxyAUC": proxy_auc,
                "AUCChange": proxy_auc - source_auc,
                "SourceCost5x": _mean_normalized_cost(
                    y_audit,
                    source_audit_probability,
                    source_threshold,
                    5.0,
                ),
                "ProxyCost5x": _mean_normalized_cost(
                    y_audit,
                    proxy_audit_probability,
                    proxy_threshold,
                    5.0,
                ),
            }
        )
        print(
            f"completed sealed bootstrap release {release_index}/{len(release_seeds)}",
            flush=True,
        )

    audit = audit_proxy_mechanisms(
        candidate_values,
        release_to_mechanism,
        requirements=_requirements(registry),
        minimum_reliability=float(registry["minimum_reliability"]),
        total_alpha=float(registry["total_alpha"]),
        release_error_share=float(registry["release_error_share"]),
        mechanism_count_mode=mechanism_count_mode,
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
    mechanism_summary["MeanAUCChange"] = diagnostics_frame["AUCChange"].mean()
    mechanism_summary["MeanCost5xChange"] = (
        diagnostics_frame["ProxyCost5x"] - diagnostics_frame["SourceCost5x"]
    ).mean()

    output_root.mkdir(parents=True, exist_ok=False)
    release_summary.to_csv(output_root / "release_summary.csv", index=False)
    audit.release_audit.requirement_detail.to_csv(
        output_root / "requirement_detail.csv",
        index=False,
    )
    mechanism_summary.to_csv(output_root / "mechanism_summary.csv", index=False)
    pd.concat(loss_frames, ignore_index=True).to_csv(
        output_root / "paired_losses.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256": expected_registry_hash,
                "audit_opened_after_registry_verification": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return release_summary, audit.release_audit.requirement_detail, mechanism_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or run the sealed MAGIC Gamma mechanism audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--raw-path",
        default="data/uci_magic_gamma/magic04.data",
    )
    prepare.add_argument(
        "--archive-path",
        default="data/uci_magic_gamma/magic_gamma.zip",
    )
    prepare.add_argument(
        "--sealed-root",
        default="data/uci_magic_gamma/sealed_v1",
    )
    prepare.add_argument(
        "--registry",
        default="registries/proxyguard_magic_sealed_registry.json",
    )
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--registry",
        default="registries/proxyguard_magic_sealed_registry.json",
    )
    audit.add_argument(
        "--output-root",
        default="outputs/proxyguard_magic_sealed_mechanism",
    )
    audit.add_argument(
        "--mechanism-count-mode",
        choices=("holm", "simes"),
        default="holm",
        help="Aggregate release evidence by release-level Holm certification or Simes count.",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        registry = prepare_registry(
            raw_path=Path(args.raw_path),
            archive_path=Path(args.archive_path),
            sealed_root=Path(args.sealed_root),
            registry_path=Path(args.registry),
            audit_fraction=0.30,
            reserve_seed=27_072_701,
            development_seed=27_072_702,
            release_seed_start=127_001,
            releases=30,
        )
        print(
            json.dumps(
                {
                    "registry": args.registry,
                    "registry_sha256": _sha256(Path(args.registry)),
                    "development_records": registry["partition"][
                        "development_records"
                    ],
                    "sealed_audit_records": registry["partition"]["audit_records"],
                    "audit_outcomes_inspected": False,
                },
                indent=2,
            )
        )
        return

    _, _, mechanism_summary = run_sealed_audit(
        registry_path=Path(args.registry),
        output_root=Path(args.output_root),
        mechanism_count_mode=args.mechanism_count_mode,
    )
    print(
        mechanism_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
