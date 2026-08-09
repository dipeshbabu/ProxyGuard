from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from proxyguard.core import (
    RiskRequirement,
    audit_proxy_mechanisms,
    paired_prediction_losses,
)
from proxyguard.shared_target import shared_target_conditional_mean_lower_bound
from scripts.proxyguard.run_proxyguard_aim_audit import choose_cost_threshold


TARGET = "class"
FALSE_NEGATIVE_COST = 5.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_json(path: Path) -> dict:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_csv_bytes(registry: dict) -> bytes:
    dataset = registry["dataset"]
    archive_path = Path(dataset["archive_path"])
    if _sha256(archive_path) != str(dataset["archive_sha256"]):
        raise ValueError("Official UCI archive hash mismatch.")
    with zipfile.ZipFile(archive_path) as outer:
        nested_bytes = outer.read(str(dataset["outer_member"]))
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
        return nested.read(str(dataset["data_member"]))


def prepare_partition(registry_path: Path) -> dict:
    registry = _verified_json(registry_path)
    raw = _nested_csv_bytes(registry)
    lines = raw.splitlines(keepends=True)
    expected_records = int(registry["dataset"]["expected_records"])
    if len(lines) != expected_records + 1:
        raise ValueError(
            f"Expected one header and {expected_records} rows, found {len(lines)} lines."
        )

    partition = registry["partition"]
    sealed_root = Path(partition["sealed_root"])
    if sealed_root.exists():
        raise FileExistsError(f"Refusing to overwrite sealed partition: {sealed_root}")
    sealed_root.mkdir(parents=True)

    rng = np.random.default_rng(int(partition["reserve_seed"]))
    audit_positions = np.sort(
        rng.choice(
            expected_records,
            size=int(partition["sealed_audit_rows"]),
            replace=False,
        )
    )
    audit_set = set(audit_positions.tolist())
    development_positions = np.asarray(
        [position for position in range(expected_records) if position not in audit_set],
        dtype=int,
    )
    header = lines[0]

    development_path = sealed_root / "development.csv"
    audit_path = sealed_root / "sealed_audit.csv"
    development_positions_path = sealed_root / "development_positions.csv"
    audit_positions_path = sealed_root / "audit_positions.csv"
    development_path.write_bytes(
        header + b"".join(lines[position + 1] for position in development_positions)
    )
    audit_path.write_bytes(
        header + b"".join(lines[position + 1] for position in audit_positions)
    )
    pd.DataFrame({"RecordPosition": development_positions}).to_csv(
        development_positions_path,
        index=False,
    )
    pd.DataFrame({"RecordPosition": audit_positions}).to_csv(
        audit_positions_path,
        index=False,
    )

    manifest = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "development_records": int(development_positions.size),
        "audit_records": int(audit_positions.size),
        "development_path": str(development_path),
        "development_sha256": _sha256(development_path),
        "sealed_audit_path": str(audit_path),
        "sealed_audit_sha256": _sha256(audit_path),
        "development_positions_path": str(development_positions_path),
        "development_positions_sha256": _sha256(development_positions_path),
        "audit_positions_path": str(audit_positions_path),
        "audit_positions_sha256": _sha256(audit_positions_path),
        "outcomes_parsed_during_partition": False,
    }
    manifest_path = sealed_root / "partition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (sealed_root / "partition_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    return manifest


def prepare_domain_amendment(
    parent_registry_path: Path,
    amendment_path: Path,
    output_registry_path: Path,
) -> dict:
    if output_registry_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite amended registry: {output_registry_path}"
        )
    parent = _verified_json(parent_registry_path)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    expected_amendment_hash = amendment_path.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256(amendment_path) != expected_amendment_hash:
        raise ValueError("Domain amendment hash mismatch.")
    expected_parent_hash = str(amendment["parent_registry_sha256"])
    if _sha256(parent_registry_path) != expected_parent_hash:
        raise ValueError("Domain amendment names a different parent registry.")
    if bool(amendment["sealed_audit_opened"]):
        raise ValueError("A domain amendment cannot follow target access.")

    revised = json.loads(json.dumps(parent))
    revised["registry_version"] = "1.1"
    revised["analysis_status"] = (
        "Prospective amended partition and development plan. A fixed unknown-"
        "category map was registered after a development-only parsing error and "
        "before the sealed reserve opened."
    )
    revised["parent_registry"] = str(parent_registry_path)
    revised["parent_registry_sha256"] = expected_parent_hash
    revised["domain_amendment"] = str(amendment_path)
    revised["domain_amendment_sha256"] = expected_amendment_hash
    revised["dataset"]["unknown_category_policy"] = str(
        amendment["change"]["unknown_category_policy"]
    )
    output_registry_path.write_text(
        json.dumps(revised, indent=2) + "\n",
        encoding="utf-8",
    )
    revised_hash = _sha256(output_registry_path)
    output_registry_path.with_suffix(".sha256").write_text(
        f"{revised_hash}  {output_registry_path.name}\n",
        encoding="utf-8",
    )
    return revised


def prepare_mechanism_amendment(
    parent_registry_path: Path,
    amendment_path: Path,
    output_registry_path: Path,
) -> dict:
    """Freeze a replacement mechanism after a development-only compute failure."""
    if output_registry_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite amended registry: {output_registry_path}"
        )
    parent = _verified_json(parent_registry_path)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    expected_amendment_hash = amendment_path.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256(amendment_path) != expected_amendment_hash:
        raise ValueError("Mechanism amendment hash mismatch.")
    expected_parent_hash = str(amendment["parent_registry_sha256"])
    if _sha256(parent_registry_path) != expected_parent_hash:
        raise ValueError("Mechanism amendment names a different parent registry.")
    if bool(amendment["sealed_audit_opened"]):
        raise ValueError("A mechanism amendment cannot follow target access.")

    revised = json.loads(json.dumps(parent))
    revised["registry_version"] = "1.2"
    revised["analysis_status"] = (
        "Prospective amended development plan. The registered AIM pilot produced "
        "no complete artifact in two infrastructure-limited attempts. Before the "
        "sealed reserve opened, it was replaced by the fully specified private "
        "class-conditional categorical mechanism recorded in the amendment."
    )
    revised["mechanism_parent_registry"] = str(parent_registry_path)
    revised["mechanism_parent_registry_sha256"] = expected_parent_hash
    revised["mechanism_amendment"] = str(amendment_path)
    revised["mechanism_amendment_sha256"] = expected_amendment_hash
    replacement = {**parent["mechanism"], **amendment["mechanism_changes"]}
    for key in amendment["removed_mechanism_keys"]:
        replacement.pop(str(key), None)
    revised["mechanism"] = replacement
    output_registry_path.write_text(
        json.dumps(revised, indent=2) + "\n",
        encoding="utf-8",
    )
    revised_hash = _sha256(output_registry_path)
    output_registry_path.with_suffix(".sha256").write_text(
        f"{revised_hash}  {output_registry_path.name}\n",
        encoding="utf-8",
    )
    return revised


def _encode_frame(path: Path, registry: dict) -> tuple[pd.DataFrame, pd.Series]:
    dataset = registry["dataset"]
    domains: dict[str, list[str]] = dataset["feature_domains"]
    frame = pd.read_csv(
        path,
        sep=str(dataset["delimiter"]),
        dtype=str,
        keep_default_na=False,
    )
    expected_columns = list(domains)
    missing = set(expected_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing registered columns: {sorted(missing)}")

    encoded = pd.DataFrame(index=frame.index)
    for column, labels in domains.items():
        mapping = {label: index for index, label in enumerate(labels)}
        values = frame[column].astype(str).str.strip().replace("", "?")
        unknown = sorted(set(values.unique()) - set(mapping))
        if unknown:
            policy = str(dataset.get("unknown_category_policy", "reject"))
            if policy != "map_to_registered_unknown":
                raise ValueError(f"Unregistered values in {column}: {unknown}")
            if "?" not in mapping:
                raise ValueError(
                    f"Cannot map unregistered values in {column} without a '?' domain."
                )
            values = values.where(values.isin(mapping), "?")
        encoded[column] = values.map(mapping).astype(int)
    y = encoded.pop(TARGET).astype(int)
    return encoded, y


def _development_split(
    X: pd.DataFrame,
    y: pd.Series,
    registry: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    partition = registry["partition"]
    train_positions, validation_positions = train_test_split(
        np.arange(len(y)),
        test_size=float(partition["development_validation_fraction"]),
        random_state=int(partition["development_seed"]),
        stratify=y,
    )
    return (
        X.iloc[train_positions].reset_index(drop=True),
        X.iloc[validation_positions].reset_index(drop=True),
        y.iloc[train_positions].reset_index(drop=True),
        y.iloc[validation_positions].reset_index(drop=True),
    )


def _feature_names(registry: dict) -> list[str]:
    return [name for name in registry["dataset"]["feature_domains"] if name != TARGET]


def _build_classifier(registry: dict, seed: int) -> Pipeline:
    features = _feature_names(registry)
    domains = registry["dataset"]["feature_domains"]
    categories = [np.arange(len(domains[feature]), dtype=int) for feature in features]
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(categories=categories, handle_unknown="ignore"), features)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("encode", encoder),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    registry: dict,
    seed: int,
) -> tuple[Pipeline, float, np.ndarray]:
    model = _build_classifier(registry, seed)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_validation)[:, 1]
    threshold = choose_cost_threshold(y_validation, probability, FALSE_NEGATIVE_COST)
    return model, threshold, probability


def _aim_constraints(registry: dict) -> dict[str, Any]:
    try:
        from snsynth.transform import BinTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SmartNoise Synth is required. Run with `uv run --with smartnoise-synth`."
        ) from exc

    constraints: dict[str, Any] = {}
    for column, labels in registry["dataset"]["feature_domains"].items():
        constraints[column] = BinTransformer(
            bins=len(labels),
            lower=-0.5,
            upper=len(labels) - 0.5,
            nullable=False,
        )
    return constraints


def _fit_aim_generator(
    train_table: pd.DataFrame,
    registry: dict,
    seed: int,
) -> Any:
    try:
        from snsynth import Synthesizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SmartNoise Synth is required. Run with `uv run --with smartnoise-synth`."
        ) from exc

    mechanism = registry["mechanism"]
    np.random.seed(seed)
    synthesizer = Synthesizer.create(
        "aim",
        epsilon=float(mechanism["epsilon"]),
        delta=float(mechanism["delta"]),
        degree=int(mechanism["degree"]),
        max_model_size=int(mechanism["max_model_size"]),
        num_marginals=int(mechanism["num_marginals"]),
        max_cells=int(mechanism["max_cells"]),
    )
    synthesizer.fit(train_table, transformer=_aim_constraints(registry))
    return synthesizer


def _fit_dp_conditional_generator(
    train_table: pd.DataFrame,
    registry: dict,
    seed: int,
) -> dict[str, Any]:
    """Fit a pure-DP class-conditional categorical generator.

    The released vector contains the class histogram and one feature-by-class
    contingency table per feature. One record changes one cell in each of these
    tables, so under add-or-remove-one adjacency the vector query has L1
    sensitivity ``1 + p`` for ``p`` features.
    Independent Laplace noise at that sensitivity divided by epsilon therefore
    gives the registered pure-DP fit; all subsequent sampling is post-processing.
    """
    mechanism = registry["mechanism"]
    features = _feature_names(registry)
    domains = registry["dataset"]["feature_domains"]
    epsilon = float(mechanism["epsilon"])
    sensitivity = 1 + len(features)
    if int(mechanism["vector_l1_sensitivity"]) != sensitivity:
        raise ValueError("Registered vector sensitivity does not match the query.")
    scale = sensitivity / epsilon
    pseudocount = float(mechanism["post_noise_pseudocount"])
    rng = np.random.default_rng(seed)

    class_count = np.bincount(
        train_table[TARGET].to_numpy(dtype=int),
        minlength=len(domains[TARGET]),
    ).astype(float)
    noisy_class = np.maximum(class_count + rng.laplace(0.0, scale, class_count.size), 0.0)
    class_probability = (noisy_class + pseudocount) / (
        noisy_class.sum() + pseudocount * noisy_class.size
    )

    conditional: dict[str, np.ndarray] = {}
    y = train_table[TARGET].to_numpy(dtype=int)
    for feature in features:
        counts = np.zeros((len(domains[TARGET]), len(domains[feature])), dtype=float)
        np.add.at(
            counts,
            (y, train_table[feature].to_numpy(dtype=int)),
            1.0,
        )
        noisy = np.maximum(counts + rng.laplace(0.0, scale, counts.shape), 0.0)
        noisy += pseudocount
        conditional[feature] = noisy / noisy.sum(axis=1, keepdims=True)

    return {
        "class_probability": class_probability,
        "conditional_probability": conditional,
        "laplace_scale": scale,
    }


def _fit_private_generator(
    train_table: pd.DataFrame,
    registry: dict,
    seed: int,
) -> Any:
    generator_type = str(registry["mechanism"].get("generator_type", "aim"))
    if generator_type == "aim":
        return _fit_aim_generator(train_table, registry, seed)
    if generator_type == "dp_class_conditional_categorical":
        return _fit_dp_conditional_generator(train_table, registry, seed)
    raise ValueError(f"Unsupported generator type: {generator_type}")


def _sample_aim_generator(
    synthesizer: Any,
    registry: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    mechanism = registry["mechanism"]
    columns = [*_feature_names(registry), TARGET]
    np.random.seed(seed)
    release = pd.DataFrame(
        synthesizer.sample(int(mechanism["synthetic_records"])),
        columns=columns,
    )
    domains = registry["dataset"]["feature_domains"]
    for column in columns:
        release[column] = (
            pd.to_numeric(release[column], errors="raise")
            .round()
            .clip(0, len(domains[column]) - 1)
            .astype(int)
        )
    y = release.pop(TARGET).astype(int)
    return release, y


def _sample_dp_conditional_generator(
    synthesizer: dict[str, Any],
    registry: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    mechanism = registry["mechanism"]
    domains = registry["dataset"]["feature_domains"]
    features = _feature_names(registry)
    records = int(mechanism["synthetic_records"])
    rng = np.random.default_rng(seed)
    y = rng.choice(
        len(domains[TARGET]),
        size=records,
        p=np.asarray(synthesizer["class_probability"], dtype=float),
    )
    columns: dict[str, np.ndarray] = {}
    for feature in features:
        values = np.empty(records, dtype=int)
        probabilities = np.asarray(
            synthesizer["conditional_probability"][feature],
            dtype=float,
        )
        for class_value in range(len(domains[TARGET])):
            mask = y == class_value
            values[mask] = rng.choice(
                len(domains[feature]),
                size=int(mask.sum()),
                p=probabilities[class_value],
            )
        columns[feature] = values
    return pd.DataFrame(columns), pd.Series(y, name=TARGET)


def _sample_private_generator(
    synthesizer: Any,
    registry: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    generator_type = str(registry["mechanism"].get("generator_type", "aim"))
    if generator_type == "aim":
        return _sample_aim_generator(synthesizer, registry, seed)
    if generator_type == "dp_class_conditional_categorical":
        return _sample_dp_conditional_generator(synthesizer, registry, seed)
    raise ValueError(f"Unsupported generator type: {generator_type}")


def _fit_release_procedure(
    release: pd.DataFrame,
    y_release: pd.Series,
    registry: dict,
    seed: int,
) -> tuple[Pipeline, float]:
    if y_release.nunique() != 2:
        raise RuntimeError("A synthetic release contains only one target class.")
    train_positions, validation_positions = train_test_split(
        np.arange(len(y_release)),
        test_size=0.2,
        random_state=seed,
        stratify=y_release,
    )
    model, threshold, _ = _fit_classifier(
        release.iloc[train_positions].reset_index(drop=True),
        y_release.iloc[train_positions].reset_index(drop=True),
        release.iloc[validation_positions].reset_index(drop=True),
        y_release.iloc[validation_positions].reset_index(drop=True),
        registry,
        seed,
    )
    return model, threshold


def _paired_losses(
    y: pd.Series,
    source_probability: np.ndarray,
    proxy_probability: np.ndarray,
    source_threshold: float,
    proxy_threshold: float,
    record_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    return paired_prediction_losses(
        y_true=y,
        source_probability=source_probability,
        proxy_probability=proxy_probability,
        source_thresholds={FALSE_NEGATIVE_COST: source_threshold},
        proxy_thresholds={FALSE_NEGATIVE_COST: proxy_threshold},
        record_ids=record_ids,
    )


def _manifest(registry: dict) -> tuple[dict, Path]:
    sealed_root = Path(registry["partition"]["sealed_root"])
    manifest_path = sealed_root / "partition_manifest.json"
    expected = (sealed_root / "partition_manifest.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256(manifest_path) != expected:
        raise ValueError("Partition manifest hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("development_path", "development_sha256"),
        ("sealed_audit_path", "sealed_audit_sha256"),
        ("development_positions_path", "development_positions_sha256"),
        ("audit_positions_path", "audit_positions_sha256"),
    ):
        if _sha256(Path(manifest[path_key])) != str(manifest[hash_key]):
            raise ValueError(f"Partition file hash mismatch for {path_key}.")
    return manifest, manifest_path


def run_pilot(registry_path: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    registry = _verified_json(registry_path)
    manifest, manifest_path = _manifest(registry)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output_root}")

    X_development, y_development = _encode_frame(
        Path(manifest["development_path"]),
        registry,
    )
    X_train, X_validation, y_train, y_validation = _development_split(
        X_development,
        y_development,
        registry,
    )
    source_model, source_threshold, source_probability = _fit_classifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        registry,
        int(registry["partition"]["development_seed"]),
    )
    source_losses = _paired_losses(
        y_validation,
        source_probability,
        source_probability,
        source_threshold,
        source_threshold,
    )

    train_table = X_train.copy()
    train_table[TARGET] = y_train.to_numpy()
    fit_started = time.perf_counter()
    synthesizer = _fit_private_generator(
        train_table,
        registry,
        int(registry["mechanism"]["pilot_fit_seed"]),
    )
    fit_seconds = time.perf_counter() - fit_started

    rows: list[dict[str, float | int]] = []
    release_started = time.perf_counter()
    for release_index, seed in enumerate(
        registry["mechanism"]["pilot_release_seeds"],
        start=1,
    ):
        release, y_release = _sample_private_generator(
            synthesizer,
            registry,
            int(seed),
        )
        proxy_model, proxy_threshold = _fit_release_procedure(
            release,
            y_release,
            registry,
            int(seed),
        )
        proxy_probability = proxy_model.predict_proba(X_validation)[:, 1]
        losses = _paired_losses(
            y_validation,
            source_probability,
            proxy_probability,
            source_threshold,
            proxy_threshold,
        )
        row: dict[str, float | int] = {
            "Release": release_index,
            "ReleaseSeed": int(seed),
            "ProxyAUC": roc_auc_score(y_validation, proxy_probability),
        }
        for requirement in registry["claims"]["requirements"]:
            column = str(requirement["value_column"])
            row[column] = float(losses[column].mean())
        rows.append(row)
    release_seconds = time.perf_counter() - release_started
    pilot = pd.DataFrame(rows)

    output_root.mkdir(parents=True)
    pilot_path = output_root / "pilot_release_means.csv"
    pilot.to_csv(pilot_path, index=False)
    summary = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "partition_manifest": str(manifest_path),
        "partition_manifest_sha256": _sha256(manifest_path),
        "pilot_release_means": str(pilot_path),
        "pilot_release_means_sha256": _sha256(pilot_path),
        "sealed_audit_opened": False,
        "development_records": len(X_development),
        "training_records": len(X_train),
        "validation_records": len(X_validation),
        "source_auc": roc_auc_score(y_validation, source_probability),
        "source_threshold": source_threshold,
        "source_brier": float(source_losses["brier_proxy"].mean()),
        "source_cost5x": float(source_losses["cost5x_proxy"].mean()),
        "generator_fit_seconds": fit_seconds,
        "pilot_sampling_and_training_seconds": release_seconds,
    }
    summary_path = output_root / "pilot_settings.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return pilot, summary


def _round_up(value: float, decimals: int = 3) -> float:
    scale = 10**decimals
    return math.ceil(scale * value - 1e-12) / scale


def freeze_audit_registry(
    partition_registry_path: Path,
    pilot_root: Path,
    audit_registry_path: Path,
) -> dict:
    if audit_registry_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen registry: {audit_registry_path}")
    partition_registry = _verified_json(partition_registry_path)
    manifest, manifest_path = _manifest(partition_registry)
    pilot_settings_path = pilot_root / "pilot_settings.json"
    pilot_means_path = pilot_root / "pilot_release_means.csv"
    pilot_settings = json.loads(pilot_settings_path.read_text(encoding="utf-8"))
    if _sha256(pilot_means_path) != str(pilot_settings["pilot_release_means_sha256"]):
        raise ValueError("Pilot means hash mismatch.")
    if _sha256(partition_registry_path) != str(
        pilot_settings["partition_registry_sha256"]
    ):
        raise ValueError("Pilot used a different partition registry.")
    pilot = pd.read_csv(pilot_means_path)

    requirements: list[dict[str, float | str]] = []
    for registered in partition_registry["claims"]["requirements"]:
        column = str(registered["value_column"])
        cutoff = _round_up(
            float(pilot[column].max()) + float(registered["pilot_guard"])
        )
        tolerance = _round_up(cutoff + float(registered["slack"]))
        lower = float(registered["lower"])
        upper = float(registered["upper"])
        if not lower < cutoff < tolerance < upper:
            raise ValueError(f"Derived limits for {registered['name']} leave the bounded domain.")
        requirements.append(
            {
                **registered,
                "score_cutoff": cutoff,
                "tolerance": tolerance,
                "pilot_maximum": float(pilot[column].max()),
            }
        )

    registry = {
        "registry_version": "1.0",
        "frozen_on": "2026-08-02",
        "analysis_status": (
            "Prospective direct shared-target audit. The private generator, "
            "release seeds, requirements, and error allocations were frozen "
            "after development-only planning and before the sealed reserve opened."
        ),
        "partition_registry": str(partition_registry_path),
        "partition_registry_sha256": _sha256(partition_registry_path),
        "partition_manifest": str(manifest_path),
        "partition_manifest_sha256": _sha256(manifest_path),
        "pilot_settings": str(pilot_settings_path),
        "pilot_settings_sha256": _sha256(pilot_settings_path),
        "pilot_release_means": str(pilot_means_path),
        "pilot_release_means_sha256": _sha256(pilot_means_path),
        "dataset": partition_registry["dataset"],
        "partition": {
            **partition_registry["partition"],
            **manifest,
        },
        "development": {
            "source_model": "Fixed one-hot logistic regression",
            "source_threshold": float(pilot_settings["source_threshold"]),
            "source_validation_auc": float(pilot_settings["source_auc"]),
            "source_validation_brier": float(pilot_settings["source_brier"]),
            "source_validation_cost5x": float(pilot_settings["source_cost5x"]),
            "limit_rule": partition_registry["claims"]["limit_rule"],
        },
        "mechanism": partition_registry["mechanism"],
        "requirements": requirements,
        "minimum_reliability": float(
            partition_registry["claims"]["minimum_reliability"]
        ),
        "total_alpha": float(partition_registry["claims"]["total_alpha"]),
        "named_release_error_share": float(
            partition_registry["claims"]["named_release_error_share"]
        ),
        "direct_target_error_fraction": float(
            partition_registry["claims"]["direct_target_error_fraction"]
        ),
        "decision_rule": partition_registry["claims"]["decision_rule"],
        "sealed_audit_opened_before_freeze": False,
    }
    audit_registry_path.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )
    registry_hash = _sha256(audit_registry_path)
    audit_registry_path.with_suffix(".sha256").write_text(
        f"{registry_hash}  {audit_registry_path.name}\n",
        encoding="utf-8",
    )
    return registry


def _risk_requirements(registry: dict) -> list[RiskRequirement]:
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


def run_audit(registry_path: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = _verified_json(registry_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_root}")
    for path_key, hash_key in (
        ("development_path", "development_sha256"),
        ("sealed_audit_path", "sealed_audit_sha256"),
        ("development_positions_path", "development_positions_sha256"),
        ("audit_positions_path", "audit_positions_sha256"),
    ):
        if _sha256(Path(registry["partition"][path_key])) != str(
            registry["partition"][hash_key]
        ):
            raise ValueError(f"Partition hash mismatch for {path_key}.")

    X_development, y_development = _encode_frame(
        Path(registry["partition"]["development_path"]),
        registry,
    )
    X_train, X_validation, y_train, y_validation = _development_split(
        X_development,
        y_development,
        registry,
    )
    source_model, source_threshold, _ = _fit_classifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        registry,
        int(registry["partition"]["development_seed"]),
    )

    train_table = X_train.copy()
    train_table[TARGET] = y_train.to_numpy()
    fit_started = time.perf_counter()
    synthesizer = _fit_private_generator(
        train_table,
        registry,
        int(registry["mechanism"]["final_fit_seed"]),
    )
    fit_seconds = time.perf_counter() - fit_started

    fitted_releases: list[tuple[int, Pipeline, float]] = []
    release_started = time.perf_counter()
    for release_index, seed in enumerate(
        registry["mechanism"]["release_seeds"],
        start=1,
    ):
        release, y_release = _sample_private_generator(
            synthesizer,
            registry,
            int(seed),
        )
        model, threshold = _fit_release_procedure(
            release,
            y_release,
            registry,
            int(seed),
        )
        fitted_releases.append((int(seed), model, threshold))
        if release_index % 10 == 0:
            print(f"fitted release procedure {release_index}/{len(registry['mechanism']['release_seeds'])}", flush=True)
    release_seconds = time.perf_counter() - release_started

    # First parse of the sealed reserve: every generator and downstream
    # procedure above is already fixed.
    X_audit, y_audit = _encode_frame(
        Path(registry["partition"]["sealed_audit_path"]),
        registry,
    )
    audit_positions = pd.read_csv(registry["partition"]["audit_positions_path"])[
        "RecordPosition"
    ].to_numpy(dtype=int)
    source_probability = source_model.predict_proba(X_audit)[:, 1]
    source_auc = roc_auc_score(y_audit, source_probability)

    requirements = registry["requirements"]
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    release_to_mechanism: dict[str, str] = {}
    release_means = np.empty((len(fitted_releases), len(requirements)), dtype=float)
    diagnostics: list[dict[str, float | int | str]] = []
    loss_frames: list[pd.DataFrame] = []
    mechanism_name = str(registry["mechanism"]["name"])
    for release_index, (seed, model, threshold) in enumerate(fitted_releases, start=1):
        candidate = f"{mechanism_name}::release_{release_index:03d}"
        proxy_probability = model.predict_proba(X_audit)[:, 1]
        losses = _paired_losses(
            y_audit,
            source_probability,
            proxy_probability,
            source_threshold,
            threshold,
            record_ids=audit_positions,
        )
        losses.insert(0, "Release", release_index)
        losses.insert(0, "Candidate", candidate)
        loss_frames.append(losses)
        values: dict[str, np.ndarray] = {}
        for requirement_index, requirement in enumerate(requirements):
            name = str(requirement["name"])
            column = str(requirement["value_column"])
            values[name] = losses[column].to_numpy()
            release_means[release_index - 1, requirement_index] = float(
                losses[column].mean()
            )
        candidate_values[candidate] = values
        release_to_mechanism[candidate] = mechanism_name
        diagnostics.append(
            {
                "Candidate": candidate,
                "Mechanism": mechanism_name,
                "Release": release_index,
                "ReleaseSeed": seed,
                "ProxyAUC": roc_auc_score(y_audit, proxy_probability),
                "SourceAUC": source_auc,
            }
        )

    named_audit = audit_proxy_mechanisms(
        candidate_values,
        release_to_mechanism,
        requirements=_risk_requirements(registry),
        minimum_reliability=float(registry["minimum_reliability"]),
        total_alpha=float(registry["total_alpha"]),
        release_error_share=float(registry["named_release_error_share"]),
        violation_total_alpha=float(registry["total_alpha"]),
        violation_release_error_share=float(registry["named_release_error_share"]),
        mechanism_count_mode="holm",
        bound_method="empirical_bernstein",
    )

    direct = shared_target_conditional_mean_lower_bound(
        release_means,
        target_records=len(y_audit),
        tolerances=[float(requirement["tolerance"]) for requirement in requirements],
        slacks=[float(requirement["slack"]) for requirement in requirements],
        lower_bounds=[float(requirement["lower"]) for requirement in requirements],
        upper_bounds=[float(requirement["upper"]) for requirement in requirements],
        error_rate=float(registry["total_alpha"]),
        target_error_fraction=float(registry["direct_target_error_fraction"]),
        mechanisms=1,
    )
    score_cutoffs = np.asarray(
        [float(requirement["score_cutoff"]) for requirement in requirements]
    )
    scores = np.all(release_means <= score_cutoffs.reshape(1, -1), axis=1)
    direct_summary = pd.DataFrame(
        [
            {
                "Mechanism": mechanism_name,
                "Dataset": str(registry["dataset"]["name"]),
                "Releases": len(fitted_releases),
                "AuditN": len(y_audit),
                "FavorableScores": int(scores.sum()),
                "ConditionalScoreMean": direct.conditional_score_mean,
                "ConditionalScoreLowerBound": direct.conditional_score_lower_bound,
                "InvalidReleaseScoreCeiling": direct.invalid_release_score_ceiling,
                "TargetContaminationAllowance": direct.target_contamination_allowance,
                "DirectReliabilityLowerBound": direct.reliability_lower_bound,
                "MinimumReliability": float(registry["minimum_reliability"]),
                "DirectDecision": (
                    "Validated"
                    if direct.reliability_lower_bound
                    > float(registry["minimum_reliability"])
                    else "Unresolved"
                ),
                "SourceAUC": source_auc,
                "MeanProxyAUC": float(
                    np.mean([row["ProxyAUC"] for row in diagnostics])
                ),
                "GeneratorFitSeconds": fit_seconds,
                "SamplingAndProcedureFitSeconds": release_seconds,
            }
        ]
    )

    output_root.mkdir(parents=True)
    release_summary = named_audit.release_audit.candidate_summary.merge(
        pd.DataFrame(diagnostics),
        on=["Candidate", "Mechanism"],
        how="left",
        validate="one_to_one",
    )
    release_summary.to_csv(output_root / "release_summary.csv", index=False)
    named_audit.mechanism_summary.to_csv(
        output_root / "named_mechanism_summary.csv",
        index=False,
    )
    named_audit.release_audit.requirement_detail.to_csv(
        output_root / "requirement_detail.csv",
        index=False,
    )
    pd.DataFrame(
        release_means,
        columns=[str(requirement["name"]) for requirement in requirements],
    ).assign(Release=np.arange(1, len(fitted_releases) + 1)).to_csv(
        output_root / "release_means.csv",
        index=False,
    )
    direct_summary.to_csv(output_root / "direct_summary.csv", index=False)
    pd.concat(loss_frames, ignore_index=True).to_csv(
        output_root / "paired_losses.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256": _sha256(registry_path),
                "sealed_audit_opened_after_all_procedures_were_fixed": True,
                "privacy_scope": registry["mechanism"]["privacy_claim"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return release_summary, direct_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed Secondary Mushroom direct shared-target audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--partition-registry",
        default="registries/proxyguard_secondary_mushroom_partition.json",
    )

    amend = subparsers.add_parser("amend-domain")
    amend.add_argument(
        "--parent-registry",
        default="registries/proxyguard_secondary_mushroom_partition.json",
    )
    amend.add_argument(
        "--amendment",
        default="registries/proxyguard_secondary_mushroom_domain_amendment.json",
    )
    amend.add_argument(
        "--output-registry",
        default="registries/proxyguard_secondary_mushroom_partition_v2.json",
    )

    amend_mechanism = subparsers.add_parser("amend-mechanism")
    amend_mechanism.add_argument(
        "--parent-registry",
        default="registries/proxyguard_secondary_mushroom_partition_v2.json",
    )
    amend_mechanism.add_argument(
        "--amendment",
        default="registries/proxyguard_secondary_mushroom_mechanism_amendment.json",
    )
    amend_mechanism.add_argument(
        "--output-registry",
        default="registries/proxyguard_secondary_mushroom_partition_v3.json",
    )

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument(
        "--partition-registry",
        default="registries/proxyguard_secondary_mushroom_partition.json",
    )
    pilot.add_argument(
        "--output-root",
        default="outputs/proxyguard_secondary_mushroom_pilot",
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument(
        "--partition-registry",
        default="registries/proxyguard_secondary_mushroom_partition.json",
    )
    freeze.add_argument(
        "--pilot-root",
        default="outputs/proxyguard_secondary_mushroom_pilot",
    )
    freeze.add_argument(
        "--audit-registry",
        default="registries/proxyguard_secondary_mushroom_audit.json",
    )

    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--audit-registry",
        default="registries/proxyguard_secondary_mushroom_audit.json",
    )
    audit.add_argument(
        "--output-root",
        default="outputs/proxyguard_secondary_mushroom_direct",
    )

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_partition(Path(args.partition_registry))
        print(json.dumps(result, indent=2))
    elif args.command == "amend-domain":
        result = prepare_domain_amendment(
            Path(args.parent_registry),
            Path(args.amendment),
            Path(args.output_registry),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "amend-mechanism":
        result = prepare_mechanism_amendment(
            Path(args.parent_registry),
            Path(args.amendment),
            Path(args.output_registry),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "pilot":
        pilot_frame, settings = run_pilot(
            Path(args.partition_registry),
            Path(args.output_root),
        )
        print(pilot_frame.to_string(index=False))
        print(json.dumps(settings, indent=2))
    elif args.command == "freeze":
        result = freeze_audit_registry(
            Path(args.partition_registry),
            Path(args.pilot_root),
            Path(args.audit_registry),
        )
        print(json.dumps(result, indent=2))
    else:
        release_summary, direct_summary = run_audit(
            Path(args.audit_registry),
            Path(args.output_root),
        )
        print(release_summary.to_string(index=False))
        print(direct_summary.to_string(index=False))


if __name__ == "__main__":
    main()
