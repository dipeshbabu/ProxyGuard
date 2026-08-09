"""Sealed full-pipeline neural shared-target audit implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proxyguard.core import (
    RiskRequirement,
    audit_proxy_mechanisms,
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
    holm_adjust,
)
from proxyguard.shared_target import (
    plan_conditional_shared_target,
    shared_target_conditional_mean_lower_bound,
    stratified_release_evidence,
    stratified_shared_target_conditional_witness_lower_bound,
)


LOSS_NAMES = ("classification_error", "brier", "normalized_cost5x")


def _binary_target(frame: pd.DataFrame, registry: dict[str, Any]) -> np.ndarray:
    dataset = registry["dataset"]
    values = frame[str(dataset["target"])].to_numpy()
    if "positive_label" in dataset:
        return (values == dataset["positive_label"]).astype(int)
    return values.astype(int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing registry digest: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"Registry digest mismatch for {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _feature_columns(frame: pd.DataFrame, registry: dict[str, Any]) -> list[str]:
    excluded = {
        str(registry["dataset"]["target"]),
        str(registry["dataset"]["id_column"]),
    }
    return [column for column in frame.columns if column not in excluded]


def _split_xy(
    frame: pd.DataFrame,
    registry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = _feature_columns(frame, registry)
    encoded = pd.DataFrame(index=frame.index)
    registered_domains = registry["dataset"].get("categorical_domains", {})
    for feature in features:
        series = frame[feature]
        if pd.api.types.is_bool_dtype(series):
            encoded[feature] = series.astype(int)
        elif pd.api.types.is_numeric_dtype(series):
            encoded[feature] = pd.to_numeric(series, errors="raise")
        else:
            domain = [str(value) for value in registered_domains.get(feature, [])]
            if not domain:
                raise ValueError(f"No registered categorical domain for {feature}.")
            mapping = {value: index for index, value in enumerate(domain)}
            values = series.astype(str)
            if not set(values.unique()).issubset(mapping):
                raise ValueError(f"Observed an unregistered level in {feature}.")
            encoded[feature] = values.map(mapping)
    X = encoded[features].to_numpy(dtype=float)
    y = _binary_target(frame, registry)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Every registered split must contain both outcome classes.")
    return X, y, features


def prepare_partition(registry_path: Path) -> dict[str, Any]:
    registry = _verified_json(registry_path)
    partition = registry["partition"]
    raw_path = Path(partition["raw_path"])
    source_path = Path(partition["source_path"])
    planning_path = Path(partition["planning_path"])
    target_path = Path(partition["target_path"])
    manifest_path = Path(partition["manifest_path"])
    if any(path.exists() for path in (source_path, planning_path, target_path, manifest_path)):
        raise FileExistsError("Refusing to overwrite an existing CDC partition.")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(str(registry["dataset"]["data_url"]), raw_path)

    frame = pd.read_csv(raw_path)
    if len(frame) != int(registry["dataset"]["expected_records"]):
        raise ValueError("Downloaded CDC table has an unexpected record count.")
    binary_target = _binary_target(frame, registry)
    if set(np.unique(binary_target)) != {0, 1}:
        raise ValueError("CDC target is not binary as registered.")

    rng = np.random.default_rng(int(partition["seed"]))
    split_positions: dict[str, list[int]] = {"source": [], "planning": [], "target": []}
    split_sizes = {
        "source": int(partition["source_rows_per_class"]),
        "planning": int(partition["planning_rows_per_class"]),
        "target": int(partition["sealed_target_rows_per_class"]),
    }
    for label in (0, 1):
        positions = np.flatnonzero(binary_target == label)
        positions = rng.permutation(positions)
        start = 0
        for split_name in ("source", "planning", "target"):
            stop = start + split_sizes[split_name]
            if stop > positions.size:
                raise ValueError(f"Not enough class-{label} rows for the registered split.")
            split_positions[split_name].extend(positions[start:stop].tolist())
            start = stop

    output_paths = {
        "source": source_path,
        "planning": planning_path,
        "target": target_path,
    }
    for split_name, path in output_paths.items():
        positions = np.asarray(split_positions[split_name], dtype=int)
        positions = rng.permutation(positions)
        frame.iloc[positions].to_csv(path, index=False)

    manifest = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "planning_path": str(planning_path),
        "planning_sha256": _sha256(planning_path),
        "target_path": str(target_path),
        "target_sha256": _sha256(target_path),
        "source_records": len(split_positions["source"]),
        "planning_records": len(split_positions["planning"]),
        "target_records": len(split_positions["target"]),
        "target_outcomes_used_for_development": False,
        "partition_rule": partition["sampling_design"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_classifier(settings: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(settings["regularization_c"]),
                    max_iter=int(settings["maximum_iterations"]),
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def _scale_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = X.min(axis=0)
    upper = X.max(axis=0)
    width = np.where(upper > lower, upper - lower, 1.0)
    return np.clip((X - lower) / width, 0.0, 1.0), lower, upper


def _fit_release_model(
    X_source: np.ndarray,
    y_source: np.ndarray,
    config: dict[str, Any],
    source_settings: dict[str, Any],
    synthetic_rows: int,
    seed: int,
) -> Pipeline:
    """Fit one random-feature denoising generator and its proxy procedure."""

    rng = np.random.default_rng(seed)
    scaled, lower, upper = _scale_features(X_source)
    width = np.where(upper > lower, upper - lower, 1.0)
    corrupted = np.clip(
        scaled + rng.normal(0.0, float(config["train_noise"]), scaled.shape),
        0.0,
        1.0,
    )
    design = np.column_stack([corrupted, y_source])
    hidden_units = int(config["hidden_units"])
    weights = rng.normal(
        0.0,
        1.0 / np.sqrt(design.shape[1]),
        size=(design.shape[1], hidden_units),
    )
    bias = rng.uniform(-0.5, 0.5, size=hidden_units)
    hidden = np.tanh(design @ weights + bias)
    hidden = np.column_stack([hidden, np.ones(hidden.shape[0])])
    penalty = float(config["ridge"]) * np.eye(hidden.shape[1])
    penalty[-1, -1] = 0.0
    decoder = np.linalg.solve(hidden.T @ hidden + penalty, hidden.T @ scaled)

    synthetic_y = np.arange(synthetic_rows, dtype=int) % 2
    rng.shuffle(synthetic_y)
    class_positions = {
        label: np.flatnonzero(y_source == label) for label in (0, 1)
    }
    matching = rng.random(synthetic_rows) < float(config["label_fidelity"])
    base_positions = np.empty(synthetic_rows, dtype=int)
    for label in (0, 1):
        mask = synthetic_y == label
        desired = np.where(matching[mask], label, 1 - label)
        selected = np.empty(mask.sum(), dtype=int)
        for desired_label in (0, 1):
            desired_mask = desired == desired_label
            selected[desired_mask] = rng.choice(
                class_positions[desired_label],
                size=int(desired_mask.sum()),
                replace=True,
            )
        base_positions[mask] = selected

    synthetic_input = np.clip(
        scaled[base_positions]
        + rng.normal(
            0.0,
            float(config["sample_noise"]),
            size=(synthetic_rows, scaled.shape[1]),
        ),
        0.0,
        1.0,
    )
    synthetic_design = np.column_stack([synthetic_input, synthetic_y])
    synthetic_hidden = np.tanh(synthetic_design @ weights + bias)
    synthetic_hidden = np.column_stack(
        [synthetic_hidden, np.ones(synthetic_hidden.shape[0])]
    )
    decoded = np.clip(synthetic_hidden @ decoder, 0.0, 1.0)
    synthetic_X = np.rint(decoded * width + lower)
    synthetic_X = np.clip(synthetic_X, lower, upper)

    model = _build_classifier(source_settings, seed)
    model.fit(synthetic_X, synthetic_y)
    return model


def _losses(
    y: np.ndarray,
    probability: np.ndarray,
    source_settings: dict[str, Any],
) -> np.ndarray:
    prediction = probability >= float(source_settings["decision_threshold"])
    cost_prediction = probability >= float(source_settings["cost_threshold"])
    error = (prediction != y).astype(float)
    brier = np.square(probability - y)
    false_negative = (y == 1) & (~cost_prediction)
    false_positive = (y == 0) & cost_prediction
    normalized_cost = false_negative.astype(float) + 0.2 * false_positive.astype(float)
    return np.column_stack([error, brier, normalized_cost])


def _primary_limits(
    source_risks: np.ndarray,
    registry: dict[str, Any],
    degradation_budget: float | None = None,
) -> np.ndarray:
    claims = registry["claims"]
    budget = (
        float(claims["primary_degradation_budget"])
        if degradation_budget is None
        else float(degradation_budget)
    )
    ceilings = np.asarray(
        [float(claims["absolute_ceilings"][name]) for name in LOSS_NAMES],
        dtype=float,
    )
    return np.minimum(source_risks + budget, ceilings)


def _load_partition_manifest(registry: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(registry["partition"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = (
        ("source_path", "source_sha256"),
        ("planning_path", "planning_sha256"),
        ("target_path", "target_sha256"),
    )
    for path_key, hash_key in checks:
        if _sha256(Path(manifest[path_key])) != str(manifest[hash_key]):
            raise ValueError(f"Partition artifact changed: {manifest[path_key]}")
    return manifest


def run_pilot(registry_path: Path, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = _verified_json(registry_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _load_partition_manifest(registry)
    source = pd.read_csv(manifest["source_path"])
    planning = pd.read_csv(manifest["planning_path"])
    X_source, y_source, features = _split_xy(source, registry)
    X_planning, y_planning, planning_features = _split_xy(planning, registry)
    if features != planning_features:
        raise ValueError("Source and planning feature columns differ.")

    source_settings = registry["source_procedure"]
    source_model = _build_classifier(source_settings, int(registry["partition"]["seed"]))
    source_model.fit(X_source, y_source)
    source_probability = source_model.predict_proba(X_planning)[:, 1]
    source_risks = _losses(y_planning, source_probability, source_settings).mean(axis=0)
    limits = _primary_limits(source_risks, registry)

    rows: list[dict[str, Any]] = []
    search = registry["generator_search"]
    start_time = time.perf_counter()
    for config_index, config in enumerate(search["configurations"]):
        for release_index in range(int(search["pilot_releases_per_configuration"])):
            seed = int(search["pilot_seed_base"]) + 1000 * config_index + release_index
            model = _fit_release_model(
                X_source,
                y_source,
                config,
                source_settings,
                int(search["synthetic_rows"]),
                seed,
            )
            probability = model.predict_proba(X_planning)[:, 1]
            risks = _losses(y_planning, probability, source_settings).mean(axis=0)
            row: dict[str, Any] = {
                "Configuration": str(config["id"]),
                "Release": release_index,
                "Seed": seed,
                "MinimumMargin": float(np.min(limits - risks)),
            }
            for index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(risks[index])
                row[f"Margin_{name}"] = float(limits[index] - risks[index])
            rows.append(row)
        print(f"pilot {config['id']} complete", flush=True)

    pilot = pd.DataFrame(rows)
    pilot_path = output_root / "pilot_release_risks.csv"
    pilot.to_csv(pilot_path, index=False)
    summary = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "partition_manifest": str(registry["partition"]["manifest_path"]),
        "partition_manifest_sha256": _sha256(Path(registry["partition"]["manifest_path"])),
        "pilot_release_risks": str(pilot_path),
        "pilot_release_risks_sha256": _sha256(pilot_path),
        "features": features,
        "source_planning_risks": dict(zip(LOSS_NAMES, source_risks.tolist(), strict=True)),
        "primary_limits": dict(zip(LOSS_NAMES, limits.tolist(), strict=True)),
        "pilot_seconds": time.perf_counter() - start_time,
        "sealed_target_opened": False,
    }
    summary_path = output_root / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return pilot, summary


def _select_configurations(
    pilot: pd.DataFrame,
    target_margin: float,
) -> dict[str, str]:
    grouped = pilot.groupby("Configuration")["MinimumMargin"]
    statistics = pd.DataFrame(
        {
            "q10": grouped.quantile(0.1),
            "median": grouped.median(),
            "q90": grouped.quantile(0.9),
        }
    ).sort_index()
    high = str(statistics["q10"].idxmax())
    degraded = str(statistics.drop(index=high)["q90"].idxmin())
    candidates = statistics.drop(index=[high, degraded])
    positive = candidates[candidates["q10"] > 0.0]
    if not positive.empty:
        candidates = positive
    distance = (candidates["median"] - target_margin).abs()
    moderate = str(distance.sort_values(kind="stable").index[0])
    return {
        "high_fidelity": high,
        "moderate_evidence": moderate,
        "degraded": degraded,
    }


def freeze_audit_registry(
    partition_registry_path: Path,
    pilot_root: Path,
    audit_registry_path: Path,
) -> dict[str, Any]:
    registry = _verified_json(partition_registry_path)
    pilot_summary_path = pilot_root / "pilot_summary.json"
    pilot_path = pilot_root / "pilot_release_risks.csv"
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    if _sha256(pilot_path) != str(pilot_summary["pilot_release_risks_sha256"]):
        raise ValueError("Pilot risks changed after development.")
    if bool(pilot_summary["sealed_target_opened"]):
        raise ValueError("Cannot freeze after target access.")
    pilot = pd.read_csv(pilot_path)
    limits = np.asarray([pilot_summary["primary_limits"][name] for name in LOSS_NAMES])

    selected_slack: float | None = None
    selected_plan: dict[str, Any] | None = None
    for slack in registry["claims"]["score_slack_grid"]:
        if np.any(limits - float(slack) <= 0.0):
            continue
        plan = plan_conditional_shared_target(
            target_records=2 * int(registry["partition"]["sealed_target_rows_per_class"]),
            tolerances=limits,
            slacks=np.full(len(LOSS_NAMES), float(slack)),
            lower_bounds=np.zeros(len(LOSS_NAMES)),
            upper_bounds=np.ones(len(LOSS_NAMES)),
            minimum_reliability=float(registry["claims"]["minimum_reliability"]),
            error_rate=float(registry["claims"]["total_alpha"]),
            target_error_fraction=float(
                registry["claims"]["direct_target_error_fraction"]
            ),
            mechanisms=int(registry["claims"]["registered_configurations"]),
        )
        if plan.target_contamination_allowance <= 0.04:
            selected_slack = float(slack)
            selected_plan = {
                "invalid_release_score_ceiling": plan.invalid_release_score_ceiling,
                "target_contamination_allowance": plan.target_contamination_allowance,
                "minimum_target_records": plan.minimum_target_records,
                "minimum_releases": plan.minimum_releases,
                "release_error_rate": plan.release_error_rate,
                "target_error_rate": plan.target_error_rate,
            }
            break
    if selected_slack is None or selected_plan is None:
        raise RuntimeError("No registered slack meets the development-only planning rule.")

    selected_ids = _select_configurations(pilot, selected_slack + 0.015)
    configs = {
        str(config["id"]): config
        for config in registry["generator_search"]["configurations"]
    }
    selected_configs = {
        role: configs[config_id] for role, config_id in selected_ids.items()
    }
    release_seeds = {
        role: list(
            range(
                int(registry["claims"]["release_seed_bases"][role]),
                int(registry["claims"]["release_seed_bases"][role])
                + int(registry["claims"]["audit_releases"][role]),
            )
        )
        for role in selected_ids
    }

    payload: dict[str, Any] = {
        "registry_version": "1.0",
        "frozen_on": "2026-08-02",
        "stage": "sealed_audit",
        "analysis_status": "Prospective full-pipeline neural-generator audit frozen after development-only planning and before target access.",
        "partition_registry": str(partition_registry_path),
        "partition_registry_sha256": _sha256(partition_registry_path),
        "pilot_summary": str(pilot_summary_path),
        "pilot_summary_sha256": _sha256(pilot_summary_path),
        "pilot_release_risks": str(pilot_path),
        "pilot_release_risks_sha256": _sha256(pilot_path),
        "dataset": registry["dataset"],
        "partition": registry["partition"],
        "source_procedure": registry["source_procedure"],
        "generator": {
            "name": registry["generator_search"]["name"],
            "description": registry["generator_search"]["description"],
            "synthetic_rows": registry["generator_search"]["synthetic_rows"],
            "selected_configurations": selected_configs,
            "selection_rule": registry["generator_search"]["selection_rule"],
            "release_seeds": release_seeds,
            "mechanism_scope": "Each draw independently refits the random hidden representation, reconstruction map, synthetic release, and proxy classifier.",
            "privacy_claim": "No differential privacy claim is made for this neural generator.",
        },
        "requirements": [
            {
                "name": name,
                "limit": float(limits[index]),
                "lower": 0.0,
                "upper": 1.0,
                "primary_degradation_budget": float(
                    registry["claims"]["primary_degradation_budget"]
                ),
                "absolute_ceiling": float(
                    registry["claims"]["absolute_ceilings"][name]
                ),
                "source_planning_risk": float(
                    pilot_summary["source_planning_risks"][name]
                ),
                "score_slack": selected_slack,
                "score_cutoff": float(limits[index] - selected_slack),
            }
            for index, name in enumerate(LOSS_NAMES)
        ],
        "specification_envelope": {
            "strict_degradation_budget": registry["claims"][
                "strict_degradation_budget"
            ],
            "primary_degradation_budget": registry["claims"][
                "primary_degradation_budget"
            ],
            "permissive_degradation_budget": registry["claims"][
                "permissive_degradation_budget"
            ],
            "headline_specification": "primary",
        },
        "minimum_reliability": registry["claims"]["minimum_reliability"],
        "total_alpha": registry["claims"]["total_alpha"],
        "named_release_error_share": registry["claims"][
            "named_release_error_share"
        ],
        "direct_target_error_fraction": registry["claims"][
            "direct_target_error_fraction"
        ],
        "registered_mechanisms": registry["claims"]["registered_configurations"],
        "direct_plan": selected_plan,
        "headline_rule": registry["claims"]["headline_rule"],
        "sealed_target_opened_before_freeze": False,
    }
    _write_hashed_json(audit_registry_path, payload)
    return payload


def generate_release_procedures(
    audit_registry_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    registry = _verified_json(audit_registry_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite generated releases: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _load_partition_manifest(registry)
    source = pd.read_csv(manifest["source_path"])
    X_source, y_source, features = _split_xy(source, registry)
    source_model = _build_classifier(registry["source_procedure"], 2026080201)
    source_model.fit(X_source, y_source)

    models: dict[str, list[Pipeline]] = {}
    timings: dict[str, float] = {}
    for role, config in registry["generator"]["selected_configurations"].items():
        started = time.perf_counter()
        role_models: list[Pipeline] = []
        seeds = registry["generator"]["release_seeds"][role]
        for release_index, seed in enumerate(seeds, start=1):
            role_models.append(
                _fit_release_model(
                    X_source,
                    y_source,
                    config,
                    registry["source_procedure"],
                    int(registry["generator"]["synthetic_rows"]),
                    int(seed),
                )
            )
            if release_index % 25 == 0 or release_index == len(seeds):
                print(f"generated {role} {release_index}/{len(seeds)}", flush=True)
        models[role] = role_models
        timings[role] = time.perf_counter() - started

    bundle_path = output_root / "release_procedures.joblib"
    joblib.dump(
        {
            "source_model": source_model,
            "release_models": models,
            "features": features,
        },
        bundle_path,
        compress=3,
    )
    generation_manifest = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "release_bundle": str(bundle_path),
        "release_bundle_sha256": _sha256(bundle_path),
        "release_counts": {role: len(role_models) for role, role_models in models.items()},
        "generation_seconds": timings,
        "sealed_target_opened": False,
        "full_pipeline_draw": registry["generator"]["mechanism_scope"],
    }
    generation_manifest_path = output_root / "generation_manifest.json"
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return generation_manifest


def _requirements(registry: dict[str, Any]) -> list[RiskRequirement]:
    return [
        RiskRequirement(
            name=str(requirement["name"]),
            tolerance=float(requirement["limit"]),
            lower=float(requirement["lower"]),
            upper=float(requirement["upper"]),
        )
        for requirement in registry["requirements"]
    ]


def _stratified_named_summary(
    evidence: Any,
    *,
    registry: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    releases, requirements = evidence.validation_pvalues.shape
    mechanisms = int(registry["registered_mechanisms"])
    local_total = float(registry["total_alpha"]) / mechanisms
    release_alpha = local_total * float(registry["named_release_error_share"])
    mechanism_alpha = local_total - release_alpha

    candidate_names = [f"{role}_{index:04d}" for index in range(releases)]
    release_pvalues = {
        candidate: float(evidence.validation_pvalues[index].max())
        for index, candidate in enumerate(candidate_names)
    }
    adjusted_release = holm_adjust(release_pvalues)
    validated = np.asarray(
        [adjusted_release[candidate] <= release_alpha for candidate in candidate_names]
    )

    component_pvalues = {
        f"{candidate_names[release_index]}::{requirement_index}": float(
            evidence.violation_pvalues[release_index, requirement_index]
        )
        for release_index in range(releases)
        for requirement_index in range(requirements)
    }
    adjusted_components = holm_adjust(component_pvalues)
    violations = np.zeros(releases, dtype=bool)
    for release_index, candidate in enumerate(candidate_names):
        violations[release_index] = any(
            adjusted_components[f"{candidate}::{requirement_index}"] <= release_alpha
            for requirement_index in range(requirements)
        )

    validated_count = int(validated.sum())
    violation_count = int(violations.sum())
    reliability_lower = clopper_pearson_lower_bound(
        validated_count,
        releases,
        mechanism_alpha,
    )
    reliability_upper = clopper_pearson_upper_bound(
        releases - violation_count,
        releases,
        mechanism_alpha,
    )
    minimum_reliability = float(registry["minimum_reliability"])
    if reliability_lower > minimum_reliability:
        status = "Mechanism validated"
    elif reliability_upper < minimum_reliability:
        status = "Reliability violation detected"
    else:
        status = "Unresolved"

    rows: list[dict[str, Any]] = []
    requirement_names = [str(item["name"]) for item in registry["requirements"]]
    for release_index, candidate in enumerate(candidate_names):
        row: dict[str, Any] = {
            "Candidate": candidate,
            "CandidatePValue": release_pvalues[candidate],
            "HolmAdjustedPValue": adjusted_release[candidate],
            "Validated": bool(validated[release_index]),
            "ViolationDetected": bool(violations[release_index]),
        }
        for requirement_index, requirement_name in enumerate(requirement_names):
            row[f"Risk_{requirement_name}"] = float(
                evidence.weighted_means[release_index, requirement_index]
            )
            row[f"ValidationP_{requirement_name}"] = float(
                evidence.validation_pvalues[release_index, requirement_index]
            )
            row[f"ViolationP_{requirement_name}"] = float(
                evidence.violation_pvalues[release_index, requirement_index]
            )
        rows.append(row)
    return (
        {
            "ValidatedReleases": validated_count,
            "DetectedReleaseViolations": violation_count,
            "ReliabilityLCB": reliability_lower,
            "ReliabilityUCB": reliability_upper,
            "Status": status,
        },
        pd.DataFrame(rows),
    )


def run_audit(
    audit_registry_path: Path,
    generation_root: Path,
    output_root: Path,
    sampling_correction_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = _verified_json(audit_registry_path)
    sampling_correction = (
        None
        if sampling_correction_path is None
        else _verified_json(sampling_correction_path)
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_root}")
    output_root.mkdir(parents=True)
    partition_manifest = _load_partition_manifest(registry)
    generation_manifest_path = generation_root / "generation_manifest.json"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    if bool(generation_manifest["sealed_target_opened"]):
        raise ValueError("Generation manifest says the target was opened.")
    if _sha256(audit_registry_path) != str(generation_manifest["audit_registry_sha256"]):
        raise ValueError("Release procedures were generated under another registry.")
    bundle_path = Path(generation_manifest["release_bundle"])
    if _sha256(bundle_path) != str(generation_manifest["release_bundle_sha256"]):
        raise ValueError("Generated release bundle changed before audit.")
    bundle = joblib.load(bundle_path)

    target_opened_at = time.perf_counter()
    target = pd.read_csv(partition_manifest["target_path"])
    X_target, y_target, features = _split_xy(target, registry)
    if features != bundle["features"]:
        raise ValueError("Target features differ from the frozen release bundle.")
    source_probability = bundle["source_model"].predict_proba(X_target)[:, 1]
    source_losses = _losses(
        y_target,
        source_probability,
        registry["source_procedure"],
    )
    source_target_risks = source_losses.mean(axis=0)

    stratum_masks: list[np.ndarray] | None = None
    stratum_weights: list[float] | None = None
    if sampling_correction is not None:
        if str(sampling_correction["audit_registry_sha256"]) != _sha256(
            audit_registry_path
        ):
            raise ValueError("Sampling correction names another audit registry.")
        labels = [int(value) for value in sampling_correction["stratum_labels"]]
        stratum_weights = [
            float(value) for value in sampling_correction["stratum_weights"]
        ]
        stratum_masks = [y_target == label for label in labels]
        if any(not bool(mask.any()) for mask in stratum_masks):
            raise ValueError("A registered target stratum is empty.")
        source_target_risks = np.sum(
            [
                weight * source_losses[mask].mean(axis=0)
                for weight, mask in zip(
                    stratum_weights,
                    stratum_masks,
                    strict=True,
                )
            ],
            axis=0,
        )

    tolerances = np.asarray(
        [float(requirement["limit"]) for requirement in registry["requirements"]]
    )
    slacks = np.asarray(
        [float(requirement["score_slack"]) for requirement in registry["requirements"]]
    )
    requirements = _requirements(registry)
    release_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    mechanisms = int(registry["registered_mechanisms"])

    for role, models in bundle["release_models"].items():
        candidate_losses: dict[str, dict[str, np.ndarray]] = {}
        release_to_mechanism: dict[str, str] = {}
        release_means = np.empty((len(models), len(LOSS_NAMES)), dtype=float)
        role_losses = np.empty(
            (len(models), len(target), len(LOSS_NAMES)),
            dtype=float,
        )
        for release_index, model in enumerate(models):
            probability = model.predict_proba(X_target)[:, 1]
            losses = _losses(y_target, probability, registry["source_procedure"])
            role_losses[release_index] = losses
            release_means[release_index] = losses.mean(axis=0)
            candidate = f"{role}_{release_index:04d}"
            candidate_losses[candidate] = {
                name: losses[:, index] for index, name in enumerate(LOSS_NAMES)
            }
            release_to_mechanism[candidate] = role

        if stratum_masks is None or stratum_weights is None:
            named = audit_proxy_mechanisms(
                candidate_losses,
                release_to_mechanism=release_to_mechanism,
                requirements=requirements,
                minimum_reliability=float(registry["minimum_reliability"]),
                total_alpha=float(registry["total_alpha"]) / mechanisms,
                release_error_share=float(registry["named_release_error_share"]),
                violation_total_alpha=float(registry["total_alpha"]) / mechanisms,
                violation_release_error_share=float(
                    registry["named_release_error_share"]
                ),
            )
            direct = shared_target_conditional_mean_lower_bound(
                release_means,
                target_records=len(target),
                tolerances=tolerances,
                slacks=slacks,
                lower_bounds=np.zeros(len(LOSS_NAMES)),
                upper_bounds=np.ones(len(LOSS_NAMES)),
                error_rate=float(registry["total_alpha"]),
                target_error_fraction=float(
                    registry["direct_target_error_fraction"]
                ),
                mechanisms=mechanisms,
            )
            named_row: Any = named.mechanism_summary.iloc[0]
            named_release_frame = named.release_audit.candidate_summary.assign(
                ConfigurationRole=role
            )
        else:
            losses_by_stratum = [role_losses[:, mask, :] for mask in stratum_masks]
            evidence = stratified_release_evidence(
                losses_by_stratum,
                stratum_weights=stratum_weights,
                tolerances=tolerances,
                lower_bounds=np.zeros(len(LOSS_NAMES)),
                upper_bounds=np.ones(len(LOSS_NAMES)),
            )
            named_row, named_release_frame = _stratified_named_summary(
                evidence,
                registry=registry,
                role=role,
            )
            release_means = evidence.weighted_means
            direct = stratified_shared_target_conditional_witness_lower_bound(
                losses_by_stratum,
                stratum_weights=stratum_weights,
                tolerances=tolerances,
                slacks=slacks,
                lower_bounds=np.zeros(len(LOSS_NAMES)),
                upper_bounds=np.ones(len(LOSS_NAMES)),
                error_rate=float(registry["total_alpha"]),
                target_error_fraction=float(
                    registry["direct_target_error_fraction"]
                ),
                mechanisms=mechanisms,
            )
        mechanism_rows.append(
            {
                "ConfigurationRole": role,
                "Configuration": registry["generator"]["selected_configurations"][role]["id"],
                "Releases": len(models),
                "TargetRecords": len(target),
                "NamedValidatedReleases": int(named_row["ValidatedReleases"]),
                "NamedViolatingReleases": int(named_row["DetectedReleaseViolations"]),
                "NamedReliabilityLCB": float(named_row["ReliabilityLCB"]),
                "NamedReliabilityUCB": float(named_row["ReliabilityUCB"]),
                "NamedDecision": str(named_row["Status"]),
                "DirectFavorableScores": int(
                    np.rint(direct.conditional_score_mean * len(models))
                ),
                "DirectScoreRate": direct.conditional_score_mean,
                "DirectScoreLCB": direct.conditional_score_lower_bound,
                "InvalidScoreCeiling": direct.invalid_release_score_ceiling,
                "ContaminationAllowance": direct.target_contamination_allowance,
                "DirectReliabilityLCB": direct.reliability_lower_bound,
                "DirectDecision": (
                    "Mechanism validated"
                    if direct.reliability_lower_bound
                    > float(registry["minimum_reliability"])
                    else "Unresolved"
                ),
                "MinimumReliability": float(registry["minimum_reliability"]),
            }
        )
        named_release_frame.to_csv(
            output_root / f"{role}_named_release_summary.csv",
            index=False,
        )
        for release_index, means in enumerate(release_means):
            row: dict[str, Any] = {
                "ConfigurationRole": role,
                "Release": release_index,
            }
            for index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(means[index])
                row[f"Margin_{name}"] = float(tolerances[index] - means[index])
            release_rows.append(row)

    release_frame = pd.DataFrame(release_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    release_frame.to_csv(output_root / "release_risks.csv", index=False)
    mechanism_frame.to_csv(output_root / "mechanism_summary.csv", index=False)
    result_manifest = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "generation_manifest": str(generation_manifest_path),
        "generation_manifest_sha256": _sha256(generation_manifest_path),
        "target_path": partition_manifest["target_path"],
        "target_sha256": partition_manifest["target_sha256"],
        "target_opened_after_release_bundle_frozen": True,
        "target_records": len(target),
        "sampling_correction": (
            None if sampling_correction_path is None else str(sampling_correction_path)
        ),
        "sampling_correction_sha256": (
            None
            if sampling_correction_path is None
            else _sha256(sampling_correction_path)
        ),
        "source_target_risks": dict(
            zip(LOSS_NAMES, source_target_risks.tolist(), strict=True)
        ),
        "evaluation_seconds": time.perf_counter() - target_opened_at,
        "mechanism_summary_sha256": _sha256(output_root / "mechanism_summary.csv"),
        "release_risks_sha256": _sha256(output_root / "release_risks.csv"),
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(result_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return release_frame, mechanism_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed CDC full-pipeline neural shared-target audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--partition-registry",
        default="registries/proxyguard_cdc_neural_partition.json",
    )
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument(
        "--partition-registry",
        default="registries/proxyguard_cdc_neural_partition.json",
    )
    pilot.add_argument(
        "--output-root",
        default="outputs/proxyguard_cdc_neural_direct/pilot",
    )
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument(
        "--partition-registry",
        default="registries/proxyguard_cdc_neural_partition.json",
    )
    freeze.add_argument(
        "--pilot-root",
        default="outputs/proxyguard_cdc_neural_direct/pilot",
    )
    freeze.add_argument(
        "--audit-registry",
        default="registries/proxyguard_cdc_neural_audit.json",
    )
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--audit-registry",
        default="registries/proxyguard_cdc_neural_audit.json",
    )
    generate.add_argument(
        "--output-root",
        default="outputs/proxyguard_cdc_neural_direct/generated",
    )
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--audit-registry",
        default="registries/proxyguard_cdc_neural_audit.json",
    )
    audit.add_argument(
        "--generation-root",
        default="outputs/proxyguard_cdc_neural_direct/generated",
    )
    audit.add_argument(
        "--output-root",
        default="outputs/proxyguard_cdc_neural_direct/audit",
    )
    audit.add_argument(
        "--sampling-correction",
        default=None,
        help="Optional hashed stratified-target correction registry.",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        print(
            json.dumps(
                prepare_partition(Path(args.partition_registry)),
                indent=2,
            )
        )
    elif args.command == "pilot":
        pilot_frame, summary = run_pilot(
            Path(args.partition_registry),
            Path(args.output_root),
        )
        print(pilot_frame.groupby("Configuration")["MinimumMargin"].describe())
        print(json.dumps(summary, indent=2))
    elif args.command == "freeze":
        print(
            json.dumps(
                freeze_audit_registry(
                    Path(args.partition_registry),
                    Path(args.pilot_root),
                    Path(args.audit_registry),
                ),
                indent=2,
            )
        )
    elif args.command == "generate":
        print(
            json.dumps(
                generate_release_procedures(
                    Path(args.audit_registry),
                    Path(args.output_root),
                ),
                indent=2,
            )
        )
    else:
        _, mechanism_frame = run_audit(
            Path(args.audit_registry),
            Path(args.generation_root),
            Path(args.output_root),
            None
            if args.sampling_correction is None
            else Path(args.sampling_correction),
        )
        print(mechanism_frame.to_string(index=False))


if __name__ == "__main__":
    main()
