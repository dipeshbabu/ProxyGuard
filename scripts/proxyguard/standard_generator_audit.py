"""Prospective full-pipeline audit for a standard tabular generator."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from proxyguard.core import RiskRequirement, audit_proxy_mechanisms
from proxyguard.shared_target import (
    plan_conditional_shared_target,
    shared_target_conditional_mean_lower_bound,
)
from scripts.proxyguard.neural_direct_audit import (
    LOSS_NAMES,
    _build_classifier,
    _losses,
    _sha256,
    _split_xy,
    _verified_json,
    _write_hashed_json,
)


def _load_manifest(registry: dict[str, Any]) -> dict[str, Any]:
    path = Path(registry["partition"]["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("source_path", "source_sha256"),
        ("planning_path", "planning_sha256"),
        ("target_path", "target_sha256"),
    ):
        if _sha256(Path(manifest[path_key])) != str(manifest[hash_key]):
            raise ValueError(f"Partition artifact changed: {path_key}.")
    return manifest


def _load_partition_plan(
    registry_path: Path,
    amendment_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    registry = _verified_json(registry_path)
    if amendment_path is None:
        return registry, None
    amendment = _verified_json(amendment_path)
    if str(amendment["partition_registry_sha256"]) != _sha256(registry_path):
        raise ValueError("CTGAN amendment names another partition registry.")
    resolved = deepcopy(registry)
    resolved["generator_search"] = amendment["replacement_generator_search"]
    return resolved, amendment


def prepare_partition(registry_path: Path) -> dict[str, Any]:
    registry = _verified_json(registry_path)
    partition = registry["partition"]
    paths = {
        name: Path(partition[f"{name}_path"])
        for name in ("raw", "source", "planning", "target_population", "target")
    }
    manifest_path = Path(partition["manifest_path"])
    if any(path.exists() for name, path in paths.items() if name != "raw") or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a standard-generator partition.")
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["source"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["raw"].exists():
        urllib.request.urlretrieve(str(registry["dataset"]["data_url"]), paths["raw"])

    frame = pd.read_csv(paths["raw"])
    if len(frame) != int(registry["dataset"]["expected_records"]):
        raise ValueError("Downloaded table has an unexpected record count.")
    rng = np.random.default_rng(int(partition["seed"]))
    positions = rng.permutation(len(frame))
    source_stop = int(partition["source_records"])
    planning_stop = source_stop + int(partition["planning_records"])
    source_positions = positions[:source_stop]
    planning_positions = positions[source_stop:planning_stop]
    target_population_positions = positions[planning_stop:]
    if target_population_positions.size < 1:
        raise ValueError("The registered partition leaves no target population.")

    frame.iloc[source_positions].to_csv(paths["source"], index=False)
    frame.iloc[planning_positions].to_csv(paths["planning"], index=False)
    target_population = frame.iloc[target_population_positions]
    target_population.to_csv(paths["target_population"], index=False)
    audit_positions = rng.choice(
        target_population_positions.size,
        size=int(partition["target_records"]),
        replace=True,
    )
    target_population.iloc[audit_positions].to_csv(paths["target"], index=False)

    manifest = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "raw_path": str(paths["raw"]),
        "raw_sha256": _sha256(paths["raw"]),
        "source_path": str(paths["source"]),
        "source_sha256": _sha256(paths["source"]),
        "planning_path": str(paths["planning"]),
        "planning_sha256": _sha256(paths["planning"]),
        "target_population_path": str(paths["target_population"]),
        "target_population_sha256": _sha256(paths["target_population"]),
        "target_path": str(paths["target"]),
        "target_sha256": _sha256(paths["target"]),
        "source_records": int(source_positions.size),
        "planning_records": int(planning_positions.size),
        "target_population_records": int(target_population_positions.size),
        "target_records": int(audit_positions.size),
        "target_sampling": "iid with replacement from the sealed empirical reserve",
        "target_outcomes_used_for_development": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _fit_ctgan_release(
    source: pd.DataFrame,
    registry: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> Any:
    try:
        from ctgan import CTGAN, TVAE
    except ImportError as exc:  # pragma: no cover - exercised in the artifact runtime
        raise RuntimeError("Install the official ctgan package to run this audit.") from exc

    rng = np.random.default_rng(seed)
    target_name = str(registry["dataset"]["target"])
    training_rows = int(config["training_rows"])
    selected = rng.choice(len(source), size=training_rows, replace=False)
    training = source.iloc[selected].copy().reset_index(drop=True)
    flip_probability = float(config["label_flip_probability"])
    if flip_probability > 0.0:
        flip = rng.random(training_rows) < flip_probability
        labels = training[target_name].astype(str).to_numpy(copy=True)
        first, second = [str(value) for value in registry["dataset"]["class_labels"]]
        labels[flip & (labels == first)] = second
        labels[flip & (labels == second)] = first
        training[target_name] = labels

    generator_type = str(config.get("generator_type", "CTGAN")).upper()
    if generator_type == "CTGAN":
        model = CTGAN(
            embedding_dim=int(config["embedding_dim"]),
            generator_dim=tuple(int(value) for value in config["generator_dim"]),
            discriminator_dim=tuple(int(value) for value in config["discriminator_dim"]),
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            pac=int(config["pac"]),
            enable_gpu=False,
            verbose=False,
        )
    elif generator_type == "TVAE":
        model = TVAE(
            embedding_dim=int(config["embedding_dim"]),
            compress_dims=tuple(int(value) for value in config["compress_dims"]),
            decompress_dims=tuple(int(value) for value in config["decompress_dims"]),
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            l2scale=float(config.get("l2scale", 1e-5)),
            loss_factor=float(config.get("loss_factor", 2.0)),
            enable_gpu=False,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported standard generator type: {generator_type}.")
    model.set_random_state(seed)
    model.fit(training, discrete_columns=[target_name])
    if "synthetic_rows" in config:
        synthetic_rows = int(config["synthetic_rows"])
    elif "generator_search" in registry:
        synthetic_rows = int(registry["generator_search"]["synthetic_rows"])
    else:
        synthetic_rows = int(registry["generator"]["synthetic_rows"])
    if bool(config.get("conditional_class_sampling", False)):
        if generator_type != "CTGAN":
            raise ValueError("Conditional class sampling is registered only for CTGAN.")
        class_labels = [str(value) for value in registry["dataset"]["class_labels"]]
        class_counts = [synthetic_rows // len(class_labels)] * len(class_labels)
        for index in range(synthetic_rows % len(class_labels)):
            class_counts[index] += 1
        synthetic = pd.concat(
            [
                model.sample(
                    count,
                    condition_column=target_name,
                    condition_value=label,
                )
                for label, count in zip(class_labels, class_counts, strict=True)
            ],
            ignore_index=True,
        )
        synthetic = synthetic.iloc[rng.permutation(len(synthetic))].reset_index(drop=True)
    else:
        synthetic = model.sample(synthetic_rows)

    feature_names = [column for column in source.columns if column != target_name]
    for feature in feature_names:
        synthetic[feature] = pd.to_numeric(synthetic[feature], errors="coerce")
        median = float(pd.to_numeric(source[feature]).median())
        synthetic[feature] = synthetic[feature].fillna(median)
        synthetic[feature] = synthetic[feature].clip(
            float(pd.to_numeric(source[feature]).min()),
            float(pd.to_numeric(source[feature]).max()),
        )
    allowed_labels = [str(value) for value in registry["dataset"]["class_labels"]]
    synthetic[target_name] = synthetic[target_name].astype(str)
    synthetic = synthetic[synthetic[target_name].isin(allowed_labels)].copy()
    X_synthetic, y_synthetic, _ = _split_xy(synthetic, registry)
    if set(np.unique(y_synthetic)) != {0, 1}:
        raise ValueError("CTGAN release contains only one registered class.")
    proxy = _build_classifier(registry["source_procedure"], seed)
    proxy.fit(X_synthetic, y_synthetic)
    return proxy


def _limits(source_risks: np.ndarray, registry: dict[str, Any]) -> np.ndarray:
    policy = registry["claims"]["primary_policy"]
    degradation = float(policy["normalized_degradation_budget"])
    ceilings = np.asarray(
        [float(policy["absolute_ceilings"][name]) for name in LOSS_NAMES]
    )
    return np.minimum(source_risks + degradation, ceilings)


def run_pilot(
    registry_path: Path,
    output_root: Path,
    amendment_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry, amendment = _load_partition_plan(registry_path, amendment_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _load_manifest(registry)
    source = pd.read_csv(manifest["source_path"])
    planning = pd.read_csv(manifest["planning_path"])
    X_source, y_source, features = _split_xy(source, registry)
    X_planning, y_planning, planning_features = _split_xy(planning, registry)
    if features != planning_features:
        raise ValueError("Planning features differ from source features.")
    source_model = _build_classifier(registry["source_procedure"], int(registry["partition"]["seed"]))
    source_model.fit(X_source, y_source)
    source_risks = _losses(
        y_planning,
        source_model.predict_proba(X_planning)[:, 1],
        registry["source_procedure"],
    ).mean(axis=0)
    limits = _limits(source_risks, registry)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for configuration_index, config in enumerate(registry["generator_search"]["configurations"]):
        for pilot_index in range(int(registry["generator_search"]["pilot_releases_per_configuration"])):
            seed = int(registry["generator_search"]["pilot_seed_base"]) + 100 * configuration_index + pilot_index
            generation_error: str | None = None
            try:
                model = _fit_ctgan_release(source, registry, config, seed)
                risks = _losses(
                    y_planning,
                    model.predict_proba(X_planning)[:, 1],
                    registry["source_procedure"],
                ).mean(axis=0)
            except ValueError as exc:
                generation_error = str(exc)
                risks = np.ones(len(LOSS_NAMES), dtype=float)
            row: dict[str, Any] = {
                "Configuration": str(config["id"]),
                "PilotRelease": pilot_index,
                "Seed": seed,
                "GenerationFailure": generation_error is not None,
                "GenerationError": generation_error,
                "MinimumMargin": float(np.min(limits - risks)),
            }
            for index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(risks[index])
                row[f"Margin_{name}"] = float(limits[index] - risks[index])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "pilot_release_risks.csv", index=False)
    summary = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "pre_audit_amendment": None if amendment_path is None else str(amendment_path),
        "pre_audit_amendment_sha256": (
            None if amendment_path is None else _sha256(amendment_path)
        ),
        "source_planning_risks": dict(zip(LOSS_NAMES, source_risks.tolist(), strict=True)),
        "primary_limits": dict(zip(LOSS_NAMES, limits.tolist(), strict=True)),
        "pilot_release_risks_sha256": _sha256(output_root / "pilot_release_risks.csv"),
        "elapsed_seconds": time.perf_counter() - started,
        "target_opened": False,
    }
    (output_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return frame, summary


def _select_roles(frame: pd.DataFrame, slack: float) -> dict[str, str]:
    grouped = frame.groupby("Configuration")["MinimumMargin"]
    statistics = pd.DataFrame(
        {
            "q10": grouped.quantile(0.10),
            "median": grouped.median(),
            "q90": grouped.quantile(0.90),
        }
    ).sort_index()
    high = str(statistics["q10"].idxmax())
    positive = statistics[statistics["median"] > slack]
    if positive.empty:
        moderate = str((statistics["median"] - (slack + 0.01)).abs().idxmin())
    else:
        moderate = str((positive["median"] - (slack + 0.01)).abs().idxmin())
    degraded = str(statistics["q90"].idxmin())
    if len({high, moderate, degraded}) < 3:
        ordered = statistics["median"].sort_values()
        degraded = str(ordered.index[0])
        high = str(ordered.index[-1])
        middle = [str(value) for value in ordered.index if str(value) not in {high, degraded}]
        moderate = min(middle, key=lambda value: abs(float(statistics.loc[value, "median"]) - (slack + 0.01)))
    return {"high_fidelity": high, "moderate_evidence": moderate, "degraded": degraded}


def freeze_audit(
    partition_registry_path: Path,
    pilot_root: Path,
    audit_registry_path: Path,
    amendment_path: Path | None = None,
) -> dict[str, Any]:
    partition_registry, amendment = _load_partition_plan(
        partition_registry_path,
        amendment_path,
    )
    pilot_path = pilot_root / "pilot_release_risks.csv"
    pilot_summary_path = pilot_root / "pilot_summary.json"
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    if _sha256(pilot_path) != str(pilot_summary["pilot_release_risks_sha256"]):
        raise ValueError("Pilot risks changed before freeze.")
    frame = pd.read_csv(pilot_path)
    slack = float(partition_registry["claims"]["score_slack"])
    selected_ids = _select_roles(frame, slack)
    configurations = {
        str(item["id"]): item for item in partition_registry["generator_search"]["configurations"]
    }
    release_counts = partition_registry["claims"]["audit_releases"]
    seed_bases = partition_registry["claims"]["release_seed_bases"]
    selected = {role: configurations[value] for role, value in selected_ids.items()}
    seeds = {
        role: [int(seed_bases[role]) + index for index in range(int(release_counts[role]))]
        for role in selected
    }
    target_records = int(partition_registry["partition"]["target_records"])
    plan = plan_conditional_shared_target(
        target_records=target_records,
        minimum_reliability=float(partition_registry["claims"]["minimum_reliability"]),
        tolerances=list(pilot_summary["primary_limits"].values()),
        slacks=[slack] * len(LOSS_NAMES),
        error_rate=float(partition_registry["claims"]["total_alpha"]),
        target_error_fraction=float(partition_registry["claims"]["direct_target_error_fraction"]),
        mechanisms=int(partition_registry["claims"]["registered_configurations"]),
    )
    degradation = float(partition_registry["claims"]["primary_policy"]["normalized_degradation_budget"])
    requirements = [
        {
            "name": name,
            "limit": float(pilot_summary["primary_limits"][name]),
            "lower": 0.0,
            "upper": 1.0,
            "policy": (
                f"source planning risk plus {degradation:g}, capped by the "
                "registered absolute ceiling"
            ),
            "score_slack": slack,
            "score_cutoff": float(pilot_summary["primary_limits"][name]) - slack,
        }
        for name in LOSS_NAMES
    ]
    frozen = {
        "registry_version": "1.0",
        "frozen_on": "2026-08-03",
        "analysis_status": "Prospective standard-generator audit frozen after development-only planning and before target access.",
        "partition_registry": str(partition_registry_path),
        "partition_registry_sha256": _sha256(partition_registry_path),
        "pre_audit_amendment": None if amendment_path is None else str(amendment_path),
        "pre_audit_amendment_sha256": (
            None if amendment_path is None else _sha256(amendment_path)
        ),
        "pilot_summary": str(pilot_summary_path),
        "pilot_summary_sha256": _sha256(pilot_summary_path),
        "pilot_release_risks": str(pilot_path),
        "pilot_release_risks_sha256": _sha256(pilot_path),
        "dataset": partition_registry["dataset"],
        "partition": partition_registry["partition"],
        "source_procedure": partition_registry["source_procedure"],
        "generator": {
            "name": str(partition_registry["generator_search"]["name"]),
            "selected_configurations": selected,
            "release_seeds": seeds,
            "selection_rule": partition_registry["generator_search"]["selection_rule"],
            "synthetic_rows": int(partition_registry["generator_search"]["synthetic_rows"]),
        },
        "requirements": requirements,
        "minimum_reliability": float(partition_registry["claims"]["minimum_reliability"]),
        "total_alpha": float(partition_registry["claims"]["total_alpha"]),
        "named_release_error_share": float(partition_registry["claims"]["named_release_error_share"]),
        "direct_target_error_fraction": float(partition_registry["claims"]["direct_target_error_fraction"]),
        "registered_mechanisms": int(partition_registry["claims"]["registered_configurations"]),
        "direct_plan": {
            "minimum_target_records": plan.minimum_target_records,
            "minimum_releases": plan.minimum_releases,
            "contamination_allowance": plan.target_contamination_allowance,
        },
        "headline_rule": partition_registry["claims"]["headline_rule"],
        "sealed_target_opened_before_freeze": False,
    }
    _write_hashed_json(audit_registry_path, frozen)
    return frozen


def generate_releases(
    audit_registry_path: Path,
    output_root: Path,
    generation_amendment_path: Path | None = None,
) -> dict[str, Any]:
    registry = _verified_json(audit_registry_path)
    generation_amendment: dict[str, Any] | None = None
    if generation_amendment_path is not None:
        generation_amendment = _verified_json(generation_amendment_path)
        if str(generation_amendment["audit_registry_sha256"]) != _sha256(
            audit_registry_path
        ):
            raise ValueError("Generation amendment names another audit registry.")
        if bool(generation_amendment["target_opened"]):
            raise ValueError("Generation amendment was not frozen before target access.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite generated releases: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _load_manifest(registry)
    source = pd.read_csv(manifest["source_path"])
    X_source, y_source, features = _split_xy(source, registry)
    source_model = _build_classifier(registry["source_procedure"], 0)
    source_model.fit(X_source, y_source)
    release_models: dict[str, list[Any | None]] = {}
    release_failures: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    for role, config in registry["generator"]["selected_configurations"].items():
        started = time.perf_counter()
        release_models[role] = []
        release_failures[role] = []
        for release_index, seed_value in enumerate(registry["generator"]["release_seeds"][role]):
            seed = int(seed_value)
            try:
                release_models[role].append(
                    _fit_ctgan_release(source, registry, config, seed)
                )
            except ValueError as exc:
                if generation_amendment is None or str(
                    generation_amendment.get("failed_release_policy")
                ) != "assign_unit_loss":
                    raise
                release_models[role].append(None)
                release_failures[role].append(
                    {
                        "release_index": release_index,
                        "seed": seed,
                        "error": str(exc),
                    }
                )
        timings[role] = time.perf_counter() - started
    bundle_path = output_root / "release_procedures.joblib"
    joblib.dump(
        {"source_model": source_model, "release_models": release_models, "features": features},
        bundle_path,
        compress=3,
    )
    result = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "generation_amendment": (
            None if generation_amendment_path is None else str(generation_amendment_path)
        ),
        "generation_amendment_sha256": (
            None
            if generation_amendment_path is None
            else _sha256(generation_amendment_path)
        ),
        "release_bundle": str(bundle_path),
        "release_bundle_sha256": _sha256(bundle_path),
        "generation_seconds": timings,
        "generation_failures": release_failures,
        "failed_release_policy": (
            "A release that cannot produce a two-class downstream training table "
            "is retained and assigned loss one for every registered requirement."
        ),
        "sealed_target_opened": False,
    }
    (output_root / "generation_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _requirements(registry: dict[str, Any]) -> list[RiskRequirement]:
    return [
        RiskRequirement(str(item["name"]), float(item["limit"]), 0.0, 1.0)
        for item in registry["requirements"]
    ]


def audit_releases(
    audit_registry_path: Path,
    generation_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = _verified_json(audit_registry_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _load_manifest(registry)
    generation_manifest_path = generation_root / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if bool(generation_manifest["sealed_target_opened"]):
        raise ValueError("Generation manifest says the target was opened.")
    bundle_path = Path(generation_manifest["release_bundle"])
    if _sha256(bundle_path) != str(generation_manifest["release_bundle_sha256"]):
        raise ValueError("Release bundle changed before target access.")
    bundle = joblib.load(bundle_path)
    target_opened_at = time.perf_counter()
    target = pd.read_csv(manifest["target_path"])
    X_target, y_target, features = _split_xy(target, registry)
    if features != bundle["features"]:
        raise ValueError("Target features differ from the frozen bundle.")
    source_risks = _losses(
        y_target,
        bundle["source_model"].predict_proba(X_target)[:, 1],
        registry["source_procedure"],
    ).mean(axis=0)
    tolerances = np.asarray([float(item["limit"]) for item in registry["requirements"]])
    slacks = np.asarray([float(item["score_slack"]) for item in registry["requirements"]])
    mechanisms = int(registry["registered_mechanisms"])
    release_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for role, models in bundle["release_models"].items():
        release_means = np.empty((len(models), len(LOSS_NAMES)))
        candidate_losses: dict[str, dict[str, np.ndarray]] = {}
        release_map: dict[str, str] = {}
        for index, model in enumerate(models):
            if model is None:
                losses = np.ones((len(target), len(LOSS_NAMES)), dtype=float)
            else:
                losses = _losses(
                    y_target,
                    model.predict_proba(X_target)[:, 1],
                    registry["source_procedure"],
                )
            release_means[index] = losses.mean(axis=0)
            candidate = f"{role}_{index:04d}"
            candidate_losses[candidate] = {
                name: losses[:, requirement_index]
                for requirement_index, name in enumerate(LOSS_NAMES)
            }
            release_map[candidate] = role
        named = audit_proxy_mechanisms(
            candidate_losses,
            release_to_mechanism=release_map,
            requirements=_requirements(registry),
            minimum_reliability=float(registry["minimum_reliability"]),
            total_alpha=float(registry["total_alpha"]) / mechanisms,
            release_error_share=float(registry["named_release_error_share"]),
            violation_total_alpha=float(registry["total_alpha"]) / mechanisms,
            violation_release_error_share=float(registry["named_release_error_share"]),
        )
        direct = shared_target_conditional_mean_lower_bound(
            release_means,
            target_records=len(target),
            tolerances=tolerances,
            slacks=slacks,
            error_rate=float(registry["total_alpha"]),
            target_error_fraction=float(registry["direct_target_error_fraction"]),
            mechanisms=mechanisms,
        )
        named_row = named.mechanism_summary.iloc[0]
        mechanism_rows.append(
            {
                "ConfigurationRole": role,
                "Configuration": registry["generator"]["selected_configurations"][role]["id"],
                "Releases": len(models),
                "NamedValidatedReleases": int(named_row["ValidatedReleases"]),
                "NamedViolatingReleases": int(named_row["DetectedReleaseViolations"]),
                "NamedReliabilityLCB": float(named_row["ReliabilityLCB"]),
                "NamedReliabilityUCB": float(named_row["ReliabilityUCB"]),
                "NamedDecision": str(named_row["Status"]),
                "DirectFavorableScores": int(np.rint(direct.conditional_score_mean * len(models))),
                "DirectScoreLCB": direct.conditional_score_lower_bound,
                "ContaminationAllowance": direct.target_contamination_allowance,
                "DirectReliabilityLCB": direct.reliability_lower_bound,
                "DirectDecision": (
                    "Mechanism validated"
                    if direct.reliability_lower_bound > float(registry["minimum_reliability"])
                    else "Unresolved"
                ),
            }
        )
        named.release_audit.candidate_summary.to_csv(
            output_root / f"{role}_named_release_summary.csv", index=False
        )
        for release_index, means in enumerate(release_means):
            row: dict[str, Any] = {"ConfigurationRole": role, "Release": release_index}
            for requirement_index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(means[requirement_index])
                row[f"Margin_{name}"] = float(tolerances[requirement_index] - means[requirement_index])
            release_rows.append(row)
    release_frame = pd.DataFrame(release_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    release_frame.to_csv(output_root / "release_risks.csv", index=False)
    mechanism_frame.to_csv(output_root / "mechanism_summary.csv", index=False)
    result = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "target_path": manifest["target_path"],
        "target_sha256": manifest["target_sha256"],
        "target_opened_after_release_bundle_frozen": True,
        "source_target_risks": dict(zip(LOSS_NAMES, source_risks.tolist(), strict=True)),
        "evaluation_seconds": time.perf_counter() - target_opened_at,
        "mechanism_summary_sha256": _sha256(output_root / "mechanism_summary.csv"),
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return release_frame, mechanism_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prospective standard-generator audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "pilot", "freeze", "generate", "audit"):
        subparsers.add_parser(command)
    parser.add_argument(
        "--partition-registry",
        default="registries/proxyguard_rice_ctgan_partition.json",
    )
    parser.add_argument(
        "--audit-registry",
        default="registries/proxyguard_rice_ctgan_audit.json",
    )
    parser.add_argument("--pilot-root", default="outputs/proxyguard_rice_ctgan/pilot")
    parser.add_argument("--amendment", default=None)
    parser.add_argument("--generation-amendment", default=None)
    parser.add_argument("--generation-root", default="outputs/proxyguard_rice_ctgan/generated")
    parser.add_argument("--audit-root", default="outputs/proxyguard_rice_ctgan/audit")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_partition(Path(args.partition_registry)), indent=2))
    elif args.command == "pilot":
        frame, summary = run_pilot(
            Path(args.partition_registry),
            Path(args.pilot_root),
            None if args.amendment is None else Path(args.amendment),
        )
        print(frame.groupby("Configuration")["MinimumMargin"].describe())
        print(json.dumps(summary, indent=2))
    elif args.command == "freeze":
        print(
            json.dumps(
                freeze_audit(
                    Path(args.partition_registry),
                    Path(args.pilot_root),
                    Path(args.audit_registry),
                    None if args.amendment is None else Path(args.amendment),
                ),
                indent=2,
            )
        )
    elif args.command == "generate":
        print(
            json.dumps(
                generate_releases(
                    Path(args.audit_registry),
                    Path(args.generation_root),
                    None
                    if args.generation_amendment is None
                    else Path(args.generation_amendment),
                ),
                indent=2,
            )
        )
    else:
        _, summary = audit_releases(
            Path(args.audit_registry),
            Path(args.generation_root),
            Path(args.audit_root),
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
