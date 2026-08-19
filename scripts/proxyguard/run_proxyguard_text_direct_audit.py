"""Prospective non-tabular audit for a class-conditional text generator.

The development stages load only the official 20 Newsgroups training subset.
The audit stage is the first code path that loads the official test subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from proxyguard.core import RiskRequirement, audit_proxy_mechanisms
from proxyguard.shared_target import (
    plan_conditional_shared_target,
    shared_target_conditional_mean_lower_bound,
    shared_target_smooth_conditional_mean_lower_bound,
)


LOSS_NAMES = ("Error", "Brier", "Cost3x")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_json(path: Path) -> dict[str, Any]:
    digest_path = path.with_suffix(".sha256")
    if digest_path.exists():
        expected = digest_path.read_text(encoding="utf-8").split()[0]
        if _sha256(path) != expected:
            raise ValueError(f"Registry digest mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite frozen registry: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _load_audit_registry(
    registry_path: Path,
    amendment_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    registry = _verified_json(registry_path)
    if amendment_path is None:
        return registry, None
    amendment = _verified_json(amendment_path)
    if str(amendment["audit_registry_sha256"]) != _sha256(registry_path):
        raise ValueError("Target-size amendment names another audit registry.")
    if bool(amendment["test_subset_loaded_before_amendment"]):
        raise ValueError("Target-size amendment was not frozen before target access.")
    if int(registry["partition"]["target_records"]) != int(amendment["original_target_records"]):
        raise ValueError("Target-size amendment does not match the frozen target size.")
    resolved = deepcopy(registry)
    resolved["partition"]["target_records"] = int(amendment["replacement_target_records"])
    resolved["direct_plan"] = amendment["replacement_direct_plan"]
    resolved["pre_audit_target_amendment"] = str(amendment_path)
    resolved["pre_audit_target_amendment_sha256"] = _sha256(amendment_path)
    return resolved, amendment


def _load_subset(registry: dict[str, Any], subset: str) -> tuple[list[str], np.ndarray]:
    if subset not in {"train", "test"}:
        raise ValueError("subset must be 'train' or 'test'.")
    dataset = registry["dataset"]
    frame = fetch_20newsgroups(
        subset=subset,
        categories=list(dataset["categories"]),
        remove=tuple(dataset["remove"]),
        data_home=str(dataset["data_home"]),
        download_if_missing=True,
        shuffle=False,
    )
    return list(frame.data), np.asarray(frame.target, dtype=int)


def _development_data(
    registry: dict[str, Any],
) -> tuple[CountVectorizer, Any, np.ndarray, Any, np.ndarray]:
    texts, labels = _load_subset(registry, "train")
    partition = registry["partition"]
    if len(texts) != int(partition["expected_training_records"]):
        raise ValueError("The training subset has an unexpected record count.")
    rng = np.random.default_rng(int(partition["development_split_seed"]))
    positions = rng.permutation(len(texts))
    source_stop = int(partition["source_records"])
    source_positions = positions[:source_stop]
    planning_positions = positions[source_stop:]
    vectorizer = CountVectorizer(
        max_features=int(registry["vectorizer"]["max_features"]),
        min_df=int(registry["vectorizer"]["min_df"]),
        stop_words=str(registry["vectorizer"]["stop_words"]),
        lowercase=bool(registry["vectorizer"]["lowercase"]),
        dtype=np.int32,
    )
    source_texts = [texts[index] for index in source_positions]
    planning_texts = [texts[index] for index in planning_positions]
    X_source = vectorizer.fit_transform(source_texts)
    X_planning = vectorizer.transform(planning_texts)
    return (
        vectorizer,
        X_source,
        labels[source_positions],
        X_planning,
        labels[planning_positions],
    )


def _losses(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities >= 0.5
    error = (predictions != labels).astype(float)
    brier = np.square(probabilities - labels)
    cost = np.where(
        (labels == 1) & ~predictions,
        1.0,
        np.where((labels == 0) & predictions, 1.0 / 3.0, 0.0),
    )
    return np.column_stack([error, brier, cost])


def _fit_source_model(X_source: Any, y_source: np.ndarray, registry: dict[str, Any]) -> MultinomialNB:
    model = MultinomialNB(alpha=float(registry["source_procedure"]["alpha"]))
    model.fit(X_source, y_source)
    return model


def _generator_state(X_source: Any, y_source: np.ndarray, registry: dict[str, Any]) -> dict[str, Any]:
    smoothing = float(registry["generator_search"]["token_smoothing"])
    token_probabilities: list[np.ndarray] = []
    document_lengths: list[np.ndarray] = []
    for label in (0, 1):
        class_rows = X_source[y_source == label]
        counts = np.asarray(class_rows.sum(axis=0), dtype=float).ravel() + smoothing
        token_probabilities.append(counts / counts.sum())
        lengths = np.asarray(class_rows.sum(axis=1), dtype=int).ravel()
        lower = int(registry["generator_search"]["minimum_document_tokens"])
        upper = int(registry["generator_search"]["maximum_document_tokens"])
        document_lengths.append(np.clip(lengths, lower, upper))
    return {
        "token_probabilities": token_probabilities,
        "document_lengths": document_lengths,
        "features": int(X_source.shape[1]),
    }


def _fit_release(
    state: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    registry: dict[str, Any],
) -> MultinomialNB:
    rng = np.random.default_rng(seed)
    rows = int(config["synthetic_rows"])
    class_rows = [rows // 2, rows - rows // 2]
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    temperature = float(config["temperature"])
    for label in (0, 1):
        base = np.asarray(state["token_probabilities"][label], dtype=float)
        probabilities = np.power(base, 1.0 / temperature)
        probabilities /= probabilities.sum()
        source_lengths = np.asarray(state["document_lengths"][label], dtype=int)
        lengths = rng.choice(source_lengths, size=class_rows[label], replace=True)
        matrix = np.empty((class_rows[label], int(state["features"])), dtype=np.int16)
        for length_index, length in enumerate(lengths):
            matrix[length_index] = rng.multinomial(int(length), probabilities)
        matrices.append(matrix)
        labels.append(np.full(class_rows[label], label, dtype=int))
    synthetic_X = np.vstack(matrices)
    synthetic_y = np.concatenate(labels)
    flip_probability = float(config.get("label_flip_probability", 0.0))
    if flip_probability > 0.0:
        flip = rng.random(rows) < flip_probability
        synthetic_y[flip] = 1 - synthetic_y[flip]
    order = rng.permutation(rows)
    model = MultinomialNB(alpha=float(registry["source_procedure"]["alpha"]))
    model.fit(synthetic_X[order], synthetic_y[order])
    return model


def _primary_limits(source_risks: np.ndarray, registry: dict[str, Any]) -> np.ndarray:
    policy = registry["claims"]["specification_envelope"]["primary"]
    budgets = np.asarray(policy["degradation_budgets"], dtype=float)
    ceilings = np.asarray(policy["absolute_ceilings"], dtype=float)
    return np.minimum(source_risks + budgets, ceilings)


def run_pilot(registry_path: Path, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = _verified_json(registry_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output_root}")
    output_root.mkdir(parents=True)
    vectorizer, X_source, y_source, X_planning, y_planning = _development_data(registry)
    source_model = _fit_source_model(X_source, y_source, registry)
    source_losses = _losses(y_planning, source_model.predict_proba(X_planning)[:, 1])
    source_risks = source_losses.mean(axis=0)
    limits = _primary_limits(source_risks, registry)
    slacks = np.asarray(registry["claims"]["score_slacks"], dtype=float)
    state = _generator_state(X_source, y_source, registry)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    pilot_releases = int(registry["generator_search"]["pilot_releases_per_configuration"])
    seed_base = int(registry["generator_search"]["pilot_seed_base"])
    for configuration_index, config in enumerate(registry["generator_search"]["configurations"]):
        for release_index in range(pilot_releases):
            seed = seed_base + 1000 * configuration_index + release_index
            model = _fit_release(state, config, seed, registry)
            release_losses = _losses(y_planning, model.predict_proba(X_planning)[:, 1])
            risks = release_losses.mean(axis=0)
            row: dict[str, Any] = {
                "Configuration": str(config["id"]),
                "PilotRelease": release_index,
                "Seed": seed,
                "MinimumValidityMargin": float(np.min(limits - risks)),
                "MinimumScoreMargin": float(np.min(limits - slacks - risks)),
            }
            for loss_index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(risks[loss_index])
                row[f"ScoreMargin_{name}"] = float(limits[loss_index] - slacks[loss_index] - risks[loss_index])
            rows.append(row)
    frame = pd.DataFrame(rows)
    risks_path = output_root / "pilot_release_risks.csv"
    frame.to_csv(risks_path, index=False)
    summary = {
        "partition_registry": str(registry_path),
        "partition_registry_sha256": _sha256(registry_path),
        "training_records": int(X_source.shape[0] + X_planning.shape[0]),
        "source_records": int(X_source.shape[0]),
        "planning_records": int(X_planning.shape[0]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "source_planning_risks": dict(zip(LOSS_NAMES, source_risks.tolist(), strict=True)),
        "primary_limits": dict(zip(LOSS_NAMES, limits.tolist(), strict=True)),
        "score_cutoffs": dict(zip(LOSS_NAMES, (limits - slacks).tolist(), strict=True)),
        "pilot_release_risks_sha256": _sha256(risks_path),
        "elapsed_seconds": time.perf_counter() - started,
        "target_subset_loaded": False,
    }
    (output_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return frame, summary


def _select_roles(frame: pd.DataFrame, target_margin: float) -> dict[str, str]:
    grouped = frame.groupby("Configuration")["MinimumScoreMargin"]
    statistics = pd.DataFrame(
        {
            "q10": grouped.quantile(0.10),
            "median": grouped.median(),
            "q90": grouped.quantile(0.90),
        }
    ).sort_index()
    high = str(statistics["q10"].idxmax())
    degraded = str(statistics["q90"].idxmin())
    candidates = statistics.drop(index=[high, degraded], errors="ignore")
    positive = candidates[candidates["median"] >= 0.0]
    pool = positive if not positive.empty else candidates
    if pool.empty:
        raise ValueError("The registered search cannot select three distinct roles.")
    moderate = str((pool["median"] - target_margin).abs().idxmin())
    return {
        "high_signal": high,
        "moderate_evidence": moderate,
        "degraded": degraded,
    }


def freeze_audit(
    partition_registry_path: Path,
    pilot_root: Path,
    audit_registry_path: Path,
) -> dict[str, Any]:
    registry = _verified_json(partition_registry_path)
    pilot_path = pilot_root / "pilot_release_risks.csv"
    summary_path = pilot_root / "pilot_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if _sha256(pilot_path) != str(summary["pilot_release_risks_sha256"]):
        raise ValueError("Pilot release risks changed before the audit freeze.")
    frame = pd.read_csv(pilot_path)
    roles = _select_roles(
        frame,
        float(registry["generator_search"]["moderate_target_score_margin"]),
    )
    configurations = {str(item["id"]): item for item in registry["generator_search"]["configurations"]}
    selected = {role: configurations[identifier] for role, identifier in roles.items()}
    release_counts = registry["claims"]["audit_releases"]
    seed_bases = registry["claims"]["release_seed_bases"]
    release_seeds = {
        role: [int(seed_bases[role]) + index for index in range(int(release_counts[role]))] for role in selected
    }
    limits = np.asarray(list(summary["primary_limits"].values()), dtype=float)
    slacks = np.asarray(registry["claims"]["score_slacks"], dtype=float)
    target_records = int(registry["partition"]["target_records"])
    mechanisms = int(registry["claims"]["registered_configurations"])
    plan = plan_conditional_shared_target(
        target_records=target_records,
        minimum_reliability=float(registry["claims"]["minimum_reliability"]),
        tolerances=limits,
        slacks=slacks,
        error_rate=float(registry["claims"]["total_alpha"]),
        target_error_fraction=float(registry["claims"]["direct_target_error_fraction"]),
        mechanisms=mechanisms,
    )
    source_risks = np.asarray(list(summary["source_planning_risks"].values()), dtype=float)
    envelope: dict[str, dict[str, float]] = {}
    for name, policy in registry["claims"]["specification_envelope"].items():
        policy_limits = np.minimum(
            source_risks + np.asarray(policy["degradation_budgets"], dtype=float),
            np.asarray(policy["absolute_ceilings"], dtype=float),
        )
        envelope[name] = dict(zip(LOSS_NAMES, policy_limits.tolist(), strict=True))
    requirements = [
        {
            "name": name,
            "limit": float(limits[index]),
            "lower": 0.0,
            "upper": 1.0,
            "score_slack": float(slacks[index]),
            "score_cutoff": float(limits[index] - slacks[index]),
            "policy": "source planning risk plus the registered primary degradation budget, capped by the absolute ceiling",
        }
        for index, name in enumerate(LOSS_NAMES)
    ]
    frozen = {
        "registry_version": "1.0",
        "frozen_on": "2026-08-17",
        "analysis_status": "Prospective non-tabular audit frozen after training-only development and before loading the official test subset.",
        "partition_registry": str(partition_registry_path),
        "partition_registry_sha256": _sha256(partition_registry_path),
        "pilot_summary": str(summary_path),
        "pilot_summary_sha256": _sha256(summary_path),
        "pilot_release_risks": str(pilot_path),
        "pilot_release_risks_sha256": _sha256(pilot_path),
        "dataset": registry["dataset"],
        "partition": registry["partition"],
        "vectorizer": registry["vectorizer"],
        "source_procedure": registry["source_procedure"],
        "generator": {
            "name": registry["generator_search"]["name"],
            "scope": registry["generator_search"]["scope"],
            "selected_configurations": selected,
            "release_seeds": release_seeds,
            "selection_rule": registry["generator_search"]["selection_rule"],
            "token_smoothing": registry["generator_search"]["token_smoothing"],
            "minimum_document_tokens": registry["generator_search"]["minimum_document_tokens"],
            "maximum_document_tokens": registry["generator_search"]["maximum_document_tokens"],
        },
        "requirements": requirements,
        "specification_envelope": envelope,
        "minimum_reliability": float(registry["claims"]["minimum_reliability"]),
        "total_alpha": float(registry["claims"]["total_alpha"]),
        "named_release_error_share": float(registry["claims"]["named_release_error_share"]),
        "direct_target_error_fraction": float(registry["claims"]["direct_target_error_fraction"]),
        "registered_mechanisms": mechanisms,
        "smooth_ramp_fraction_of_cutoff": float(registry["claims"]["smooth_ramp_fraction_of_cutoff"]),
        "headline_rule": registry["claims"]["headline_rule"],
        "direct_plan": {
            "minimum_target_records": plan.minimum_target_records,
            "minimum_releases": plan.minimum_releases,
            "contamination_allowance": plan.target_contamination_allowance,
        },
        "test_subset_loaded_before_freeze": False,
    }
    _write_hashed_json(audit_registry_path, frozen)
    return frozen


def generate_releases(
    audit_registry_path: Path,
    output_root: Path,
    amendment_path: Path | None = None,
) -> dict[str, Any]:
    registry, amendment = _load_audit_registry(audit_registry_path, amendment_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite generated releases: {output_root}")
    output_root.mkdir(parents=True)
    vectorizer, X_source, y_source, _, _ = _development_data(registry)
    source_model = _fit_source_model(X_source, y_source, registry)
    search_view = {
        "generator_search": {
            "token_smoothing": registry["generator"]["token_smoothing"],
            "minimum_document_tokens": registry["generator"]["minimum_document_tokens"],
            "maximum_document_tokens": registry["generator"]["maximum_document_tokens"],
        }
    }
    state = _generator_state(X_source, y_source, search_view)
    release_models: dict[str, list[MultinomialNB]] = {}
    timings: dict[str, float] = {}
    for role, config in registry["generator"]["selected_configurations"].items():
        started = time.perf_counter()
        release_models[role] = [
            _fit_release(state, config, int(seed), registry) for seed in registry["generator"]["release_seeds"][role]
        ]
        timings[role] = time.perf_counter() - started
    bundle_path = output_root / "release_procedures.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "source_model": source_model,
            "release_models": release_models,
        },
        bundle_path,
        compress=3,
    )
    result = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "pre_audit_target_amendment": (None if amendment_path is None else str(amendment_path)),
        "pre_audit_target_amendment_sha256": (None if amendment_path is None else _sha256(amendment_path)),
        "release_bundle": str(bundle_path),
        "release_bundle_sha256": _sha256(bundle_path),
        "generation_seconds": timings,
        "test_subset_loaded": False,
    }
    (output_root / "generation_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _requirements(registry: dict[str, Any]) -> list[RiskRequirement]:
    return [RiskRequirement(str(item["name"]), float(item["limit"]), 0.0, 1.0) for item in registry["requirements"]]


def audit_releases(
    audit_registry_path: Path,
    generation_root: Path,
    output_root: Path,
    amendment_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry, amendment = _load_audit_registry(audit_registry_path, amendment_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_root}")
    output_root.mkdir(parents=True)
    generation_manifest_path = generation_root / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    expected_amendment_hash = None if amendment_path is None else _sha256(amendment_path)
    if generation_manifest.get("pre_audit_target_amendment_sha256") != expected_amendment_hash:
        raise ValueError("Generation manifest and audit use different target amendments.")
    bundle_path = Path(generation_manifest["release_bundle"])
    if _sha256(bundle_path) != str(generation_manifest["release_bundle_sha256"]):
        raise ValueError("Release bundle changed before target access.")
    bundle = joblib.load(bundle_path)

    target_started = time.perf_counter()
    test_texts, test_labels = _load_subset(registry, "test")
    rng = np.random.default_rng(int(registry["partition"]["target_sampling_seed"]))
    positions = rng.choice(
        len(test_texts),
        size=int(registry["partition"]["target_records"]),
        replace=True,
    )
    target_texts = [test_texts[index] for index in positions]
    y_target = test_labels[positions]
    X_target = bundle["vectorizer"].transform(target_texts)
    source_losses = _losses(
        y_target,
        bundle["source_model"].predict_proba(X_target)[:, 1],
    )
    source_risks = source_losses.mean(axis=0)
    tolerances = np.asarray([float(item["limit"]) for item in registry["requirements"]])
    slacks = np.asarray([float(item["score_slack"]) for item in registry["requirements"]])
    mechanisms = int(registry["registered_mechanisms"])
    release_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for role, models in bundle["release_models"].items():
        release_means = np.empty((len(models), len(LOSS_NAMES)), dtype=float)
        candidate_losses: dict[str, dict[str, np.ndarray]] = {}
        release_map: dict[str, str] = {}
        for release_index, model in enumerate(models):
            losses = _losses(y_target, model.predict_proba(X_target)[:, 1])
            release_means[release_index] = losses.mean(axis=0)
            candidate = f"{role}_{release_index:04d}"
            candidate_losses[candidate] = {name: losses[:, loss_index] for loss_index, name in enumerate(LOSS_NAMES)}
            release_map[candidate] = role
            row: dict[str, Any] = {
                "ConfigurationRole": role,
                "Release": release_index,
            }
            for loss_index, name in enumerate(LOSS_NAMES):
                row[f"Risk_{name}"] = float(release_means[release_index, loss_index])
                row[f"Margin_{name}"] = float(tolerances[loss_index] - release_means[release_index, loss_index])
            release_rows.append(row)
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
            target_records=len(y_target),
            tolerances=tolerances,
            slacks=slacks,
            error_rate=float(registry["total_alpha"]),
            target_error_fraction=float(registry["direct_target_error_fraction"]),
            mechanisms=mechanisms,
        )
        ramp_fraction = float(registry["smooth_ramp_fraction_of_cutoff"])
        ramp_widths = ramp_fraction * (tolerances - slacks)
        smooth = shared_target_smooth_conditional_mean_lower_bound(
            release_means,
            target_records=len(y_target),
            tolerances=tolerances,
            slacks=slacks,
            ramp_widths=ramp_widths,
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
                "NamedRecognized": int(named_row["ValidatedReleases"]),
                "NamedViolations": int(named_row["DetectedReleaseViolations"]),
                "NamedReliabilityLCB": float(named_row["ReliabilityLCB"]),
                "NamedReliabilityUCB": float(named_row["ReliabilityUCB"]),
                "NamedDecision": str(named_row["Status"]),
                "DirectFavorableScores": int(np.rint(direct.conditional_score_mean * len(models))),
                "DirectScoreLCB": direct.conditional_score_lower_bound,
                "DirectContaminationAllowance": direct.target_contamination_allowance,
                "DirectReliabilityLCB": direct.reliability_lower_bound,
                "DirectDecision": (
                    "Mechanism validated"
                    if direct.reliability_lower_bound > float(registry["minimum_reliability"])
                    else "Unresolved"
                ),
                "SmoothScoreMean": smooth.conditional_score_mean,
                "SmoothContaminationAllowance": smooth.target_contamination_allowance,
                "SmoothReliabilityLCB": smooth.reliability_lower_bound,
                "SmoothDecision": (
                    "Mechanism validated"
                    if smooth.reliability_lower_bound > float(registry["minimum_reliability"])
                    else "Unresolved"
                ),
            }
        )
        named.release_audit.candidate_summary.to_csv(
            output_root / f"{role}_named_release_summary.csv",
            index=False,
        )

    release_frame = pd.DataFrame(release_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    release_path = output_root / "release_risks.csv"
    mechanism_path = output_root / "mechanism_summary.csv"
    release_frame.to_csv(release_path, index=False)
    mechanism_frame.to_csv(mechanism_path, index=False)
    position_path = output_root / "target_positions.npy"
    np.save(position_path, positions)
    result = {
        "audit_registry": str(audit_registry_path),
        "audit_registry_sha256": _sha256(audit_registry_path),
        "pre_audit_target_amendment": (None if amendment_path is None else str(amendment_path)),
        "pre_audit_target_amendment_sha256": expected_amendment_hash,
        "generation_manifest": str(generation_manifest_path),
        "generation_manifest_sha256": _sha256(generation_manifest_path),
        "target_population_records": len(test_texts),
        "target_records": len(y_target),
        "target_sampling": "iid with replacement from the untouched official test subset",
        "target_positions_sha256": _sha256(position_path),
        "source_target_risks": dict(zip(LOSS_NAMES, source_risks.tolist(), strict=True)),
        "release_risks_sha256": _sha256(release_path),
        "mechanism_summary_sha256": _sha256(mechanism_path),
        "target_evaluation_seconds": time.perf_counter() - target_started,
        "target_subset_first_loaded_after_audit_freeze": True,
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return mechanism_frame, release_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "freeze", "generate", "audit"))
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_text_direct_partition.json",
    )
    parser.add_argument(
        "--pilot-root",
        default="outputs/proxyguard_text_direct_pilot",
    )
    parser.add_argument(
        "--audit-registry",
        default="registries/proxyguard_text_direct_audit.json",
    )
    parser.add_argument(
        "--amendment",
        default="registries/proxyguard_text_direct_target_amendment.json",
    )
    parser.add_argument(
        "--generation-root",
        default="outputs/proxyguard_text_direct_generation",
    )
    parser.add_argument(
        "--audit-root",
        default="outputs/proxyguard_text_direct_audit",
    )
    args = parser.parse_args()
    if args.stage == "pilot":
        frame, summary = run_pilot(Path(args.registry), Path(args.pilot_root))
        print(frame.groupby("Configuration")["MinimumScoreMargin"].describe())
        print(json.dumps(summary, indent=2))
    elif args.stage == "freeze":
        frozen = freeze_audit(
            Path(args.registry),
            Path(args.pilot_root),
            Path(args.audit_registry),
        )
        print(json.dumps(frozen, indent=2))
    elif args.stage == "generate":
        manifest = generate_releases(
            Path(args.audit_registry),
            Path(args.generation_root),
            Path(args.amendment),
        )
        print(json.dumps(manifest, indent=2))
    else:
        mechanisms, _ = audit_releases(
            Path(args.audit_registry),
            Path(args.generation_root),
            Path(args.audit_root),
            Path(args.amendment),
        )
        print(mechanisms.to_string(index=False))


if __name__ == "__main__":
    main()
