from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config  # noqa: E402
from risk_models.dataset import load_dataset  # noqa: E402
from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_mechanisms,
    paired_prediction_losses,
)
from scripts.proxyguard.run_proxyguard_aim_audit import (  # noqa: E402
    _mean_normalized_cost,
    fit_selected_procedure,
    make_split,
)


def bootstrap_release(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X and y must have the same positive length.")
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(y), size=len(y))
    return (
        X.iloc[positions].reset_index(drop=True),
        y.iloc[positions].reset_index(drop=True),
    )


def _requirements(registry: dict) -> list[RiskRequirement]:
    return [
        RiskRequirement(
            str(requirement["name"]),
            tolerance=float(requirement["tolerance"]),
            lower=float(requirement["lower"]),
            upper=float(requirement["upper"]),
            estimand=str(requirement.get("estimand", "relative_regret")),
        )
        for requirement in registry["requirements"]
    ]


def _registered_audit_values(
    losses: pd.DataFrame,
    registry: dict,
) -> dict[str, np.ndarray]:
    default_columns = {
        "Brier": "brier_regret",
        "Clipped log loss": "logloss_regret",
        "Cost5x": "cost5x_regret",
    }
    values: dict[str, np.ndarray] = {}
    for requirement in registry["requirements"]:
        name = str(requirement["name"])
        column = str(requirement.get("value_column", default_columns.get(name, "")))
        if not column or column not in losses:
            raise ValueError(
                f"Requirement {name!r} must name an existing value_column; "
                f"got {column!r}."
            )
        values[name] = losses[column].to_numpy()
    return values


def _constant_policy_threshold(
    labels: pd.Series,
    probability: float,
) -> float:
    probabilities = np.full(len(labels), probability, dtype=float)
    candidates = (0.0, 1.0)
    return min(
        candidates,
        key=lambda threshold: (
            _mean_normalized_cost(
                labels,
                probabilities,
                threshold,
                5.0,
            ),
            threshold,
        ),
    )


def _absolute_risk_baselines(
    split,
    source_audit_probability: np.ndarray,
    source_threshold: float,
) -> pd.DataFrame:
    training_prevalence = float(split.y_train.mean())
    baseline_specs = [
        ("Constant 0.5", 0.5),
        ("Training prevalence", training_prevalence),
    ]
    rows: list[dict[str, float | str]] = []

    source_losses = paired_prediction_losses(
        y_true=split.y_audit,
        source_probability=source_audit_probability,
        proxy_probability=source_audit_probability,
        source_thresholds={5.0: source_threshold},
        proxy_thresholds={5.0: source_threshold},
    )
    rows.append(
        {
            "Baseline": "Source procedure",
            "Probability": np.nan,
            "Threshold": source_threshold,
            "BrierRisk": float(source_losses["brier_proxy"].mean()),
            "ClippedLogLossRisk": float(source_losses["logloss_proxy"].mean()),
            "Cost5xRisk": float(source_losses["cost5x_proxy"].mean()),
        }
    )

    for name, probability in baseline_specs:
        threshold = _constant_policy_threshold(
            split.y_validation,
            probability,
        )
        audit_probability = np.full(
            len(split.y_audit),
            probability,
            dtype=float,
        )
        losses = paired_prediction_losses(
            y_true=split.y_audit,
            source_probability=source_audit_probability,
            proxy_probability=audit_probability,
            source_thresholds={5.0: source_threshold},
            proxy_thresholds={5.0: threshold},
        )
        rows.append(
            {
                "Baseline": name,
                "Probability": probability,
                "Threshold": threshold,
                "BrierRisk": float(losses["brier_proxy"].mean()),
                "ClippedLogLossRisk": float(losses["logloss_proxy"].mean()),
                "Cost5xRisk": float(losses["cost5x_proxy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_bootstrap_mechanism(
    registry: dict,
    release_limit: int | None = None,
    mechanism_count_mode: str = "holm",
    collective_dependence_verified: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_name = str(registry["dataset"])
    bundle = load_dataset(get_dataset_config(dataset_name))
    X = bundle["X"].reset_index(drop=True).astype(float)
    y = pd.Series(bundle["y"]).reset_index(drop=True).astype(int)
    split_seed = int(registry["split_seed"])
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

    registered_releases = int(registry["releases"])
    releases = registered_releases if release_limit is None else int(release_limit)
    if not 1 <= releases <= registered_releases:
        raise ValueError("release_limit must lie within the registered release count.")

    mechanism = f"{dataset_name}::bootstrap"
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    release_to_mechanism: dict[str, str] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    loss_frames: list[pd.DataFrame] = []
    for release_index in range(releases):
        release_number = release_index + 1
        release_seed = int(registry["release_seed_start"]) + release_index
        candidate = f"{mechanism}::release_{release_number:02d}"
        bootstrap_X, bootstrap_y = bootstrap_release(
            split.X_train,
            split.y_train,
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
        losses.insert(0, "Release", release_number)
        losses.insert(0, "Candidate", candidate)
        loss_frames.append(losses)
        candidate_regrets[candidate] = _registered_audit_values(losses, registry)
        release_to_mechanism[candidate] = mechanism
        source_auc = roc_auc_score(split.y_audit, source_audit_probability)
        proxy_auc = roc_auc_score(split.y_audit, proxy_audit_probability)
        diagnostics.append(
            {
                "Candidate": candidate,
                "Mechanism": mechanism,
                "Dataset": dataset_name,
                "Release": release_number,
                "ReleaseSeed": release_seed,
                "AuditN": len(split.y_audit),
                "SourceModel": source_model_name,
                "ProxyModel": proxy_model_name,
                "SourceAUC": source_auc,
                "ProxyAUC": proxy_auc,
                "AUCChange": proxy_auc - source_auc,
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
                "BootstrapPositiveRate": float(bootstrap_y.mean()),
            }
        )
        print(
            f"completed bootstrap release {release_number}/{releases}: "
            f"{proxy_model_name}",
            flush=True,
        )

    audit = audit_proxy_mechanisms(
        candidate_regrets,
        release_to_mechanism,
        requirements=_requirements(registry),
        minimum_reliability=float(registry["minimum_reliability"]),
        total_alpha=float(registry["total_alpha"]),
        release_error_share=float(registry["release_error_share"]),
        mechanism_count_mode=mechanism_count_mode,
        collective_dependence_verified=collective_dependence_verified,
        bound_method=str(registry["bound_method"]),
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["Cost5xChange"] = (
        diagnostics_frame["ProxyCost5x"] - diagnostics_frame["SourceCost5x"]
    )
    release_summary = audit.release_audit.candidate_summary.merge(
        diagnostics_frame,
        on=["Candidate", "Mechanism"],
        how="left",
        validate="one_to_one",
    )
    mechanism_summary = audit.mechanism_summary.copy()
    mechanism_summary["MeanAUCChange"] = diagnostics_frame["AUCChange"].mean()
    mechanism_summary["SDAUCChange"] = diagnostics_frame["AUCChange"].std()
    mechanism_summary["MeanCost5xChange"] = diagnostics_frame["Cost5xChange"].mean()
    mechanism_summary["SDCost5xChange"] = diagnostics_frame["Cost5xChange"].std()
    baselines = _absolute_risk_baselines(
        split,
        source_audit_probability,
        source_threshold,
    )
    return (
        release_summary,
        audit.release_audit.requirement_detail,
        mechanism_summary,
        pd.concat(loss_frames, ignore_index=True),
        baselines,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered bootstrap release-mechanism fidelity control."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_bootstrap_mechanism_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_bootstrap_mechanism",
    )
    parser.add_argument(
        "--release-limit",
        type=int,
        help="Run only the first registered releases for a smoke test.",
    )
    parser.add_argument(
        "--mechanism-count-mode",
        choices=("holm", "simes"),
        default="holm",
        help="Aggregate release evidence by release-level Holm certification or Simes count.",
    )
    parser.add_argument(
        "--collective-dependence-verified",
        action="store_true",
        help=(
            "Assert that Simes mode has independent audit batches or a "
            "registered PRDS justification."
        ),
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    (
        release_summary,
        detail,
        mechanism_summary,
        losses,
        baselines,
    ) = run_bootstrap_mechanism(
        registry,
        release_limit=args.release_limit,
        mechanism_count_mode=args.mechanism_count_mode,
        collective_dependence_verified=args.collective_dependence_verified,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    release_summary.to_csv(output_root / "release_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    mechanism_summary.to_csv(output_root / "mechanism_summary.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)
    baselines.to_csv(output_root / "absolute_risk_baselines.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registry_sha256_file": str(registry_path.with_suffix(".sha256")),
                "release_limit": args.release_limit,
                "mechanism_count_mode": args.mechanism_count_mode,
                "collective_dependence_verified": args.collective_dependence_verified,
                "privacy_scope": registry["privacy_scope"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(
        mechanism_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
