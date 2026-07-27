from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config  # noqa: E402
from risk_models.dataset import load_dataset  # noqa: E402
from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
    paired_prediction_losses,
)


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_audit: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_audit: pd.Series
    train_positions: np.ndarray
    validation_positions: np.ndarray
    audit_positions: np.ndarray


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_split(X: pd.DataFrame, y: pd.Series, seed: int) -> DatasetSplit:
    positions = np.arange(len(y))
    train_positions, holdout_positions = train_test_split(
        positions,
        test_size=0.40,
        random_state=seed,
        stratify=y,
    )
    validation_positions, audit_positions = train_test_split(
        holdout_positions,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y.iloc[holdout_positions],
    )
    return DatasetSplit(
        X_train=X.iloc[train_positions].reset_index(drop=True),
        X_validation=X.iloc[validation_positions].reset_index(drop=True),
        X_audit=X.iloc[audit_positions].reset_index(drop=True),
        y_train=y.iloc[train_positions].reset_index(drop=True),
        y_validation=y.iloc[validation_positions].reset_index(drop=True),
        y_audit=y.iloc[audit_positions].reset_index(drop=True),
        train_positions=np.asarray(train_positions, dtype=int),
        validation_positions=np.asarray(validation_positions, dtype=int),
        audit_positions=np.asarray(audit_positions, dtype=int),
    )


def build_classifier_library(seed: int) -> dict[str, Pipeline]:
    return {
        "Logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Histogram boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=31,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def build_classifier(seed: int) -> Pipeline:
    return build_classifier_library(seed)["Logistic"]


def choose_cost_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
    false_negative_cost: float,
) -> float:
    thresholds = np.linspace(0.0, 1.0, 501)
    labels = y_true.to_numpy(dtype=int)
    prediction = probability[:, None] >= thresholds[None, :]
    false_negative = ((labels[:, None] == 1) & ~prediction).mean(axis=0)
    false_positive = ((labels[:, None] == 0) & prediction).mean(axis=0)
    cost = false_negative_cost * false_negative + false_positive
    return float(thresholds[int(np.argmin(cost))])


def _public_domain_constraints(
    public_schema: pd.DataFrame,
    target_column: str,
    bins: int,
) -> dict[str, Any]:
    try:
        from snsynth.transform import BinTransformer, LabelTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "SmartNoise Synth is required. Run with `uv run --with smartnoise-synth`."
        ) from exc

    constraints: dict[str, Any] = {}
    for column in public_schema.columns:
        values = pd.to_numeric(public_schema[column], errors="coerce")
        unique_count = int(values.nunique(dropna=True))
        if column == target_column or unique_count <= 16:
            constraints[column] = LabelTransformer(nullable=False)
            continue
        lower = float(values.min())
        upper = float(values.max())
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"Non-finite public bounds for {column!r}.")
        if lower == upper:
            constraints[column] = LabelTransformer(nullable=False)
        else:
            constraints[column] = BinTransformer(
                bins=bins,
                lower=lower,
                upper=upper,
                nullable=False,
            )
    return constraints


def fit_aim_release(
    train_table: pd.DataFrame,
    public_schema: pd.DataFrame,
    epsilon: float,
    delta: float,
    seed: int,
    bins: int = 10,
    max_model_size: int = 40,
    num_marginals: int = 40,
) -> pd.DataFrame:
    try:
        from snsynth import Synthesizer
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "SmartNoise Synth is required. Run with `uv run --with smartnoise-synth`."
        ) from exc

    np.random.seed(seed)
    target_column = "__target__"
    constraints = _public_domain_constraints(public_schema, target_column, bins)
    synthesizer = Synthesizer.create(
        "aim",
        epsilon=epsilon,
        delta=delta,
        degree=2,
        max_model_size=max_model_size,
        num_marginals=num_marginals,
        max_cells=5_000,
    )
    synthesizer.fit(train_table, transformer=constraints)
    release = synthesizer.sample(len(train_table))
    return pd.DataFrame(release, columns=train_table.columns)


def _mean_normalized_cost(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
    false_negative_cost: float,
) -> float:
    labels = y_true.to_numpy(dtype=int)
    prediction = probability >= threshold
    cost = (
        false_negative_cost * ((labels == 1) & ~prediction)
        + ((labels == 0) & prediction)
    ) / max(false_negative_cost, 1.0)
    return float(cost.mean())


def fit_selected_procedure(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
    false_negative_cost: float = 5.0,
) -> tuple[str, Pipeline, float, np.ndarray]:
    candidates: list[tuple[float, str, Pipeline, float, np.ndarray]] = []
    for name, model in build_classifier_library(seed).items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_validation)[:, 1]
        threshold = choose_cost_threshold(
            y_validation,
            probability,
            false_negative_cost=false_negative_cost,
        )
        validation_cost = _mean_normalized_cost(
            y_validation,
            probability,
            threshold,
            false_negative_cost,
        )
        candidates.append((validation_cost, name, model, threshold, probability))
    _, name, model, threshold, probability = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return name, model, threshold, probability


def audit_aim_candidates(
    dataset_names: list[str],
    epsilon_values: list[float],
    alpha: float,
    seed: int,
    delta: float,
    bins: int,
    max_model_size: int,
    num_marginals: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    paired_losses: dict[str, pd.DataFrame] = {}

    for dataset_offset, dataset_name in enumerate(dataset_names):
        bundle = load_dataset(get_dataset_config(dataset_name))
        X = bundle["X"].reset_index(drop=True).astype(float)
        y = pd.Series(bundle["y"]).reset_index(drop=True).astype(int)
        split = make_split(X, y, seed + dataset_offset * 101)

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
            seed=seed + dataset_offset,
        )
        source_audit_probability = source_model.predict_proba(split.X_audit)[:, 1]

        train_table = split.X_train.copy()
        train_table["__target__"] = split.y_train.to_numpy()
        public_schema = X.copy()
        public_schema["__target__"] = y.to_numpy()

        for epsilon_index, epsilon in enumerate(epsilon_values):
            candidate = f"{dataset_name}::aim_e{epsilon:g}"
            release_seed = seed + dataset_offset * 10_000 + epsilon_index * 1_000
            synthetic = fit_aim_release(
                train_table=train_table,
                public_schema=public_schema,
                epsilon=epsilon,
                delta=delta,
                seed=release_seed,
                bins=bins,
                max_model_size=max_model_size,
                num_marginals=num_marginals,
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
            losses.insert(0, "Candidate", candidate)
            paired_losses[candidate] = losses
            candidate_regrets[candidate] = {
                "Brier": losses["brier_regret"].to_numpy(),
                "Clipped log loss": losses["logloss_regret"].to_numpy(),
                "Cost5x": losses["cost5x_regret"].to_numpy(),
            }
            diagnostics.append(
                {
                    "Candidate": candidate,
                    "Dataset": dataset_name,
                    "Mechanism": "AIM",
                    "Epsilon": epsilon,
                    "Delta": delta,
                    "AuditN": len(split.y_audit),
                    "SourceModel": source_model_name,
                    "ProxyModel": proxy_model_name,
                    "SourceAUC": roc_auc_score(split.y_audit, source_audit_probability),
                    "ProxyAUC": roc_auc_score(split.y_audit, proxy_audit_probability),
                    "AUCChange": roc_auc_score(split.y_audit, proxy_audit_probability)
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

    requirements = [
        RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
        RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
        RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
    ]
    result = audit_proxy_candidates(
        candidate_regrets,
        requirements=requirements,
        alpha=alpha,
        bound_method="empirical_bernstein",
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    candidate_summary = result.candidate_summary.merge(
        diagnostic_frame,
        on="Candidate",
        how="left",
        validate="one_to_one",
    )
    return candidate_summary, result.requirement_detail, diagnostic_frame, paired_losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a risk-controlled audit of SmartNoise AIM proxy releases."
    )
    parser.add_argument(
        "--datasets",
        default="australian_credit,german_credit,taiwan_default",
    )
    parser.add_argument("--epsilons", default="1,5,10")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-model-size", type=int, default=40)
    parser.add_argument("--num-marginals", type=int, default=40)
    parser.add_argument("--output-root", default="outputs/proxyguard_aim_audit")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_summary, requirement_detail, diagnostics, paired_losses = audit_aim_candidates(
        dataset_names=parse_csv(args.datasets),
        epsilon_values=[float(value) for value in parse_csv(args.epsilons)],
        alpha=args.alpha,
        seed=args.seed,
        delta=args.delta,
        bins=args.bins,
        max_model_size=args.max_model_size,
        num_marginals=args.num_marginals,
    )
    candidate_summary.to_csv(output_root / "candidate_summary.csv", index=False)
    requirement_detail.to_csv(output_root / "requirement_detail.csv", index=False)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    pd.concat(paired_losses.values(), ignore_index=True).to_csv(
        output_root / "paired_losses.csv",
        index=False,
    )
    settings = {
        "datasets": parse_csv(args.datasets),
        "epsilons": [float(value) for value in parse_csv(args.epsilons)],
        "alpha": args.alpha,
        "delta": args.delta,
        "seed": args.seed,
        "bins": args.bins,
        "max_model_size": args.max_model_size,
        "num_marginals": args.num_marginals,
        "public_domain_note": (
            "Feature bounds and category domains are treated as public benchmark schema. "
            "The audit guarantee concerns downstream transfer, not differential privacy."
        ),
    }
    (output_root / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )
    print(
        candidate_summary[
            ["Candidate", "AuditN", "Status", "AUCChange", "SourceCost5x", "ProxyCost5x"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
