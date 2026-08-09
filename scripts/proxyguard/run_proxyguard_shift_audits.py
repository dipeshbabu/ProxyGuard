from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.tabular.metrics import (  # noqa: E402
    apply_calibrator,
    best_cost_threshold_from_val,
    fit_calibrator,
)
from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
    paired_prediction_losses,
)
from scripts.proxyguard.run_temporal_proxy_audit import (  # noqa: E402
    campaign_boundaries,
    make_model,
    split_train_validation,
)

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

HEART_COLUMNS = [
    "age",
    "sex",
    "chest_pain",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "st_slope",
    "major_vessels",
    "thal",
    "diagnosis",
]

REQUIREMENTS = [
    RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
]


@dataclass(frozen=True)
class ProcedureResult:
    model_name: str
    threshold: float
    validation_cost: float
    target_probability: np.ndarray


def _random_train_validation(
    frame: pd.DataFrame,
    labels: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = np.arange(len(frame))
    train_positions, validation_positions = train_test_split(
        positions,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    return (
        frame.iloc[train_positions].copy(),
        frame.iloc[validation_positions].copy(),
    )


def fit_selected_procedure(
    train_period: pd.DataFrame,
    target_features: pd.DataFrame,
    target_column: str,
    label_function: Callable[[pd.Series], pd.Series],
    seed: int,
    ordered_validation: bool,
) -> ProcedureResult:
    all_labels = label_function(train_period[target_column]).astype(int)
    if ordered_validation:
        train, validation = split_train_validation(train_period)
    else:
        train, validation = _random_train_validation(
            train_period,
            all_labels,
            seed,
        )
    x_train = train.drop(columns=[target_column])
    y_train = label_function(train[target_column]).astype(int)
    x_validation = validation.drop(columns=[target_column])
    y_validation = label_function(validation[target_column]).astype(int)

    candidates: list[ProcedureResult] = []
    for offset, model_name in enumerate(("logreg", "histgb")):
        model = make_model(model_name, x_train, seed + offset)
        model.fit(x_train, y_train)
        raw_validation = model.predict_proba(x_validation)[:, 1]
        raw_target = model.predict_proba(target_features)[:, 1]
        calibrator = fit_calibrator(
            raw_validation,
            y_validation,
            method="temperature",
        )
        validation_probability = apply_calibrator(calibrator, raw_validation)
        target_probability = apply_calibrator(calibrator, raw_target)
        threshold, validation_cost = best_cost_threshold_from_val(
            y_validation,
            validation_probability,
            fn_cost=5.0,
            fp_cost=1.0,
        )
        candidates.append(
            ProcedureResult(
                model_name=model_name,
                threshold=float(threshold),
                validation_cost=float(validation_cost),
                target_probability=np.asarray(target_probability, dtype=float),
            )
        )
    return min(candidates, key=lambda result: (result.validation_cost, result.model_name))


def _normalized_cost(
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> float:
    truth = labels.to_numpy(dtype=int)
    prediction = probability >= threshold
    return float(
        (
            5.0 * ((truth == 1) & ~prediction)
            + ((truth == 0) & prediction)
        ).mean()
        / 5.0
    )


def _audit_pair(
    name: str,
    source_train: pd.DataFrame,
    proxy_train: pd.DataFrame,
    target: pd.DataFrame,
    source_features: list[str],
    proxy_features: list[str],
    target_column: str,
    label_function: Callable[[pd.Series], pd.Series],
    seed: int,
    ordered_validation: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float | int | str]]:
    target_labels = label_function(target[target_column]).astype(int).reset_index(drop=True)
    source = fit_selected_procedure(
        source_train[source_features + [target_column]],
        target[source_features],
        target_column,
        label_function,
        seed,
        ordered_validation,
    )
    proxy = fit_selected_procedure(
        proxy_train[proxy_features + [target_column]],
        target[proxy_features],
        target_column,
        label_function,
        seed + 100,
        ordered_validation,
    )
    losses = paired_prediction_losses(
        y_true=target_labels,
        source_probability=source.target_probability,
        proxy_probability=proxy.target_probability,
        source_thresholds={5.0: source.threshold},
        proxy_thresholds={5.0: proxy.threshold},
        record_ids=np.arange(len(target_labels)),
    )
    losses.insert(0, "Candidate", name)
    regrets = {
        "Brier": losses["brier_regret"].to_numpy(),
        "Clipped log loss": losses["logloss_regret"].to_numpy(),
        "Cost5x": losses["cost5x_regret"].to_numpy(),
    }
    diagnostics: dict[str, float | int | str] = {
        "Candidate": name,
        "SourceRows": len(source_train),
        "ProxyRows": len(proxy_train),
        "AuditN": len(target),
        "SourceFeatures": len(source_features),
        "ProxyFeatures": len(proxy_features),
        "SourceModel": source.model_name,
        "ProxyModel": proxy.model_name,
        "SourceAUC": roc_auc_score(target_labels, source.target_probability),
        "ProxyAUC": roc_auc_score(target_labels, proxy.target_probability),
        "AUCChange": roc_auc_score(target_labels, proxy.target_probability)
        - roc_auc_score(target_labels, source.target_probability),
        "SourceCost5x": _normalized_cost(
            target_labels,
            source.target_probability,
            source.threshold,
        ),
        "ProxyCost5x": _normalized_cost(
            target_labels,
            proxy.target_probability,
            proxy.threshold,
        ),
        "SourceThreshold": source.threshold,
        "ProxyThreshold": proxy.threshold,
    }
    return regrets, losses, diagnostics


def temporal_case(data_path: Path, seed: int):
    frame = pd.read_csv(data_path, sep=";")
    first_january, second_january = campaign_boundaries(frame)
    source = frame.iloc[:first_january].copy()
    proxy = frame.iloc[first_january:second_january].copy()
    target = frame.iloc[second_january:].copy()
    features = [column for column in frame.columns if column != "y"]
    return _audit_pair(
        name="Bank Marketing temporal",
        source_train=source,
        proxy_train=proxy,
        target=target,
        source_features=features,
        proxy_features=features,
        target_column="y",
        label_function=lambda values: values.astype(str).str.strip().eq("yes"),
        seed=seed,
        ordered_validation=True,
    )


def _read_heart_site(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=HEART_COLUMNS,
        na_values="?",
    )
    for column in HEART_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["diagnosis"]).reset_index(drop=True)


def site_case(data_root: Path, seed: int):
    source = _read_heart_site(data_root / "processed.cleveland.data")
    proxy = _read_heart_site(data_root / "processed.hungarian.data")
    target = pd.concat(
        [
            _read_heart_site(data_root / "processed.switzerland.data"),
            _read_heart_site(data_root / "processed.va.data"),
        ],
        ignore_index=True,
    )
    features = [column for column in HEART_COLUMNS if column != "diagnosis"]
    return _audit_pair(
        name="Heart Disease site",
        source_train=source,
        proxy_train=proxy,
        target=target,
        source_features=features,
        proxy_features=features,
        target_column="diagnosis",
        label_function=lambda values: pd.to_numeric(values, errors="coerce").gt(0),
        seed=seed,
        ordered_validation=False,
    )


def schema_case(
    data_path: Path,
    removed_fields: list[str],
    seed: int,
):
    frame = pd.read_csv(
        data_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    frame["income"] = frame["income"].astype(str).str.strip().str.rstrip(".")
    labels = frame["income"].eq(">50K").astype(int)
    remainder_positions, target_positions = train_test_split(
        np.arange(len(frame)),
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    source_positions, proxy_positions = train_test_split(
        remainder_positions,
        test_size=1.0 / 3.0,
        random_state=seed + 1,
        stratify=labels.iloc[remainder_positions],
    )
    source = frame.iloc[source_positions].reset_index(drop=True)
    proxy = frame.iloc[proxy_positions].reset_index(drop=True)
    target = frame.iloc[target_positions].reset_index(drop=True)
    full_features = [column for column in ADULT_COLUMNS if column != "income"]
    reduced_features = [
        column for column in full_features if column not in set(removed_fields)
    ]
    return _audit_pair(
        name="Adult reduced schema",
        source_train=source,
        proxy_train=proxy,
        target=target,
        source_features=full_features,
        proxy_features=reduced_features,
        target_column="income",
        label_function=lambda values: values.astype(str).str.strip().str.rstrip(".").eq(">50K"),
        seed=seed,
        ordered_validation=False,
    )


def run_shift_audits(
    registry: dict,
    bank_path: Path,
    heart_root: Path,
    adult_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = [
        temporal_case(bank_path, seed),
        site_case(heart_root, seed + 1),
        schema_case(
            adult_path,
            list(registry["shift_audits"]["schema"]["removed_fields"]),
            seed + 2,
        ),
    ]
    regrets = {
        diagnostics["Candidate"]: candidate_regrets
        for candidate_regrets, _losses, diagnostics in cases
    }
    losses = pd.concat([case_losses for _, case_losses, _ in cases], ignore_index=True)
    diagnostics = pd.DataFrame([case_diagnostics for _, _, case_diagnostics in cases])
    result = audit_proxy_candidates(
        regrets,
        requirements=REQUIREMENTS,
        alpha=float(registry["risk_control"]["alpha"]),
        bound_method=str(registry["risk_control"]["bound_method"]),
    )
    summary = result.candidate_summary.merge(
        diagnostics,
        on="Candidate",
        how="left",
        validate="one_to_one",
    )
    return summary, result.requirement_detail, losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run temporal, site, and schema-mismatch ProxyGuard audits."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_frontier_registry.json",
    )
    parser.add_argument("--bank", default="data/bank_marketing.csv")
    parser.add_argument(
        "--heart-root",
        default="data/uci_heart_disease",
    )
    parser.add_argument("--adult", default="data/adult_income.csv")
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_shift_audits",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    summary, detail, losses = run_shift_audits(
        registry,
        bank_path=Path(args.bank),
        heart_root=Path(args.heart_root),
        adult_path=Path(args.adult),
        seed=args.seed,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "candidate_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registered_shift_audits": registry["shift_audits"],
                "heart_data_source": (
                    "https://archive.ics.uci.edu/dataset/45/heart+disease"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[
            [
                "Candidate",
                "AuditN",
                "Status",
                "AUCChange",
                "SourceCost5x",
                "ProxyCost5x",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
