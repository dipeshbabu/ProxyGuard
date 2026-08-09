from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.tabular.metrics import (  # noqa: E402
    apply_calibrator,
    best_cost_threshold_from_val,
    compute_ece,
    evaluate_predictions,
    fit_calibrator,
)


DISPLAY_MODELS = {
    "logreg": "LogReg",
    "histgb": "HistGB",
}


def campaign_boundaries(frame: pd.DataFrame) -> tuple[int, int]:
    """Return the first two January transitions in the date-ordered table."""
    month = frame["month"].astype(str).str.lower()
    january_starts = frame.index[
        month.eq("jan") & ~month.shift(fill_value="").eq("jan")
    ].tolist()
    if len(january_starts) < 2:
        raise ValueError("Bank Marketing data do not contain two January transitions.")
    return int(january_starts[0]), int(january_starts[1])


def split_train_validation(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(round(len(frame) * (1.0 - validation_fraction)))
    cut = min(max(cut, 1), len(frame) - 1)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def make_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_model(name: str, frame: pd.DataFrame, seed: int) -> Pipeline:
    if name == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        return Pipeline(
            [
                ("preprocess", make_preprocessor(frame, scale_numeric=True)),
                ("model", estimator),
            ]
        )
    if name == "histgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline(
            [
                ("preprocess", make_preprocessor(frame, scale_numeric=False)),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def age_groups(age: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(age, errors="coerce"),
        bins=[-np.inf, 29, 44, 59, np.inf],
        labels=["<30", "30--44", "45--59", "60+"],
    ).astype(str)


def worst_age_ece(y_true: pd.Series, probabilities: np.ndarray, age: pd.Series) -> float:
    groups = age_groups(age)
    values = []
    for group in sorted(groups.dropna().unique()):
        mask = groups.eq(group).to_numpy()
        if mask.sum() < 50:
            continue
        values.append(compute_ece(y_true.to_numpy()[mask], probabilities[mask], n_bins=10))
    return float(max(values)) if values else float("nan")


def fit_and_evaluate(
    model_name: str,
    train_period: pd.DataFrame,
    future_test: pd.DataFrame,
    target: str,
    seed: int,
) -> dict[str, object]:
    train, validation = split_train_validation(train_period)
    x_train = train.drop(columns=[target])
    y_train = train[target].eq("yes").astype(int)
    x_validation = validation.drop(columns=[target])
    y_validation = validation[target].eq("yes").astype(int)
    x_test = future_test.drop(columns=[target])
    y_test = future_test[target].eq("yes").astype(int)

    model = make_model(model_name, x_train, seed)
    model.fit(x_train, y_train)
    raw_validation = model.predict_proba(x_validation)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    calibrator = fit_calibrator(raw_validation, y_validation, method="temperature")
    validation_probability = apply_calibrator(calibrator, raw_validation)
    test_probability = apply_calibrator(calibrator, raw_test)
    threshold, validation_cost = best_cost_threshold_from_val(
        y_validation,
        validation_probability,
        fn_cost=5.0,
        fp_cost=1.0,
    )
    metrics = evaluate_predictions(y_test, test_probability, threshold=threshold)
    return {
        "Model": model_name,
        "ValidationCost5x": float(validation_cost),
        "Threshold": float(threshold),
        "WorstAgeECE": worst_age_ece(
            y_test.reset_index(drop=True),
            test_probability,
            future_test["age"].reset_index(drop=True),
        ),
        **metrics,
    }


def run_audit(data_path: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(data_path, sep=";")
    first_january, second_january = campaign_boundaries(frame)
    periods = {
        "source_2008": frame.iloc[:first_january].copy(),
        "proxy_2009": frame.iloc[first_january:second_january].copy(),
    }
    future_test = frame.iloc[second_january:].copy()

    rows = []
    for period_name, period in periods.items():
        for model_name in DISPLAY_MODELS:
            rows.append(
                {
                    "TrainPeriod": period_name,
                    "TrainRows": int(len(period)),
                    "TrainPositiveRate": float(period["y"].eq("yes").mean()),
                    "TestRows": int(len(future_test)),
                    "TestPositiveRate": float(future_test["y"].eq("yes").mean()),
                    **fit_and_evaluate(
                        model_name,
                        period,
                        future_test,
                        target="y",
                        seed=seed,
                    ),
                }
            )
    model_results = pd.DataFrame(rows)

    selected_rows = []
    for period_name, group in model_results.groupby("TrainPeriod", sort=False):
        selected = group.loc[group["ValidationCost5x"].idxmin()].copy()
        selected_rows.append(selected)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    source = selected[selected["TrainPeriod"] == "source_2008"].iloc[0]
    proxy = selected[selected["TrainPeriod"] == "proxy_2009"].iloc[0]
    selected.loc[selected["TrainPeriod"] == "source_2008", "Period"] = "2008 source"
    selected.loc[selected["TrainPeriod"] == "proxy_2009", "Period"] = "2009 temporal proxy"
    selected["AUCDeltaVsSource"] = selected["AUC"] - float(source["AUC"])
    selected["ECEDeltaVsSource"] = (
        selected["ECE (10-bin)"] - float(source["ECE (10-bin)"])
    )
    selected["CostDeltaVsSource"] = (
        selected["DecisionCost5x"] - float(source["DecisionCost5x"])
    )
    selected["WorstAgeECEDeltaVsSource"] = (
        selected["WorstAgeECE"] - float(source["WorstAgeECE"])
    )
    selected.attrs["boundaries"] = {
        "first_january": first_january,
        "second_january": second_january,
        "proxy_auc_delta": float(proxy["AUC"] - source["AUC"]),
    }
    return model_results, selected


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}"


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}".replace("%", "\\%")


def write_latex(selected: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Training table & AUC & ECE & Cost5x & Worst age ECE & $\\Delta$AUC & $\\Delta$Cost \\\\",
        "\\midrule",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{row['Period']} & {fmt(row['AUC'])} & {fmt(row['ECE (10-bin)'])} & "
            f"{fmt(row['DecisionCost5x'])} & {fmt(row['WorstAgeECE'])} & "
            f"{fmt_delta(row['AUCDeltaVsSource'])} & "
            f"{fmt_delta(row['CostDeltaVsSource'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Temporal stand-in audit on Bank Marketing. The 2008 source has 27{,}729 rows (5.1\\% positive), and the 2009 proxy has 14{,}862 (17.1\\% positive); both use validation-selected HistGB and are evaluated on the 2010 wave ($n=2{,}620$, 51.6\\% positive). There is no row correspondence between waves, so row-linkage probes are omitted.}\\label{tab:temporal_proxy}",
            "\\end{table}",
        ]
    )
    (output_dir / "temporal_proxy_audit.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a date-ordered Bank Marketing wave as a temporal stand-in."
    )
    parser.add_argument(
        "--data",
        default="data/bank_marketing.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_assets/ruap_audit",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results, selected = run_audit(Path(args.data), seed=args.seed)
    model_results.to_csv(output_dir / "temporal_proxy_models.csv", index=False)
    selected.to_csv(output_dir / "temporal_proxy_selected.csv", index=False)
    write_latex(selected, output_dir)
    print(
        f"wrote {len(model_results)} temporal model rows and "
        f"{len(selected)} selected rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
