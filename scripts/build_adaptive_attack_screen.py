from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config
from risk_models.dataset import load_dataset
from scripts.build_extended_exposure_audit import DATASET_DISPLAY
from scripts.build_ruap_audit import DISPLAY_VARIANTS, coarse_fingerprint_frame
from scripts.run_proxy_transform_audit import parse_csv, transform_features


DEFAULT_DATASETS = [
    "adult_income",
    "australian_credit",
    "bank_marketing",
    "breast_cancer_wdbc",
    "compas_recidivism",
    "german_credit",
    "give_me_some_credit",
    "heart_disease",
    "mammographic_mass",
    "taiwan_default",
]

DEFAULT_VARIANTS = [
    "baseline",
    "coarsen_quartile",
    "sensitive_mask",
    "synthetic_marginal",
    "dp_marginal_e1",
]


def stratified_source_holdout(X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        selected: list[int] = []
        for _, group_index in y.groupby(y).groups.items():
            group_index = np.asarray(list(group_index))
            take = max(2, int(round(max_rows * len(group_index) / len(X))))
            take = min(take, len(group_index))
            selected.extend(rng.choice(group_index, size=take, replace=False).tolist())
        selected = selected[:max_rows]
        X = X.iloc[selected].reset_index(drop=True)
        y = y.iloc[selected].reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    source_idx: list[int] = []
    holdout_idx: list[int] = []
    for _, group_index in y.groupby(y).groups.items():
        group_index = np.asarray(list(group_index))
        if len(group_index) < 4:
            continue
        rng.shuffle(group_index)
        midpoint = len(group_index) // 2
        source_idx.extend(group_index[:midpoint].tolist())
        holdout_idx.extend(group_index[midpoint:].tolist())
    return X.loc[source_idx].reset_index(drop=True), y.loc[source_idx].reset_index(drop=True), X.loc[holdout_idx].reset_index(drop=True), y.loc[holdout_idx].reset_index(drop=True)


def fingerprint_match_features(candidates: pd.DataFrame, release: pd.DataFrame) -> np.ndarray:
    def row_keys(frame: pd.DataFrame) -> pd.Series:
        coarse = coarse_fingerprint_frame(frame)
        return coarse.apply(lambda row: "|".join(str(value) for value in row.to_numpy(dtype=object)), axis=1)

    candidate_fp = row_keys(candidates)
    release_fp = set(row_keys(release).tolist())
    return candidate_fp.isin(release_fp).astype(float).to_numpy()


def adaptive_attack_auc(X: pd.DataFrame, y: pd.Series, variant: str, sensitive_columns: list[str], seed: int, max_rows: int) -> float:
    source, source_y, holdout, _ = stratified_source_holdout(X, y, max_rows=max_rows, seed=seed)
    if len(source) < 20 or len(holdout) < 20:
        return float("nan")
    release = transform_features(source, variant=variant, seed=seed, sensitive_columns=sensitive_columns, y=source_y)
    candidates = pd.concat([source, holdout], ignore_index=True)
    labels = np.concatenate([np.ones(len(source)), np.zeros(len(holdout))])

    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformer = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False)), numeric_columns),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    matrix = transformer.fit_transform(pd.concat([candidates, release], ignore_index=True))
    candidate_matrix = matrix[: len(candidates)]
    release_matrix = matrix[len(candidates) :]
    n_neighbors = min(5, max(1, len(release) - 1))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(release_matrix).kneighbors(candidate_matrix)
    attack_features = np.column_stack(
        [
            distances[:, 0],
            distances.mean(axis=1),
            distances.std(axis=1),
            np.exp(-distances[:, 0]),
            fingerprint_match_features(candidates, release),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=4,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        scores = cross_val_predict(clf, attack_features, labels, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def build_table(datasets: list[str], variants: list[str], seed: int, max_rows: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_dataset(get_dataset_config(dataset))
        X = bundle["X"]
        y = bundle["y"]
        subgroup_frame = bundle["metadata"].get("subgroup_frame", pd.DataFrame())
        sensitive_columns = list(subgroup_frame.columns) if subgroup_frame is not None else []
        baseline_auc = None
        for variant in variants:
            auc = adaptive_attack_auc(X, y, variant=variant, sensitive_columns=sensitive_columns, seed=seed, max_rows=max_rows)
            if variant == "baseline":
                baseline_auc = auc
            delta = auc - baseline_auc if baseline_auc is not None and np.isfinite(auc) and np.isfinite(baseline_auc) else float("nan")
            rows.append({"Dataset": dataset, "Variant": variant, "AdaptiveAttackAUC": auc, "AdaptiveAttackDelta": delta})
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def write_latex(table: pd.DataFrame, output_dir: Path) -> None:
    rows = table[table["Variant"] != "baseline"].copy()
    variant_order = {variant: i for i, variant in enumerate(DEFAULT_VARIANTS)}
    rows["VariantOrder"] = rows["Variant"].map(variant_order).fillna(99)
    rows = rows.sort_values(["Dataset", "VariantOrder"])
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{0.90\\textwidth}{!}{%",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Transform & Attack AUC & $\\Delta$Attack AUC \\\\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{DISPLAY_VARIANTS.get(row['Variant'], row['Variant'].replace('_', ' '))} & "
            f"{fmt(row['AdaptiveAttackAUC'])} & {fmt_delta(row['AdaptiveAttackDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Adaptive nearest-release attack screen. The attacker trains a random-forest classifier over nearest-release distances, local distance summaries, and coarse-fingerprint matches to distinguish source rows from held-out rows. Deltas compare each transformed release with the original-table baseline for the same dataset; lower AUC is better.}\\label{tab:adaptive_attack_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "adaptive_attack_screen.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser(description="Build adaptive attack screen for RUA-P.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-rows", type=int, default=1200)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(parse_csv(args.datasets), parse_csv(args.variants), seed=args.seed, max_rows=args.max_rows)
    table.to_csv(output_dir / "adaptive_attack_screen.csv", index=False)
    write_latex(table, output_dir)
    print(f"wrote {len(table)} adaptive attack rows to {output_dir}")


if __name__ == "__main__":
    main()
