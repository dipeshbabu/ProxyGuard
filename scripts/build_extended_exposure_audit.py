from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config
from risk_models.dataset import load_dataset
from scripts.build_ruap_audit import (
    DISPLAY_DATASETS,
    DISPLAY_VARIANTS,
    nearest_neighbor_risk,
    subgroup_predictability,
    uniqueness_rate,
)
from scripts.run_proxy_transform_audit import parse_csv, transform_features


DEFAULT_DATASETS = [
    "adult_income",
    "australian_credit",
    "bank_marketing",
    "compas_recidivism",
    "german_credit",
    "give_me_some_credit",
    "heart_disease",
    "mammographic_mass",
    "taiwan_default",
    "breast_cancer_wdbc",
]

DEFAULT_VARIANTS = [
    "baseline",
    "numeric_noise_20",
    "laplace_noise_20",
    "coarsen_quartile",
    "rank_swap_10",
    "feature_mask_20",
    "sensitive_mask",
    "synthetic_marginal",
    "noisy_synthetic_marginal",
    "dp_marginal_e1",
]

DATASET_DISPLAY = {
    **DISPLAY_DATASETS,
    "give_me_some_credit": "GMSC",
    "heart_disease": "Heart",
    "mammographic_mass": "Mammography",
    "breast_cancer_wdbc": "WDBC",
}


def capped_frame_pair(original: pd.DataFrame, transformed: pd.DataFrame, max_rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(original) <= max_rows:
        return original.reset_index(drop=True), transformed.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    take = rng.choice(np.arange(len(original)), size=max_rows, replace=False)
    return original.iloc[take].reset_index(drop=True), transformed.iloc[take].reset_index(drop=True)


def row_distances(left, right) -> np.ndarray:
    diff = left - right
    if hasattr(diff, "multiply"):
        return np.sqrt(np.asarray(diff.multiply(diff).sum(axis=1)).ravel())
    return np.linalg.norm(np.asarray(diff), axis=1)


def auxiliary_linkage_auc(original: pd.DataFrame, transformed: pd.DataFrame, seed: int, max_rows: int) -> float:
    """A simple auxiliary-linkage probe.

    Positives are true original/transformed record pairs. Negatives pair each
    transformed record with a randomly chosen different original record. A high
    AUC means the transformed rows remain easy to link back to the source rows.
    """
    if original.empty or len(original) < 3:
        return float("nan")
    original, transformed = capped_frame_pair(original, transformed, max_rows=max_rows, seed=seed)
    numeric_columns = original.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in original.columns if column not in numeric_columns]
    encoder_kwargs = {"handle_unknown": "ignore"}
    transformer = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False)), numeric_columns),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(**encoder_kwargs)), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    combined = pd.concat([original, transformed], ignore_index=True)
    matrix = transformer.fit_transform(combined)
    n = len(original)
    source = matrix[:n]
    proxy = matrix[n:]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if np.any(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    positive_distance = row_distances(source, proxy)
    negative_distance = row_distances(source[perm], proxy)
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    scores = -np.concatenate([positive_distance, negative_distance])
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def membership_distance_auc(
    original: pd.DataFrame,
    y: pd.Series,
    variant: str,
    seed: int,
    sensitive_columns: list[str],
    max_rows: int,
) -> float:
    """Nearest-release membership probe.

    The released proxy is generated from one half of the rows. The attack then
    scores candidate source rows and held-out rows by nearest-neighbor distance
    to that proxy. A high AUC means source rows are easier to recognize.
    """
    if original.empty or len(original) < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(original) > max_rows:
        take = rng.choice(np.arange(len(original)), size=max_rows, replace=False)
        original = original.iloc[take].reset_index(drop=True)
        y = y.iloc[take].reset_index(drop=True)
    else:
        original = original.reset_index(drop=True)
        y = y.reset_index(drop=True)
    positives = []
    negatives = []
    for _, group_index in y.groupby(y).groups.items():
        group_index = np.asarray(list(group_index))
        if len(group_index) < 4:
            continue
        rng.shuffle(group_index)
        midpoint = len(group_index) // 2
        positives.extend(group_index[:midpoint].tolist())
        negatives.extend(group_index[midpoint:].tolist())
    if len(positives) < 5 or len(negatives) < 5:
        return float("nan")

    source = original.loc[positives].copy()
    source_y = y.loc[positives]
    release = transform_features(source, variant=variant, seed=seed, sensitive_columns=sensitive_columns, y=source_y)
    candidates = pd.concat([original.loc[positives], original.loc[negatives]], ignore_index=True)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])

    numeric_columns = original.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in original.columns if column not in numeric_columns]
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
    distances, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(release_matrix).kneighbors(candidate_matrix)
    try:
        return float(roc_auc_score(labels, -distances[:, 0]))
    except ValueError:
        return float("nan")


def build_extended_exposure_table(datasets: list[str], variants: list[str], seed: int, max_rows: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in datasets:
        bundle = load_dataset(get_dataset_config(dataset_name))
        original = bundle["X"]
        y = bundle["y"]
        subgroup_frame = bundle["metadata"].get("subgroup_frame", pd.DataFrame())
        sensitive_columns = list(subgroup_frame.columns) if subgroup_frame is not None else []
        baseline_values: dict[str, float] | None = None
        for variant in variants:
            transformed = transform_features(original, variant=variant, seed=seed, sensitive_columns=sensitive_columns, y=y)
            predictability, target = subgroup_predictability(transformed, subgroup_frame, seed, max_rows=max_rows)
            values = {
                "UniquenessRate": uniqueness_rate(transformed),
                "NearestNeighborRisk": nearest_neighbor_risk(transformed, seed=seed, max_rows=max_rows),
                "SensitivePredictability": predictability,
                "AuxLinkAUC": auxiliary_linkage_auc(original, transformed, seed=seed, max_rows=max_rows),
                "MemberAUC": membership_distance_auc(original, y, variant, seed, sensitive_columns, max_rows=max_rows),
            }
            if variant == "baseline":
                baseline_values = dict(values)
            deltas = {
                f"{key}Delta": (
                    values[key] - baseline_values[key]
                    if baseline_values is not None and np.isfinite(values[key]) and np.isfinite(baseline_values[key])
                    else float("nan")
                )
                for key in values
            }
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Variant": variant,
                    **values,
                    **deltas,
                    "SensitiveTarget": target,
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def write_latex_table(table: pd.DataFrame, output_dir: Path) -> None:
    rows = table[table["Variant"] != "baseline"].copy()
    order = {variant: index for index, variant in enumerate(DEFAULT_VARIANTS)}
    rows["VariantOrder"] = rows["Variant"].map(order).fillna(99)
    rows = rows.sort_values(["Dataset", "VariantOrder"])
    lines = [
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\begin{longtable}{@{}llrrrrr@{}}",
        "\\caption{All-dataset exposure stress screen. Deltas compare each transformed table with the original table for the same dataset. AuxLink is an auxiliary-linkage AUC that distinguishes true original/transformed row pairs from random pairs; MemAUC is a nearest-release membership-distance AUC. Lower deltas are better. This is an exposure screen, not a privacy guarantee.}\\label{tab:extended_exposure_app}\\\\",
        "\\toprule",
        "Dataset & Transform & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak & $\\Delta$AuxLink & $\\Delta$MemAUC \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\caption[]{All-dataset exposure stress screen (continued).}\\\\",
        "\\toprule",
        "Dataset & Transform & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak & $\\Delta$AuxLink & $\\Delta$MemAUC \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{DISPLAY_VARIANTS.get(row['Variant'], str(row['Variant']).replace('_', ' '))} & "
            f"{fmt_delta(row['UniquenessRateDelta'])} & {fmt_delta(row['NearestNeighborRiskDelta'])} & "
            f"{fmt_delta(row['SensitivePredictabilityDelta'])} & {fmt_delta(row['AuxLinkAUCDelta'])} & "
            f"{fmt_delta(row['MemberAUCDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\end{longtable}",
            "\\end{small}",
        ]
    )
    (output_dir / "extended_exposure_stress.tex").write_text("\n".join(lines), encoding="utf-8")


def write_compact_latex_table(table: pd.DataFrame, output_dir: Path) -> None:
    """Write the representative transforms used in the manuscript appendix."""
    compact_variants = ["coarsen_quartile", "sensitive_mask", "dp_marginal_e1"]
    rows = table[table["Variant"].isin(compact_variants)].copy()
    order = {variant: index for index, variant in enumerate(compact_variants)}
    rows["VariantOrder"] = rows["Variant"].map(order)
    rows = rows.sort_values(["Dataset", "VariantOrder"])
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.04}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Transform & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak & $\\Delta$AuxLink & $\\Delta$MemAUC \\\\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{DISPLAY_VARIANTS.get(row['Variant'], str(row['Variant']).replace('_', ' '))} & "
            f"{fmt_delta(row['UniquenessRateDelta'])} & {fmt_delta(row['NearestNeighborRiskDelta'])} & "
            f"{fmt_delta(row['SensitivePredictabilityDelta'])} & {fmt_delta(row['AuxLinkAUCDelta'])} & "
            f"{fmt_delta(row['MemberAUCDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Representative all-dataset exposure screen. Deltas compare each transform with the original table; lower values are better. The released CSV contains all 100 dataset--transform cells.}\\label{tab:extended_exposure_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "extended_exposure_compact.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser(description="Build an all-dataset RUA-P exposure stress screen.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    parser.add_argument("--input-csv", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_csv:
        table = pd.read_csv(args.input_csv)
    else:
        table = build_extended_exposure_table(parse_csv(args.datasets), parse_csv(args.variants), args.seed, args.max_rows)
        table.to_csv(output_dir / "extended_exposure_stress.csv", index=False)
    write_latex_table(table, output_dir)
    write_compact_latex_table(table, output_dir)
    print(f"wrote {len(table)} extended exposure rows to {output_dir}")


if __name__ == "__main__":
    main()
