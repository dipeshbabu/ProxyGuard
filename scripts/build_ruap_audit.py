from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config
from risk_models.dataset import load_dataset
from scripts.run_proxy_transform_audit import parse_csv, transform_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_DATASETS = {
    "adult_income": "Adult",
    "australian_credit": "Australian",
    "bank_marketing": "Bank",
    "compas_recidivism": "COMPAS",
    "german_credit": "German",
    "taiwan_default": "Taiwan",
}

DISPLAY_VARIANTS = {
    "baseline": "Original",
    "numeric_noise_10": "10\\% noise",
    "numeric_noise_20": "20\\% noise",
    "laplace_noise_20": "Laplace 20\\%",
    "coarsen_quartile": "Quartile",
    "rank_swap_10": "Rank swap",
    "feature_mask_20": "20\\% mask",
    "sensitive_mask": "Sensitive mask",
    "synthetic_marginal": "Marginal synth",
    "noisy_synthetic_marginal": "Noisy marginal",
    "dp_marginal_e1": "DP marginal $\\epsilon{=}1$",
    "synthetic_marginal_y": "Marginal synth",
    "noisy_synthetic_marginal_y": "Noisy marginal",
    "dp_marginal_e1_y": "DP marginal $\\epsilon{=}1$",
}

DISPLAY_MODELS = {
    "xgb_baseline": "XGB",
    "compact_xgb": "Compact XGB",
    "tabpfn_baseline": "TabPFN",
    "tabicl_baseline": "TabICL",
    "rc_stack": "RC-Stack",
    "rrc_stack": "RRC-Stack",
}


def read_proxy_outputs(root: Path) -> pd.DataFrame:
    rows = []
    for aggregate_path in sorted((root / "proxy_transform").glob("*/*/aggregate_metrics.csv")):
        variant = aggregate_path.parent.parent.name
        dataset = aggregate_path.parent.name
        frame = pd.read_csv(aggregate_path)
        frame.insert(0, "Variant", variant)
        frame.insert(0, "Dataset", dataset)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def coarse_fingerprint_frame(X: pd.DataFrame, max_columns: int = 64) -> pd.DataFrame:
    if X.shape[1] > max_columns:
        variances = X.select_dtypes(include=[np.number]).var(numeric_only=True).sort_values(ascending=False)
        keep = list(variances.head(max_columns).index)
        if not keep:
            keep = list(X.columns[:max_columns])
        X = X[keep]
    coarse = pd.DataFrame(index=X.index)
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_bool_dtype(series) or series.dropna().nunique() <= 2:
            coarse[column] = series.fillna(False).astype(str)
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() and numeric.nunique(dropna=True) >= 8:
            ranked = numeric.rank(method="first")
            coarse[column] = pd.qcut(ranked, q=10, labels=False, duplicates="drop").astype("Int64").astype(str)
        else:
            coarse[column] = series.astype(str).fillna("missing")
    return coarse


def uniqueness_rate(X: pd.DataFrame) -> float:
    coarse = coarse_fingerprint_frame(X)
    counts = coarse.value_counts(dropna=False)
    keys = pd.MultiIndex.from_frame(coarse)
    return float(np.mean(counts.reindex(keys).to_numpy() == 1))


def nearest_neighbor_risk(X: pd.DataFrame, seed: int, max_rows: int) -> float:
    if X.empty or len(X) < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        sample_index = rng.choice(X.index.to_numpy(), size=max_rows, replace=False)
        X = X.loc[sample_index]
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
    matrix = transformer.fit_transform(X)
    n_neighbors = min(2, matrix.shape[0])
    if n_neighbors < 2:
        return float("nan")
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(matrix).kneighbors(matrix)
    nearest = distances[:, 1]
    return float(np.mean(np.exp(-nearest)))


def exposure_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")


def capped_index(y: pd.Series, max_rows: int, seed: int) -> pd.Index:
    if len(y) <= max_rows:
        return y.index
    parts = []
    rng = np.random.default_rng(seed)
    for _, group_index in y.groupby(y).groups.items():
        group_index = pd.Index(group_index)
        take = max(1, int(round(max_rows * len(group_index) / len(y))))
        take = min(take, len(group_index))
        parts.extend(rng.choice(group_index.to_numpy(), size=take, replace=False).tolist())
    if len(parts) > max_rows:
        parts = rng.choice(np.array(parts), size=max_rows, replace=False).tolist()
    return pd.Index(parts)


def subgroup_predictability(X: pd.DataFrame, subgroup_frame: pd.DataFrame, seed: int, max_rows: int) -> tuple[float, str]:
    if subgroup_frame is None or subgroup_frame.empty:
        return float("nan"), ""

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
    estimator = make_pipeline(transformer, exposure_model())
    best_score = float("nan")
    best_target = ""
    for column in subgroup_frame.columns:
        y = subgroup_frame[column].astype(str).fillna("missing")
        counts = y.value_counts()
        valid = counts[counts >= 8].index
        mask = y.isin(valid)
        y = y[mask]
        if y.nunique() < 2 or len(y) < 60:
            continue
        X_target = X.loc[mask]
        sample_index = capped_index(y, max_rows=max_rows, seed=seed)
        y = y.loc[sample_index]
        X_target = X_target.loc[sample_index]
        n_splits = min(5, int(y.value_counts().min()))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        try:
            if y.nunique() == 2:
                probs = cross_val_predict(estimator, X_target, y, cv=cv, method="predict_proba", n_jobs=1)
                score = roc_auc_score((y == sorted(y.unique())[1]).astype(int), probs[:, 1])
            else:
                preds = cross_val_predict(estimator, X_target, y, cv=cv, method="predict", n_jobs=1)
                score = balanced_accuracy_score(y, preds)
        except Exception:
            continue
        if not np.isfinite(best_score) or score > best_score:
            best_score = float(score)
            best_target = str(column)
    return best_score, best_target


def build_exposure_table(datasets: list[str], variants: list[str], seed: int, max_rows: int = 10000) -> pd.DataFrame:
    rows = []
    for dataset_name in datasets:
        bundle = load_dataset(get_dataset_config(dataset_name))
        subgroup_frame = bundle["metadata"].get("subgroup_frame", pd.DataFrame())
        sensitive_columns = list(subgroup_frame.columns) if subgroup_frame is not None else []
        for variant in variants:
            X = transform_features(bundle["X"], variant=variant, seed=seed, sensitive_columns=sensitive_columns, y=bundle["y"])
            predictability, target = subgroup_predictability(X, subgroup_frame, seed, max_rows=max_rows)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Variant": variant,
                    "UniquenessRate": uniqueness_rate(X),
                    "NearestNeighborRisk": nearest_neighbor_risk(X, seed=seed, max_rows=max_rows),
                    "SensitivePredictability": predictability,
                    "SensitiveTarget": target,
                }
            )
    return pd.DataFrame(rows)


def best_row(frame: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.Series:
    idx = frame[metric].idxmax() if higher_is_better else frame[metric].idxmin()
    return frame.loc[idx]


def build_ruap_report(proxy_results: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in proxy_results.groupby("Dataset"):
        baseline_metrics = dataset_frame[dataset_frame["Variant"] == "baseline"]
        baseline_exposure = exposure[(exposure["Dataset"] == dataset) & (exposure["Variant"] == "baseline")]
        if baseline_metrics.empty or baseline_exposure.empty:
            continue
        baseline = {
            "AUC": best_row(baseline_metrics, "AUC", True)["AUC"],
            "ECE": best_row(baseline_metrics, "ECE (10-bin)", False)["ECE (10-bin)"],
            "Cost": best_row(baseline_metrics, "DecisionCost5x", False)["DecisionCost5x"],
            "Uniqueness": float(baseline_exposure["UniquenessRate"].iloc[0]),
            "NNRisk": float(baseline_exposure["NearestNeighborRisk"].iloc[0]),
            "Leakage": float(baseline_exposure["SensitivePredictability"].iloc[0]),
        }
        for variant, frame in dataset_frame.groupby("Variant"):
            variant_exposure = exposure[(exposure["Dataset"] == dataset) & (exposure["Variant"] == variant)]
            if variant_exposure.empty:
                continue
            auc = best_row(frame, "AUC", True)
            ece = best_row(frame, "ECE (10-bin)", False)
            cost = best_row(frame, "DecisionCost5x", False)
            uniqueness = float(variant_exposure["UniquenessRate"].iloc[0])
            nn_risk = float(variant_exposure["NearestNeighborRisk"].iloc[0])
            leakage = float(variant_exposure["SensitivePredictability"].iloc[0])
            auc_delta = float(auc["AUC"] - baseline["AUC"])
            ece_delta = float(ece["ECE (10-bin)"] - baseline["ECE"])
            cost_delta = float(cost["DecisionCost5x"] - baseline["Cost"])
            unique_delta = uniqueness - baseline["Uniqueness"]
            nn_delta = nn_risk - baseline["NNRisk"] if np.isfinite(nn_risk) and np.isfinite(baseline["NNRisk"]) else float("nan")
            leakage_delta = leakage - baseline["Leakage"] if np.isfinite(leakage) and np.isfinite(baseline["Leakage"]) else float("nan")
            preserves_auc = auc_delta >= -0.01
            reliability_warning = ece_delta > 0.005 or cost_delta > 0.01
            exposure_deltas = [
                unique_delta,
                nn_delta if np.isfinite(nn_delta) else float("nan"),
                leakage_delta if np.isfinite(leakage_delta) else float("nan"),
            ]
            exposure_reduced = (
                unique_delta < -0.01
                or (np.isfinite(nn_delta) and nn_delta < -0.01)
                or (np.isfinite(leakage_delta) and leakage_delta < -0.01)
            )
            exposure_regressed = any(np.isfinite(delta) and delta > 0.01 for delta in exposure_deltas)
            if variant == "baseline":
                status = "Original"
            elif not preserves_auc:
                status = "Ranking-utility failure"
            elif reliability_warning:
                status = "Decision-utility failure"
            elif exposure_reduced and not exposure_regressed:
                status = "Requirements met"
            else:
                status = "Conflicting exposure"
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "BestAUCModel": auc["Model"],
                    "BestAUC": auc["AUC"],
                    "AUCDelta": auc_delta,
                    "BestECE": ece["ECE (10-bin)"],
                    "ECEDelta": ece_delta,
                    "BestCost5x": cost["DecisionCost5x"],
                    "CostDelta": cost_delta,
                    "UniquenessRate": uniqueness,
                    "UniquenessDelta": unique_delta,
                    "NearestNeighborRisk": nn_risk,
                    "NearestNeighborRiskDelta": nn_delta,
                    "SensitivePredictability": leakage,
                    "SensitivePredictabilityDelta": leakage_delta,
                    "SensitiveTarget": variant_exposure["SensitiveTarget"].iloc[0],
                    "Status": status,
                    "n_splits": int(frame["n_splits"].min()),
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    variant_order = {variant: index for index, variant in enumerate(DISPLAY_VARIANTS)}
    report["VariantOrder"] = report["Variant"].map(variant_order).fillna(99)
    return report.sort_values(["Dataset", "VariantOrder"]).drop(columns=["VariantOrder"]).reset_index(drop=True)


def build_ruap_certificate(
    report: pd.DataFrame,
    auc_tol: float = 0.010,
    ece_tol: float = 0.005,
    cost_tol: float = 0.010,
    exposure_tol: float | dict[str, float] = 0.010,
) -> pd.DataFrame:
    transformed = report[report["Variant"] != "baseline"].copy()
    if transformed.empty:
        return transformed
    transformed["M_U"] = np.minimum.reduce(
        [
            (transformed["AUCDelta"] + auc_tol) / auc_tol,
            (ece_tol - transformed["ECEDelta"]) / ece_tol,
            (cost_tol - transformed["CostDelta"]) / cost_tol,
        ]
    )
    exposure_columns = ["UniquenessDelta", "NearestNeighborRiskDelta", "SensitivePredictabilityDelta"]
    if isinstance(exposure_tol, dict):
        exposure_tolerances = np.asarray([exposure_tol[column] for column in exposure_columns], dtype=float)
    else:
        exposure_tolerances = np.full(len(exposure_columns), float(exposure_tol), dtype=float)
    if np.any(exposure_tolerances <= 0):
        raise ValueError("Exposure tolerances must be positive.")
    exposure_values = transformed[exposure_columns].to_numpy(dtype=float)
    exposure_gain = np.where(np.isfinite(exposure_values), -exposure_values / exposure_tolerances, np.nan)
    exposure_regression = np.where(np.isfinite(exposure_values), exposure_values / exposure_tolerances, np.nan)
    transformed["M_+"] = np.nanmax(exposure_gain, axis=1)
    transformed["M_-"] = np.nanmax(exposure_regression, axis=1)
    transformed["UtilityFeasible"] = (
        (transformed["AUCDelta"] >= -auc_tol)
        & (transformed["ECEDelta"] <= ece_tol)
        & (transformed["CostDelta"] <= cost_tol)
    )
    transformed["Frontier"] = False
    for dataset, frame in transformed.groupby("Dataset"):
        feasible = frame[frame["UtilityFeasible"]].copy()
        if feasible.empty:
            continue
        frontier_indices = []
        for idx, candidate in feasible.iterrows():
            candidate_values = candidate[exposure_columns].to_numpy(dtype=float)
            dominated = False
            for other_idx, other in feasible.iterrows():
                if other_idx == idx:
                    continue
                other_values = other[exposure_columns].to_numpy(dtype=float)
                finite = np.isfinite(candidate_values) & np.isfinite(other_values)
                if not finite.any():
                    continue
                weak = np.all(other_values[finite] <= candidate_values[finite] + exposure_tolerances[finite])
                strict = np.any(other_values[finite] < candidate_values[finite] - exposure_tolerances[finite])
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                frontier_indices.append(idx)
        transformed.loc[frontier_indices, "Frontier"] = True
    return transformed


def display(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value.replace("_", "\\_"))


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < 0.5 * (10 ** -digits):
        value = 0.0
    return f"{value:.{digits}f}"


def fmt_delta(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < 0.5 * (10 ** -digits):
        value = 0.0
    return f"{value:+.{digits}f}"


def write_latex_tables(report: pd.DataFrame, output_dir: Path) -> None:
    main = report[report["Variant"] != "baseline"].copy()
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Descriptive test-set extrema under proxy transformations. AUC, ECE, and cost can come from different learners; this table is not a model-selection procedure. Lower ECE, Cost5x, uniqueness, nearest-neighbor risk, and sensitive-attribute predictability are better.}",
        "\\label{tab:ruap_report}",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\resizebox{0.98\\textwidth}{!}{%",
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        "Dataset & Proxy & Highest AUC row & $\\Delta$AUC & $\\Delta$ECE & $\\Delta$Cost & $\\Delta$Unique & $\\Delta$NN & $\\Delta$Leak \\\\",
        "\\midrule",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"{display(row['Dataset'], DISPLAY_DATASETS)} & {display(row['Variant'], DISPLAY_VARIANTS)} & "
            f"{display(row['BestAUCModel'], DISPLAY_MODELS)} & {fmt_delta(row['AUCDelta'])} & "
            f"{fmt_delta(row['ECEDelta'])} & {fmt_delta(row['CostDelta'])} & "
            f"{fmt_delta(row['UniquenessDelta'])} & {fmt_delta(row['NearestNeighborRiskDelta'])} & "
            f"{fmt_delta(row['SensitivePredictabilityDelta'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\vskip -0.1in",
            "\\end{table*}",
            "",
        ]
    )
    (output_dir / "ruap_report_card.tex").write_text("\n".join(lines), encoding="utf-8")


def write_certificate_table(profile: pd.DataFrame, output_dir: Path) -> None:
    if profile.empty:
        return
    lines = [
        "\\begin{table}[t]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{0.98\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrll}",
        "\\toprule",
        "Dataset & Proxy & $M_U$ & $M_+$ & $M_-$ & Non-dominated & Decision state \\\\",
        "\\midrule",
    ]
    for _, row in profile.iterrows():
        frontier = "Yes" if bool(row["Frontier"]) else "No"
        lines.append(
            f"{display(row['Dataset'], DISPLAY_DATASETS)} & {display(row['Variant'], DISPLAY_VARIANTS)} & "
            f"{fmt_delta(row['M_U'], digits=2)} & {fmt(row['M_+'], digits=2)} & {fmt(row['M_-'], digits=2)} & "
            f"{frontier} & {row['Status']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Illustrative decision profile and non-dominated candidate set. Negative $M_U$ values indicate a failed utility requirement. $M_+$ and $M_-$ summarize the largest normalized exposure gain and regression. A non-dominated entry can still conflict with the original table because this set compares candidates only with one another.}\\label{tab:ruap_certificate}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "ruap_certificate.tex").write_text("\n".join(lines), encoding="utf-8")


def write_ruap_report_plot(report: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = report[report["Variant"] != "baseline"].copy()
    if plot_frame.empty:
        return
    plot_frame["Label"] = [
        f"{display(dataset, DISPLAY_DATASETS)}\n{display(variant, DISPLAY_VARIANTS)}"
        for dataset, variant in zip(plot_frame["Dataset"], plot_frame["Variant"])
    ]
    metrics = [
        ("AUCDelta", "$\\Delta$AUC"),
        ("ECEDelta", "$\\Delta$ECE"),
        ("CostDelta", "$\\Delta$Cost"),
        ("UniquenessDelta", "$\\Delta$Unique"),
        ("NearestNeighborRiskDelta", "$\\Delta$NN"),
        ("SensitivePredictabilityDelta", "$\\Delta$Leak"),
    ]
    status_colors = {
        "Requirements met": "#2a9d55",
        "Conflicting exposure": "#d19000",
        "Decision-utility failure": "#c43c39",
        "Ranking-utility failure": "#6a5acd",
    }
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    x_positions = np.arange(len(metrics))
    y_positions = np.arange(len(plot_frame))
    for y, (_, row) in zip(y_positions, plot_frame.iterrows()):
        color = status_colors.get(row["Status"], "#555555")
        for x, (column, _) in zip(x_positions, metrics):
            value = row[column]
            if not np.isfinite(value):
                continue
            size = 90 + min(abs(value) * 2200, 520)
            marker = "^" if value > 0 else "v" if value < 0 else "o"
            ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="black", linewidth=0.45, alpha=0.88)
            ax.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=7.4, color="white")
    ax.axvline(2.5, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_frame["Label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_title("RUA-P proxy audit: utility and exposure deltas", fontsize=12, pad=10)
    ax.text(1.0, -0.9, "utility", ha="center", va="center", fontsize=9)
    ax.text(4.0, -0.9, "exposure", ha="center", va="center", fontsize=9)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=status, markerfacecolor=color, markeredgecolor="black", markersize=8)
        for status, color in status_colors.items()
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False, fontsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ruap_report_card_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def classify_status(
    row: pd.Series,
    auc_tol: float,
    ece_tol: float,
    cost_tol: float,
    exposure_tol: float | dict[str, float],
) -> str:
    if row["Variant"] == "baseline":
        return "Original"
    preserves_auc = row["AUCDelta"] >= -auc_tol
    reliability_warning = row["ECEDelta"] > ece_tol or row["CostDelta"] > cost_tol
    exposure_columns = ["UniquenessDelta", "NearestNeighborRiskDelta", "SensitivePredictabilityDelta"]
    if isinstance(exposure_tol, dict):
        tolerances = {column: float(exposure_tol[column]) for column in exposure_columns}
    else:
        tolerances = {column: float(exposure_tol) for column in exposure_columns}
    exposure_reduced = any(
        np.isfinite(row[column]) and row[column] < -tolerances[column]
        for column in exposure_columns
    )
    exposure_regressed = any(
        np.isfinite(row[column]) and row[column] > tolerances[column]
        for column in exposure_columns
    )
    if not preserves_auc:
        return "Ranking-utility failure"
    if reliability_warning:
        return "Decision-utility failure"
    if exposure_reduced and not exposure_regressed:
        return "Requirements met"
    return "Conflicting exposure"


def build_threshold_sensitivity(report: pd.DataFrame) -> pd.DataFrame:
    settings = [
        ("strict", 0.005, 0.0025, 0.005, 0.005),
        ("default", 0.010, 0.0050, 0.010, 0.010),
        ("loose", 0.020, 0.0100, 0.020, 0.020),
    ]
    transformed = report[report["Variant"] != "baseline"].copy()
    rows = []
    for name, auc_tol, ece_tol, cost_tol, exposure_tol in settings:
        statuses = transformed.apply(
            classify_status,
            axis=1,
            auc_tol=auc_tol,
            ece_tol=ece_tol,
            cost_tol=cost_tol,
            exposure_tol=exposure_tol,
        )
        counts = statuses.value_counts()
        rows.append(
            {
                "Setting": name,
                "AUCTolerance": auc_tol,
                "ECETolerance": ece_tol,
                "CostTolerance": cost_tol,
                "ExposureTolerance": exposure_tol,
                "Requirements met": int(counts.get("Requirements met", 0)),
                "Conflicting exposure": int(counts.get("Conflicting exposure", 0)),
                "Decision-utility failure": int(counts.get("Decision-utility failure", 0)),
                "Ranking-utility failure": int(counts.get("Ranking-utility failure", 0)),
            }
        )
    return pd.DataFrame(rows)


def write_threshold_sensitivity_table(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\caption{Counts under three illustrative tolerance settings. Conflicting profiles preserve utility while exposure probes move in opposite directions. These counts describe the optional policy layer, not the strength of the underlying evidence.}",
        "\\label{tab:ruap_threshold_sensitivity}",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Setting & Requirements met & Conflicting & Decision-utility failure & Ranking-utility failure \\\\",
        "\\midrule",
    ]
    for _, row in sensitivity.iterrows():
        lines.append(
            f"{row['Setting']} & {int(row['Requirements met'])} & {int(row['Conflicting exposure'])} & "
            f"{int(row['Decision-utility failure'])} & {int(row['Ranking-utility failure'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "ruap_threshold_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def build_foundation_sensitivity(proxy_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus_models = ["xgb_baseline", "tabpfn_baseline", "tabicl_baseline", "rc_stack", "rrc_stack"]
    for (dataset, model), model_frame in proxy_results.groupby(["Dataset", "Model"]):
        if model not in focus_models:
            continue
        baseline = model_frame[model_frame["Variant"] == "baseline"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for _, row in model_frame[model_frame["Variant"] != "baseline"].iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": row["Variant"],
                    "Model": model,
                    "AUCDelta": float(row["AUC"] - baseline_row["AUC"]),
                    "ECEDelta": float(row["ECE (10-bin)"] - baseline_row["ECE (10-bin)"]),
                    "CostDelta": float(row["DecisionCost5x"] - baseline_row["DecisionCost5x"]),
                }
            )
    return pd.DataFrame(rows)


def write_foundation_sensitivity_table(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    if sensitivity.empty:
        return
    rows = []
    for (dataset, variant), frame in sensitivity.groupby(["Dataset", "Variant"]):
        tabpfn = frame[frame["Model"] == "tabpfn_baseline"]
        tabicl = frame[frame["Model"] == "tabicl_baseline"]
        controls = frame[frame["Model"].isin(["xgb_baseline", "rc_stack", "rrc_stack"])]
        if tabpfn.empty or tabicl.empty or controls.empty:
            continue
        best_control = controls.loc[controls["AUCDelta"].idxmax()]
        rows.append(
            {
                "Dataset": dataset,
                "Variant": variant,
                "TabPFN_AUC": float(tabpfn.iloc[0]["AUCDelta"]),
                "TabPFN_Cost": float(tabpfn.iloc[0]["CostDelta"]),
                "TabICL_AUC": float(tabicl.iloc[0]["AUCDelta"]),
                "TabICL_Cost": float(tabicl.iloc[0]["CostDelta"]),
                "Control": best_control["Model"],
                "Control_AUC": float(best_control["AUCDelta"]),
                "Control_Cost": float(best_control["CostDelta"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "foundation_model_sensitivity_summary.csv", index=False)
    lines = [
        "\\begin{table}[t]",
        "\\caption{Foundation-model sensitivity under transformed proxies. Deltas are relative to the same model on the original table. The control column reports the best AUC delta among XGB, RC-Stack, and RRC-Stack.}",
        "\\label{tab:fm_sensitivity}",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Proxy & TabPFN $\\Delta$AUC & TabPFN $\\Delta$Cost & TabICL $\\Delta$AUC & TabICL $\\Delta$Cost & Ctrl. $\\Delta$AUC & Ctrl. $\\Delta$Cost \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{display(row['Dataset'], DISPLAY_DATASETS)} & {display(row['Variant'], DISPLAY_VARIANTS)} & "
            f"{fmt_delta(row['TabPFN_AUC'])} & {fmt_delta(row['TabPFN_Cost'])} & "
            f"{fmt_delta(row['TabICL_AUC'])} & {fmt_delta(row['TabICL_Cost'])} & "
            f"{fmt_delta(row['Control_AUC'])} & {fmt_delta(row['Control_Cost'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{small}",
            "\\end{center}",
            "\\vskip -0.08in",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "foundation_model_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RUA-P utility/exposure audit assets.")
    parser.add_argument("--proxy-root", default="outputs/proxy_transform_audit")
    parser.add_argument("--datasets", default="australian_credit,german_credit,compas_recidivism,adult_income,taiwan_default")
    parser.add_argument("--variants", default="baseline,numeric_noise_10,numeric_noise_20,coarsen_quartile,feature_mask_20,sensitive_mask")
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv(args.datasets)
    variants = parse_csv(args.variants)
    exposure = build_exposure_table(datasets, variants, args.seed, max_rows=args.max_rows)
    exposure.to_csv(output_dir / "ruap_exposure_metrics.csv", index=False)
    proxy_results = read_proxy_outputs(Path(args.proxy_root))
    report = build_ruap_report(proxy_results, exposure)
    report.to_csv(output_dir / "ruap_report_card.csv", index=False)
    write_latex_tables(report, output_dir)
    profile = build_ruap_certificate(report)
    profile.to_csv(output_dir / "ruap_certificate.csv", index=False)
    write_certificate_table(profile, output_dir)
    write_ruap_report_plot(report, output_dir)
    threshold_sensitivity = build_threshold_sensitivity(report)
    threshold_sensitivity.to_csv(output_dir / "ruap_threshold_sensitivity.csv", index=False)
    write_threshold_sensitivity_table(threshold_sensitivity, output_dir)
    sensitivity = build_foundation_sensitivity(proxy_results)
    sensitivity.to_csv(output_dir / "foundation_model_sensitivity.csv", index=False)
    write_foundation_sensitivity_table(sensitivity, output_dir)
    print(f"wrote {len(exposure)} exposure rows and {len(report)} report rows to {output_dir}")


if __name__ == "__main__":
    main()
