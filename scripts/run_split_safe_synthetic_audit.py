from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import clone_experiment_config, get_default_experiment_config, get_dataset_config, get_spotlight_model_configs
from risk_models.cv_runner import apply_optional_calibrator, fit_optional_calibrator, split_train_val_test
from risk_models.dataset import load_dataset
from risk_models.eval import aggregate_metrics, best_cost_threshold_from_val, best_f1_threshold_from_val, evaluate_predictions
from risk_models.model import build_model
from scripts.build_ruap_audit import DISPLAY_DATASETS


DEFAULT_DATASETS = [
    "australian_credit",
    "german_credit",
    "compas_recidivism",
    "heart_disease",
    "mammographic_mass",
]

DATASET_DISPLAY = {
    **DISPLAY_DATASETS,
    "heart_disease": "Heart",
    "mammographic_mass": "Mammo",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class CopulaClassState:
    columns: list[str]
    values: dict[str, np.ndarray]
    means: np.ndarray
    corr: np.ndarray


class ClassConditionalGaussianCopula:
    def __init__(self, random_state: int = 3407, shrinkage: float = 0.05):
        self.random_state = random_state
        self.shrinkage = shrinkage
        self.class_states_: dict[int, CopulaClassState] = {}
        self.class_probs_: dict[int, float] = {}
        self.columns_: list[str] = []

    @staticmethod
    def _fit_class(X: pd.DataFrame, shrinkage: float) -> CopulaClassState:
        columns = list(X.columns)
        z_columns = []
        values: dict[str, np.ndarray] = {}
        n = len(X)
        for column in columns:
            series = pd.to_numeric(X[column], errors="coerce")
            fill = float(series.median()) if series.notna().any() else 0.0
            array = series.fillna(fill).to_numpy(dtype=float)
            values[column] = np.sort(array)
            if np.nanstd(array) <= 1e-12 or n < 3:
                z_columns.append(np.zeros(n))
                continue
            ranks = pd.Series(array).rank(method="average").to_numpy()
            u = np.clip(ranks / (n + 1.0), 1e-4, 1.0 - 1e-4)
            z_columns.append(norm.ppf(u))
        z = np.column_stack(z_columns) if z_columns else np.empty((n, 0))
        means = z.mean(axis=0) if z.size else np.array([])
        if z.shape[1] <= 1 or n < 3:
            corr = np.eye(max(1, z.shape[1]))
        else:
            corr = np.corrcoef(z, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
            corr = 0.5 * (corr + corr.T)
            eigvals = np.linalg.eigvalsh(corr)
            if eigvals.min() < 1e-6:
                corr += np.eye(corr.shape[0]) * (1e-6 - eigvals.min())
        return CopulaClassState(columns=columns, values=values, means=means, corr=corr)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassConditionalGaussianCopula":
        self.columns_ = list(X.columns)
        y = pd.Series(y).astype(int)
        counts = y.value_counts(normalize=True)
        self.class_probs_ = {int(label): float(prob) for label, prob in counts.items()}
        self.class_states_ = {}
        for label in sorted(self.class_probs_):
            class_X = X.loc[y[y == label].index]
            if len(class_X) < 3:
                class_X = X
            self.class_states_[label] = self._fit_class(class_X, self.shrinkage)
        return self

    def sample(self, n_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(seed)
        labels = np.array(sorted(self.class_probs_))
        probs = np.array([self.class_probs_[int(label)] for label in labels], dtype=float)
        probs = probs / probs.sum()
        sampled_y = rng.choice(labels, size=n_rows, replace=True, p=probs).astype(int)
        X_parts = []
        y_parts = []
        for label in labels:
            n_label = int(np.sum(sampled_y == label))
            if n_label == 0:
                continue
            state = self.class_states_[int(label)]
            if len(state.columns) == 0:
                continue
            z = rng.multivariate_normal(state.means, state.corr, size=n_label, method="svd")
            u = np.clip(norm.cdf(z), 0.0, 1.0)
            sampled = {}
            for j, column in enumerate(state.columns):
                sampled[column] = np.quantile(state.values[column], u[:, j], method="nearest")
            X_parts.append(pd.DataFrame(sampled, columns=self.columns_))
            y_parts.append(pd.Series(np.full(n_label, int(label), dtype=int)))
        X_synth = pd.concat(X_parts, ignore_index=True) if X_parts else pd.DataFrame(columns=self.columns_)
        y_synth = pd.concat(y_parts, ignore_index=True) if y_parts else pd.Series(dtype=int)
        order = rng.permutation(len(y_synth))
        return X_synth.iloc[order].reset_index(drop=True), y_synth.iloc[order].reset_index(drop=True)


def membership_distance_auc(source: pd.DataFrame, holdout: pd.DataFrame, release: pd.DataFrame) -> float:
    if len(source) < 5 or len(holdout) < 5 or len(release) < 5:
        return float("nan")
    combined = pd.concat([source, holdout, release], ignore_index=True)
    values = combined.apply(pd.to_numeric, errors="coerce").fillna(combined.median(numeric_only=True)).fillna(0.0)
    std = values.std(axis=0).replace(0.0, 1.0)
    values = (values - values.mean(axis=0)) / std
    n_source = len(source)
    n_holdout = len(holdout)
    candidates = values.iloc[: n_source + n_holdout].to_numpy()
    release_matrix = values.iloc[n_source + n_holdout :].to_numpy()
    distances, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(release_matrix).kneighbors(candidates)
    labels = np.concatenate([np.ones(n_source), np.zeros(n_holdout)])
    try:
        return float(roc_auc_score(labels, -distances[:, 0]))
    except ValueError:
        return float("nan")


def get_model_config(model_name: str):
    matches = [config for config in get_spotlight_model_configs(include_tabpfn=False, include_tabicl=False) if config.name == model_name]
    if not matches:
        raise ValueError(f"Unknown or unsupported model for synthetic audit: {model_name}")
    return matches[0]


def fit_predict_metrics(model_config, X_train, y_train, X_val, y_val, X_test, y_test, exp_cfg, split_seed: int, model_name: str):
    model = build_model(model_config)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    p_val = model.predict_proba(X_val)[:, 1]
    calibrator = fit_optional_calibrator(p_val, y_val, exp_cfg, enabled=True)
    p_val_cal = apply_optional_calibrator(p_val, calibrator)
    threshold, f1_val = best_f1_threshold_from_val(y_val, p_val_cal)
    cost_thresholds = {
        fn_cost: best_cost_threshold_from_val(y_val, p_val_cal, fn_cost=fn_cost, fp_cost=1.0)
        for fn_cost in (2.0, 5.0, 10.0, 20.0)
    }
    inference_start = time.perf_counter()
    p_test = apply_optional_calibrator(model.predict_proba(X_test)[:, 1], calibrator)
    inference_time = time.perf_counter() - inference_start
    metrics = evaluate_predictions(y_test, p_test, threshold=threshold, train_time=train_time, inference_time=inference_time)
    for fn_cost, (cost_threshold, cost_val) in cost_thresholds.items():
        cost_metrics = evaluate_predictions(y_test, p_test, threshold=cost_threshold)
        cost_name = f"DecisionCost{int(fn_cost)}x"
        metrics[cost_name] = cost_metrics[cost_name]
        metrics[f"{cost_name}RelApproveAll"] = cost_metrics[f"{cost_name}RelApproveAll"]
        metrics[f"val_cost_{int(fn_cost)}x_best"] = cost_val
        metrics[f"cost_threshold_{int(fn_cost)}x"] = cost_threshold
    metrics["Model"] = model_name
    metrics["split_seed"] = split_seed
    metrics["val_F1_best"] = f1_val
    metrics["threshold"] = threshold
    metrics["calibration_applied"] = 1
    return metrics


def run_dataset(dataset_name: str, model_name: str, repeats: int, seed: int, synth_multiplier: float, output_dir: Path) -> pd.DataFrame:
    bundle = load_dataset(get_dataset_config(dataset_name))
    X = bundle["X"]
    y = bundle["y"].astype(int)
    exp_cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=repeats,
        calibration_method="temperature",
        run_subgroups=False,
        save_reliability=False,
        save_shap=False,
    )
    model_config = get_model_config(model_name)
    rows = []
    for k in range(repeats):
        split_seed = seed + k
        X_train, X_val_real, X_test, y_train, y_val_real, y_test = split_train_val_test(
            X,
            y,
            seed=split_seed,
            test_size=exp_cfg.test_size,
            val_size=exp_cfg.val_size,
        )
        real_metrics = fit_predict_metrics(
            model_config,
            X_train,
            y_train,
            X_val_real,
            y_val_real,
            X_test,
            y_test,
            exp_cfg,
            split_seed,
            model_name="real_train_xgb",
        )
        real_metrics["Dataset"] = dataset_name
        real_metrics["Variant"] = "real_train"
        rows.append(real_metrics)

        synth = ClassConditionalGaussianCopula(random_state=split_seed).fit(X_train, y_train)
        n_synth = max(len(X_train) + len(X_val_real), int(round(synth_multiplier * (len(X_train) + len(X_val_real)))))
        X_synth, y_synth = synth.sample(n_synth, seed=split_seed + 17)
        X_synth_train, X_synth_val, y_synth_train, y_synth_val = train_test_split(
            X_synth,
            y_synth,
            test_size=max(0.20, exp_cfg.val_size),
            stratify=y_synth,
            random_state=split_seed + 23,
        )
        synth_metrics = fit_predict_metrics(
            model_config,
            X_synth_train,
            y_synth_train,
            X_synth_val,
            y_synth_val,
            X_test,
            y_test,
            exp_cfg,
            split_seed,
            model_name="copula_synth_xgb",
        )
        synth_metrics["Dataset"] = dataset_name
        synth_metrics["Variant"] = "copula_synth"
        synth_metrics["MemberAUC"] = membership_distance_auc(X_train, X_val_real, X_synth)
        synth_metrics["SyntheticRows"] = len(X_synth)
        rows.append(synth_metrics)
        print(f"{dataset_name} split {k + 1}/{repeats}: real AUC={real_metrics['AUC']:.3f}, synth AUC={synth_metrics['AUC']:.3f}")

    split_df = pd.DataFrame(rows)
    split_path = output_dir / "split_safe_synthetic" / dataset_name / "split_metrics.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return split_df


def summarize(split_df: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_metrics(
        split_df,
        metric_columns=[
            "AUC",
            "Brier",
            "ECE (10-bin)",
            "DecisionCost5x",
            "DecisionCost10x",
            "TrainTimeSec",
            "InferenceTimeSec",
            "MemberAUC",
        ],
    )
    counts = split_df.groupby(["Dataset", "Model"]).size().rename("n_splits").reset_index()
    agg = agg.merge(counts, on=["Dataset", "Model"], how="left")
    rows = []
    for dataset, frame in agg.groupby("Dataset"):
        real = frame[frame["Model"] == "real_train_xgb"].iloc[0]
        synth = frame[frame["Model"] == "copula_synth_xgb"].iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "AUCDelta": synth["AUC"] - real["AUC"],
                "ECEDelta": synth["ECE (10-bin)"] - real["ECE (10-bin)"],
                "CostDelta": synth["DecisionCost5x"] - real["DecisionCost5x"],
                "MemberAUC": synth["MemberAUC"],
                "SynthAUC": synth["AUC"],
                "RealAUC": real["AUC"],
                "Splits": int(synth["n_splits"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Dataset").reset_index(drop=True)


def fmt_delta(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}"


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def write_latex(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\begin{center}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & $\\Delta$AUC & $\\Delta$ECE & $\\Delta$Cost5x & MemAUC & Splits \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{DATASET_DISPLAY.get(row['Dataset'], row['Dataset'])} & "
            f"{fmt_delta(row['AUCDelta'])} & {fmt_delta(row['ECEDelta'])} & "
            f"{fmt_delta(row['CostDelta'])} & {fmt(row['MemberAUC'])} & {int(row['Splits'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{small}",
            "\\end{center}",
            "\\caption{Split-safe Gaussian-copula synthetic release audit. The synthesizer is fit only on each real training fold, samples labeled synthetic train/validation rows, trains XGBoost on synthetic train rows, calibrates and selects thresholds on synthetic validation rows, and evaluates on the real held-out test fold. Deltas compare against XGBoost trained on the real training fold for the same splits. MemAUC is a nearest-release membership-distance attack distinguishing source training rows from real validation rows; lower is better.}\\label{tab:split_safe_synth_app}",
            "\\end{table}",
        ]
    )
    (output_dir / "split_safe_synthetic_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split-safe synthetic release utility and membership audit.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--model", default="xgb_baseline")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--synth-multiplier", type=float, default=1.0)
    parser.add_argument("--output-root", default="outputs/split_safe_synthetic_audit")
    parser.add_argument("--asset-dir", default="paper_assets/ruap_audit")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    all_rows = []
    for dataset_name in parse_csv(args.datasets):
        all_rows.append(run_dataset(dataset_name, args.model, args.repeats, args.seed, args.synth_multiplier, output_root))
    split_df = pd.concat(all_rows, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_root / "split_metrics.csv", index=False)
    summary = summarize(split_df)
    summary.to_csv(output_root / "split_safe_synthetic_summary.csv", index=False)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(asset_dir / "split_safe_synthetic_summary.csv", index=False)
    write_latex(summary, asset_dir)
    print(f"wrote split-safe synthetic audit for {len(summary)} datasets")


if __name__ == "__main__":
    main()
