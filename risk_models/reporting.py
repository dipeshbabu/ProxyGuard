from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path):
    ensure_directory(path.parent)
    df.to_csv(path, index=False)


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    ensure_directory(path.parent)
    latex = df.to_latex(index=False, float_format=lambda value: f"{value:.3f}", caption=caption, label=label)
    path.write_text(latex, encoding="utf-8")


def plot_metric_summary(aggregate_metrics: pd.DataFrame, output_path: Path, metric: str = "AUC"):
    if aggregate_metrics.empty or metric not in aggregate_metrics.columns:
        return
    ensure_directory(output_path.parent)
    plot_df = aggregate_metrics.sort_values(metric, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["Model"], plot_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_reliability_diagram(y_true, y_prob, output_path: Path, n_bins: int = 10, title: Optional[str] = None):
    ensure_directory(output_path.parent)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_run_artifacts(
    run_dir: Path,
    split_metrics: Optional[pd.DataFrame] = None,
    aggregate_metrics: Optional[pd.DataFrame] = None,
    subgroup_metrics: Optional[pd.DataFrame] = None,
    feature_stability: Optional[pd.DataFrame] = None,
    audit_records: Optional[pd.DataFrame] = None,
):
    ensure_directory(run_dir)
    if split_metrics is not None:
        save_dataframe(split_metrics, run_dir / "split_metrics.csv")
    if aggregate_metrics is not None:
        save_dataframe(aggregate_metrics, run_dir / "aggregate_metrics.csv")
    if aggregate_metrics is not None and not aggregate_metrics.empty:
        save_latex_table(
            aggregate_metrics,
            run_dir / "aggregate_metrics.tex",
            caption="Aggregate benchmark metrics across repeated splits.",
            label="tab:aggregate_metrics",
        )
    if subgroup_metrics is not None and not subgroup_metrics.empty:
        save_dataframe(subgroup_metrics, run_dir / "subgroup_metrics.csv")
    if audit_records is not None and not audit_records.empty:
        save_dataframe(audit_records, run_dir / "audit_records.csv")
    if feature_stability is not None and not feature_stability.empty:
        save_dataframe(feature_stability, run_dir / "feature_stability.csv")
    if aggregate_metrics is not None:
        plot_metric_summary(aggregate_metrics, run_dir / "auc_summary.png", metric="AUC")
        plot_metric_summary(aggregate_metrics, run_dir / "ece_summary.png", metric="ECE (10-bin)")
