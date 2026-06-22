from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import (
    clone_experiment_config,
    get_default_experiment_config,
    get_dataset_config,
    get_midas_model_configs,
)
from risk_models.cv_runner import run_benchmark


MODEL_LABELS = {
    "logreg_baseline": "LogReg",
    "xgb_baseline": "XGB",
    "lightgbm_baseline": "LightGBM",
    "catboost_baseline": "CatBoost",
    "compact_xgb": "Compact XGB",
    "tabpfn_baseline": "TabPFN",
    "tabicl_baseline": "TabICL",
}


def collect_single_split_artifacts(dataset_name: str):
    cfg = clone_experiment_config(
        get_default_experiment_config(),
        n_repeats=1,
        output_root="outputs/midas_reliability_audit",
        calibration_method="temperature",
        save_reliability=False,
        save_shap=False,
        run_subgroups=False,
    )
    result = run_benchmark(
        get_dataset_config(dataset_name),
        get_midas_model_configs(include_tabpfn=True),
        cfg,
        mode="single_split",
    )
    return result["artifacts"]


def plot_dataset(axis, dataset_name: str, title: str):
    artifacts = collect_single_split_artifacts(dataset_name)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Perfect")
    for artifact in artifacts:
        model_name = artifact["Model"]
        prob_true, prob_pred = calibration_curve(
            artifact["y_test"],
            artifact["p_test"],
            n_bins=10,
            strategy="uniform",
        )
        axis.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=MODEL_LABELS.get(model_name, model_name),
        )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Predicted default probability")
    axis.set_ylabel("Empirical default rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2, linewidth=0.5)


def main() -> None:
    figure_path = Path("paper/figures/midas_reliability_audit.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True, sharey=True)
    plot_dataset(axes[0], "german_credit", "German Credit")
    plot_dataset(axes[1], "give_me_some_credit", "GMSC")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(figure_path.resolve())


if __name__ == "__main__":
    main()
