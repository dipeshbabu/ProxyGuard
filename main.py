# main.py
from eval import ModelEvaluator, ResultVisualizer
from model import SmallDataCreditPipeline
from dataset import preprocess_data
from configs import (
    DATA_PATH, RESULTS_CSV, RESULTS_TEX, METRICS_PNG, BEST_MODEL_PKL, SEED
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings("ignore")


# -----------------------
# Experiment toggles
# -----------------------
USE_PSEUDO_LABELS = False     # set False to disable pseudo-labeling in the pipeline
RUN_SANITY_CHECKS = True      # shuffle-label & high-score leakage warnings
RELIABILITY_MODELS = [
    "Small-Data Pipeline (Full)", "XGBoost Baseline"
]


def _warn_if_suspicious(metrics_row: dict, name: str, auc_hi=0.98, acc_hi=0.98):
    """Heuristics to flag likely leakage/overfit."""
    auc = metrics_row.get("AUC", np.nan)
    acc = metrics_row.get("Accuracy", np.nan)
    ece = metrics_row.get(
        "ECE (10-bin)", np.nan) or metrics_row.get("ECE", np.nan)
    if (auc >= auc_hi and acc >= acc_hi) or (auc >= 0.999):
        print(f"[warning] {name}: extremely high AUC/Accuracy ({auc:.3f}/{acc:.3f}). "
              "This often indicates leakage when labels are derived from predictors.")
    if np.isfinite(ece) and ece < 0.02 and auc > 0.95:
        print(f"[note] {name}: very low ECE with very high AUC ({auc:.3f}). "
              "Double-check that no label-defining features leak into X.")


def _shuffle_label_sanity(model, X_train, y_train, X_test, y_test, name="Model"):
    """Train/evaluate with shuffled labels to gauge baseline; AUC should ~ 0.5."""
    y_shuf = y_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    X_tr = X_train.reset_index(drop=True)
    try:
        start = time.time()
        model.fit(X_tr, y_shuf)
        dur = time.time() - start
        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(
            f"[sanity] {name} (shuffle-label): AUC={auc:.3f} | TrainTime={dur:.2f}s")
        if auc > 0.6:
            print("  -> AUC > 0.6 on shuffled labels is suspicious. Check leakage/bugs.")
    except Exception as e:
        print(f"[sanity] {name} (shuffle-label) failed: {e}")


def _build_models(scale_pos_weight: float):
    """Define models inline with class-imbalance handling."""
    models = {
        'Small-Data Pipeline (Full)': SmallDataCreditPipeline(use_all_techniques=True),
        'Small-Data Pipeline (Basic)': SmallDataCreditPipeline(use_all_techniques=False),
        'XGBoost Baseline': XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight='balanced',
            random_state=SEED
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear',
            random_state=SEED
        ),
    }
    return models


def run_experiment(data_path: str = DATA_PATH):
    # -----------------------
    # Load & split
    # -----------------------
    X, y, numeric_cols = preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Class-imbalance handling params for baselines
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(max(1.0, neg / max(1, pos)))

    # Build an unlabeled split from training for pseudo-labeling demos
    if USE_PSEUDO_LABELS:
        X_train_labeled, X_unlabeled, y_train_labeled, _ = train_test_split(
            X_train, y_train, test_size=0.3, stratify=y_train, random_state=SEED
        )
    else:
        X_train_labeled, y_train_labeled = X_train, y_train
        X_unlabeled = None

    # -----------------------
    # Train & evaluate
    # -----------------------
    models = _build_models(scale_pos_weight)
    evaluator = ModelEvaluator()
    results_rows = []

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        start = time.time()

        if isinstance(model, SmallDataCreditPipeline):
            model.fit(X_train_labeled, y_train_labeled,
                      X_unlabeled if USE_PSEUDO_LABELS else None)
        else:
            model.fit(X_train, y_train)

        train_time = time.time() - start

        metrics = evaluator.evaluate(
            model, X_train, y_train, X_test, y_test, name)
        metrics['Train Time (s)'] = train_time
        results_rows.append(metrics)

        # quick leakage/overfit heuristic
        _warn_if_suspicious(metrics, name)

        # optional shuffle-label sanity test (uses a *fresh* model instance)
        if RUN_SANITY_CHECKS:
            try:
                if isinstance(model, SmallDataCreditPipeline):
                    sm = SmallDataCreditPipeline(
                        use_all_techniques=model.use_all_techniques)
                else:
                    # re-create a similar estimator
                    sm = type(model)(**getattr(model, "get_params", lambda: {})()) \
                        if hasattr(model, "get_params") else type(model)()
                _shuffle_label_sanity(
                    sm, X_train, y_train, X_test, y_test, name)
            except Exception as e:
                print(f"[sanity] Could not reinstantiate {name}: {e}")

    # -----------------------
    # Save tables / plots
    # -----------------------
    results_df = pd.DataFrame(results_rows).set_index('Model')
    results_df_sorted = results_df.sort_values('AUC', ascending=False)
    results_df_sorted.to_csv(RESULTS_CSV)

    latex_cols = ['AUC', 'AUPRC', 'Accuracy', 'F1', 'F1@t*',
                  't*', 'Precision', 'Recall', 'Brier', 'ECE (10-bin)']
    latex_cols = [c for c in latex_cols if c in results_df_sorted.columns]
    latex_df = results_df_sorted[latex_cols].round(3)
    with open(RESULTS_TEX, 'w') as f:
        f.write(latex_df.to_latex(index=True,
                                  caption='Small-data results on German Credit (n=1000).',
                                  label='tab:small_data', escape=False))

    print("\nFinal Results:")
    print(results_df_sorted)

    viz = ResultVisualizer()
    viz.plot_metrics_comparison(results_df.reset_index(), out_path=METRICS_PNG)

    for label in RELIABILITY_MODELS:
        if label in models:
            mdl = models[label]
            try:
                y_prob = mdl.predict_proba(X_test)[:, 1]
                safe = label.replace(' ', '_').replace(
                    '(', '').replace(')', '')
                viz.plot_reliability(y_test, y_prob, name=safe)
            except Exception as e:
                print(f"[reliability] {label}: {e}")

    # SHAP summary (skip safely inside viz if unsupported)
    for name in models:
        try:
            viz.plot_shap_summary(models[name], X_test, name)
        except Exception:
            pass

    best_model_name = results_df['AUC'].idxmax()
    joblib.dump(models[best_model_name], BEST_MODEL_PKL)
    print(f"[done] wrote results & artifacts. Best model: {best_model_name}")
    return results_df_sorted


if __name__ == "__main__":
    run_experiment()
