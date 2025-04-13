# -*- coding: utf-8 -*-
# eval.py - Evaluation and interpretation

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats
from configs import RANDOM_STATE, N_BOOTSTRAP_ITERATIONS


def analyze_performance(y_true, y_pred, model_name):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred)
    }

    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    return metrics


def bootstrap_metric_comparison(y_true, y_pred1, y_pred2, metric_func, n_iterations=N_BOOTSTRAP_ITERATIONS):
    np.random.seed(RANDOM_STATE)
    metric_diffs = []

    for _ in range(n_iterations):
        indices = np.random.choice(
            range(len(y_true)), size=len(y_true), replace=True)
        try:
            m1 = metric_func(y_true[indices], y_pred1[indices])
            m2 = metric_func(y_true[indices], y_pred2[indices])
            metric_diffs.append(m1 - m2)
        except:
            continue

    if not metric_diffs:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    metric_diffs = np.array(metric_diffs)
    t_stat, p_value = stats.ttest_1samp(metric_diffs, 0)
    ci_lower, ci_upper = np.percentile(metric_diffs, [2.5, 97.5])
    mean_diff = metric_diffs.mean()

    return t_stat, p_value, ci_lower, ci_upper, mean_diff


def interpret_shap(model, X):
    X_mod = X.copy()
    if model.use_segmentation and 'Segment' not in X_mod.columns:
        try:
            X_mod['Segment'] = model.segmenter.predict(X_mod)
        except:
            X_mod['Segment'] = 0

    shap_values = model.explainer.shap_values(X_mod[model.selected_features])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_summary = pd.DataFrame({
        'Feature': model.selected_features,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    print("SHAP Feature Importance Summary:")
    print(shap_summary)

    plt.figure()
    shap.summary_plot(
        shap_values, X_mod[model.selected_features], plot_type='bar')
    plt.title("SHAP Feature Importance")
    plt.show()


def interpret_results(metrics, model_name):
    """Print a textual interpretation of model performance."""
    print(f"--- {model_name} Performance Summary ---")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
    print(f"F1 Score: {metrics['F1']:.2f}")
    print(f"AUC: {metrics['AUC']:.2f}")
    if metrics['Accuracy'] > 0.90:
        print("The model demonstrates high predictive accuracy, indicating a strong fit to the data.")
    else:
        print("The model's performance indicates room for improvement in capturing the underlying risk patterns.")


def plot_comparative_metrics(results_list, metric_name):
    labels = [res['Model'] for res in results_list]
    values = [res[metric_name] for res in results_list]
    times = [res['Time'] for res in results_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(labels, values, s=np.array(times)*50, c='blue', alpha=0.6)

    for i, (label, value, time) in enumerate(zip(labels, values, times)):
        plt.text(i, value + 0.02, f"{time:.1f}s", ha='center')

    plt.title(f"Model Comparison: {metric_name}")
    plt.xlabel("Model Configuration")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.show()
