# eval.py
import time
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        mask = (inds == b)
        nb = np.count_nonzero(mask)
        if nb == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (nb / N) * abs(acc - conf)
    return float(ece)


class ModelEvaluator:
    def __init__(self):
        self.results = []
        self.shap_values = {}

    def _bootstrap_ci(self, y_true, y_pred, metric, n_iter=1000):
        rng = np.random.default_rng(3407)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        scores = []
        for _ in range(n_iter):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            yt, yp = y_true[idx], y_pred[idx]
            try:
                s = metric(yt, yp)
                if np.isfinite(s):
                    scores.append(float(s))
            except Exception:
                continue
        if not scores:
            return (np.nan, np.nan)
        return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))

    @staticmethod
    def _best_f1_threshold(y_true, y_prob):
        """Return (t*, F1@t*)."""
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        ts = np.linspace(0.0, 1.0, 101)
        best_t, best_f1 = 0.5, 0.0
        for t in ts:
            y_hat = (y_prob >= t).astype(int)
            try:
                f1 = f1_score(y_true, y_hat)
            except Exception:
                f1 = 0.0
            if np.isfinite(f1) and f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        return best_t, best_f1

    def evaluate(self, model, X, y, X_test, y_test, name):
        # Always return at least a stub row with Model=name
        stub = {'Model': name}
        try:
            start_time = time.time()
            # OK even if already fit; consistent with your earlier flow
            model.fit(X, y)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # core metrics
            metrics = {
                'Model': name,
                'AUC': roc_auc_score(y_test, y_prob),
                'AUPRC': average_precision_score(y_test, y_prob),
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'Brier': brier_score_loss(y_test, y_prob),
                'ECE (10-bin)': compute_ece(y_test, y_prob, n_bins=10),
                'Train Time (s)': train_time,
                'AUC CI': self._bootstrap_ci(y_test, y_prob, roc_auc_score),
                'AUPRC CI': self._bootstrap_ci(y_test, y_prob, average_precision_score),
                'Accuracy CI': self._bootstrap_ci(y_test, y_pred, accuracy_score)
            }

            # add F1@t*
            try:
                t_star, f1_star = self._best_f1_threshold(y_test, y_prob)
                metrics['F1@t*'] = f1_star
                metrics['t*'] = t_star
            except Exception as e:
                # do not fail evaluation if threshold search breaks
                pass

            # optional SHAP
            if hasattr(model, "get_shap_values") and getattr(model, "is_fitted", False):
                try:
                    self.shap_values[name] = model.get_shap_values(X_test)
                    print(f"SHAP analysis successful for {name}")
                    print(
                        f"Processed features: {getattr(model, 'feature_names', None)}")
                except Exception as e:
                    print(f"SHAP failed for {name}: {str(e)}")

            self.results.append(metrics)
            return metrics

        except Exception as e:
            print(f"Evaluation failed for {name}: {str(e)}")
            # return a stub row so downstream aggregation doesn't crash
            self.results.append(stub)
            return stub


@dataclass
class SplitResult:
    name: str
    metrics: Dict[str, float]


def mean_std_ci(xs, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    mu = float(xs.mean())
    sd = float(xs.std(ddof=1)) if len(xs) > 1 else 0.0
    # normal approx CI
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2.0)
    ci = (mu - z*sd/np.sqrt(max(1, len(xs))),
          mu + z*sd/np.sqrt(max(1, len(xs))))
    return mu, sd, ci


def best_f1_threshold_from_val(y_val, p_val):
    ts = np.linspace(0.0, 1.0, 101)
    best_t, best_f1 = 0.5, 0.0
    for t in ts:
        f1 = f1_score(y_val, (p_val >= t).astype(int))
        if np.isfinite(f1) and f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


class TemperatureScaler:
    """Binary temperature scaling on logits; optimizes NLL on validation."""

    def __init__(self):
        self.T_ = 1.0

    @staticmethod
    def _logit(p, eps=1e-8):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))

    @staticmethod
    def _sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def fit(self, p_val, y_val):
        z = self._logit(p_val)
        # optimize T > 0 to minimize logloss(sigmoid(z/T), y)
        import scipy.optimize as spo

        def obj(logT):
            T = np.exp(logT)
            q = self._sigmoid(z / T)
            return log_loss(y_val, q)
        res = spo.minimize_scalar(obj, bounds=(-3, 3), method="bounded")
        self.T_ = float(np.exp(res.x)) if res.success else 1.0
        return self

    def transform(self, p):
        z = self._logit(p)
        return self._sigmoid(z / self.T_)


class ResultVisualizer:
    def plot_metrics_comparison(self, results_df, out_path="metrics_comparison.png"):
        try:
            metrics_to_plot = ['AUC', 'Accuracy', 'F1']
            results_df.set_index('Model')[metrics_to_plot].plot(
                kind='bar', figsize=(12, 6))
            plt.title('Model Performance Comparison')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        except Exception as e:
            print(f"Metric plot failed: {str(e)}")

    def plot_shap_summary(self, model, X_test, name):
        try:
            if hasattr(model, "get_shap_values") and getattr(model, "is_fitted", False):
                shap_values = model.get_shap_values(X_test)
                X_aligned = model._align_features(X_test)
                plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values, X_aligned, feature_names=model.feature_names,
                                  plot_type='bar', show=False)
                plt.title(f'SHAP Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f'shap_{name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved SHAP plot for {name}")
        except Exception as e:
            print(f"SHAP visualization failed for {name}: {e}")

    def plot_reliability(self, y_true, y_prob, name, n_bins=10):
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform')
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        plt.plot(prob_pred, prob_true, marker='o')
        plt.xlabel('Predicted probability')
        plt.ylabel('Empirical frequency')
        plt.title(f'Reliability Diagram - {name}')
        plt.tight_layout()
        plt.savefig(f'reliability_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
