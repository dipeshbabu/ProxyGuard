# cv_runner.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss)

from eval import compute_ece, best_f1_threshold_from_val, TemperatureScaler
from configs import SEED


def evaluate_once(models: Dict[str, object], X, y, n_inner_val=0.2, calibrate=True):
    """One 80/20 split; tune threshold on a validation slice; optional temperature scaling."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    # small validation taken from training to tune t* and calibration
    Xtr_fit, Xval, ytr_fit, yval = train_test_split(Xtr, ytr, test_size=n_inner_val,
                                                    stratify=ytr, random_state=SEED+1)
    rows = []
    for name, model in models.items():
        # fit
        model.fit(Xtr_fit, ytr_fit)
        # val preds
        p_val = model.predict_proba(Xval)[:, 1]
        t_star, f1_star = best_f1_threshold_from_val(yval, p_val)
        # temperature scaling
        if calibrate:
            ts = TemperatureScaler().fit(p_val, yval)
        # test preds (w/ and w/o calibration)
        p_test = model.predict_proba(Xte)[:, 1]
        if calibrate:
            p_test_cal = ts.transform(p_test)
        else:
            p_test_cal = p_test

        def row(prefix, probs):
            yhat = (probs >= t_star).astype(int)
            return {
                'Model': name + prefix,
                'AUC': roc_auc_score(yte, probs),
                'AUPRC': average_precision_score(yte, probs),
                'Accuracy': accuracy_score(yte, yhat),
                'F1': f1_score(yte, yhat),
                'F1@t*': f1_star,
                't*': t_star,
                'Precision': precision_score(yte, yhat, zero_division=0),
                'Recall': recall_score(yte, yhat, zero_division=0),
                'Brier': brier_score_loss(yte, probs),
                'ECE (10-bin)': compute_ece(yte, probs, 10),
            }
        rows.append(row("", p_test))
        rows.append(row(" +Cal", p_test_cal))
    return rows


def evaluate_k_splits(models, X, y, K=10, seed=SEED, calibrate=True):
    all_rows = []
    rng = np.random.default_rng(seed)
    for k in range(K):
        rs = int(rng.integers(0, 10_000))
        # change seeds inside models if needed, or rely on deterministic code paths
        rows = evaluate_once(models, X, y, calibrate=calibrate)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    # Aggregate mean±std and 95% CI by Model

    def agg(group):
        out = {}
        for col in ['AUC', 'AUPRC', 'Accuracy', 'F1', 'F1@t*', 't*', 'Precision', 'Recall', 'Brier', 'ECE (10-bin)']:
            xs = group[col].values
            mu = xs.mean()
            sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
            out[col] = mu
            out[col+' (std)'] = sd
        return pd.Series(out)
    return df.groupby('Model').apply(agg).reset_index()
