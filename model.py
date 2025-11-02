from matplotlib import scale
import numpy as np
import pandas as pd
import itertools
import shap
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from configs import SEED


# --- Stability-Aware Feature Selector ---
class StabilityAwareFeatureSelector:
    def __init__(self, n_features=20, n_bootstrap=40, stability_threshold=0.6,
                 required_features=None):
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.required_features = list(required_features or [])
        self.selected_features = None
        self.feature_stability_scores = None

    def fit(self, X, y):
        req = [c for c in self.required_features if c in X.columns]
        optional = [c for c in X.columns if c not in req]
        k_opt = max(0, self.n_features - len(req))

        if len(optional) == 0:
            self.selected_features = req if req else list(X.columns)[
                :self.n_features]
            return self

        counts = np.zeros(len(optional))
        mi_acc = np.zeros(len(optional))

        for b in range(self.n_bootstrap):
            n = max(1, int(0.8 * len(y)))
            idx = np.random.choice(len(y), n, replace=True)
            Xb, yb = X.iloc[idx], y.iloc[idx]
            mi = mutual_info_classif(Xb[optional], yb, random_state=3407 + b)
            mi_acc += mi
            counts[np.argsort(-mi)[:k_opt]] += 1

        stab = counts / self.n_bootstrap
        avg_mi = mi_acc / self.n_bootstrap
        stable_mask = stab >= self.stability_threshold

        if stable_mask.sum() < k_opt:
            score = stab * avg_mi
            top_idx = np.argsort(-score)[:k_opt]
        else:
            si = np.where(stable_mask)[0]
            top_idx = si[np.argsort(-avg_mi[si])[:k_opt]]

        selected_opt = [optional[i] for i in top_idx]
        self.selected_features = (req + selected_opt)[:self.n_features]

        if not self.selected_features:
            self.selected_features = list(X.columns)[:self.n_features]

        return self

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("fit first")
        cols = [c for c in self.selected_features if c in X.columns]
        return X[cols]


# --- Data-Efficient Ensemble ---
class DataEfficientEnsemble:
    def __init__(self, confidence_threshold=0.92, n_base_models=5, var_cut=0.04):
        self.confidence_threshold = confidence_threshold
        self.n_base_models = n_base_models
        self.var_cut = var_cut
        self.base_models = []
        self.final_model = None
        self.pseudo_label_stats = {}

    def _pos_weight(self, y):
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return float(max(1.0, neg / max(1, pos)))

    def _models(self, y=None):
        spw = self._pos_weight(y) if y is not None else 1.0
        return [
            XGBClassifier(n_estimators=50,
                          max_depth=3,
                          learning_rate=0.1,
                          subsample=0.7,
                          random_state=SEED,
                          eval_metric='logloss',
                          scale_pos_weight=spw),
            XGBClassifier(n_estimators=100,
                          max_depth=2,
                          learning_rate=0.05,
                          subsample=0.8,
                          random_state=SEED+1,
                          eval_metric='logloss',
                          scale_pos_weight=spw),
            RandomForestClassifier(n_estimators=50,
                                   max_depth=4,
                                   min_samples_split=5,
                                   random_state=SEED,
                                   class_weight='balanced'),
            LogisticRegression(C=1.0,
                               max_iter=1000,
                               random_state=SEED,
                               class_weight='balanced'),
            XGBClassifier(n_estimators=75,
                          max_depth=3,
                          learning_rate=0.08,
                          colsample_bytree=0.8,
                          random_state=SEED+2,
                          eval_metric='logloss',
                          scale_pos_weight=spw),
            XGBClassifier(n_estimators=2000,
                          max_depth=3,
                          learning_rate=0.02,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          min_child_weight=5,
                          reg_lambda=2.0,
                          random_state=SEED,
                          eval_metric='logloss')
        ][:self.n_base_models]

    def fit(self, Xl, yl, Xu=None):
        # build base models with class imbalance handled
        self.base_models = self._models(yl)
        for m in self.base_models:
            m.fit(Xl, yl)

        # choose pos_weight for the final model as well
        spw = self._pos_weight(yl)

        if Xu is not None and len(Xu) > 0:
            probs = self._avg_proba(Xu)
            maxp = probs.max(axis=1)
            preds = probs.argmax(axis=1)
            allp = np.array([m.predict_proba(Xu)[:, 1]
                            for m in self.base_models])
            var = allp.var(axis=0)
            mask = (maxp > self.confidence_threshold) & (var < self.var_cut)

            self.pseudo_label_stats = {
                'n_unlabeled': int(len(Xu)),
                'n_confident': int((maxp > self.confidence_threshold).sum()),
                'n_low_variance': int((var < 0.1).sum()),
                'n_used': int(mask.sum()),
                'avg_confidence': float(maxp[mask].mean()) if mask.any() else 0.0
            }

            if mask.any():
                Xc = pd.concat([Xl, Xu[mask]])
                yc = np.concatenate([yl, preds[mask]])
                self.final_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=SEED,
                    eval_metric='logloss',
                    scale_pos_weight=spw
                ).fit(Xc, yc)
            else:
                self.final_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=SEED,
                    eval_metric='logloss',
                    scale_pos_weight=spw
                ).fit(Xl, yl)
        else:
            self.final_model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                eval_metric='logloss',
                scale_pos_weight=spw
            ).fit(Xl, yl)
        return self

    def _avg_proba(self, X):
        P = np.zeros((len(X), 2))
        for m in self.base_models:
            P += m.predict_proba(X)
        return P / len(self.base_models)

    def predict(self, X):
        return self.final_model.predict(X) if self.final_model is not None else self._avg_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.final_model.predict_proba(X) if self.final_model is not None else self._avg_proba(X)


# --- Bayesian Uncertainty Estimator ---
class BayesianUncertaintyEstimator:
    def __init__(self, n_models=10, subsample_ratio=0.8):
        self.n_models = n_models
        self.subsample_ratio = subsample_ratio
        self.models = []
        self.model_weights = None

    def fit(self, X, y):
        vals = []
        for i in range(self.n_models):
            n = max(1, int(self.subsample_ratio * len(y)))
            idx = np.random.choice(len(y), n, replace=True)
            val_idx = np.setdiff1d(np.arange(len(y)), idx)
            Xt, yt = X.iloc[idx], y.iloc[idx]
            m = XGBClassifier(
                n_estimators=50 + np.random.randint(0, 100),
                max_depth=2 + np.random.randint(0, 3),
                learning_rate=0.01 + np.random.rand() * 0.09,
                subsample=0.6 + np.random.rand() * 0.3,
                colsample_bytree=0.6 + np.random.rand() * 0.3,
                random_state=SEED + i,
                eval_metric='logloss'
            )
            m.fit(Xt, yt)
            self.models.append(m)
            if len(val_idx) > 0:
                Xv, yv = X.iloc[val_idx], y.iloc[val_idx]
                p = m.predict_proba(Xv)[:, 1]
                vals.append(roc_auc_score(yv, p))
            else:
                vals.append(0.5)
        vs = np.array(vals)
        self.model_weights = np.exp(vs) / np.exp(vs).sum()
        return self

    def predict_with_uncertainty(self, X):
        P = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        mean = np.average(P, axis=0, weights=self.model_weights)
        epi = np.sqrt(np.average((P - mean) ** 2, axis=0,
                      weights=self.model_weights))
        std = P.std(axis=0)
        return {
            'mean': mean,
            'epistemic_uncertainty': epi,
            'predictive_std': std,
            'ci_lower': np.percentile(P, 2.5, axis=0),
            'ci_upper': np.percentile(P, 97.5, axis=0),
            'all_predictions': P
        }


# --- Compositional Feature Engineer ---
class CompositionalFeatureEngineer:
    def __init__(self, max_features=5, min_mi_improvement=0.01):
        self.max_features = max_features
        self.min_mi_improvement = min_mi_improvement
        self.generated_features = {}
        self.feature_importance = {}

    def _safe_divide(self, x, y):
        return x / (y + 1e-8)

    def _create_financial_features(self, X):
        new = {}
        if 'Credit amount' in X.columns and 'Duration' in X.columns:
            new['payment_burden'] = X['Credit amount'] / (X['Duration'] + 1)
        if 'Credit amount' in X.columns and 'Age' in X.columns:
            new['credit_to_age'] = X['Credit amount'] / (X['Age'] + 1)
        if 'Saving accounts' in X.columns and 'Checking account' in X.columns:
            new['account_balance'] = X['Saving accounts'] + X['Checking account']
            new['account_ratio'] = self._safe_divide(
                X['Saving accounts'], X['Checking account'])
        if 'Business_Age' in X.columns and 'Credit amount' in X.columns:
            new['maturity_credit'] = X['Business_Age'] * \
                np.log1p(X['Credit amount'])
        return new

    def _numeric_safe_cols(self, X):
        SAFE_BLOCKLIST = {'Credit amount', 'Duration',
                          'Saving accounts', 'Checking account'}
        SAFE_PREFIX_BLOCKLIST = ('Purpose_',)

        cols = X.select_dtypes(include=[np.number]).columns
        cols = [c for c in cols if c not in SAFE_BLOCKLIST and not any(
            c.startswith(p) for p in SAFE_PREFIX_BLOCKLIST)]
        return cols

    def fit(self, X, y):
        nums = self._numeric_safe_cols(X)
        C = self._create_financial_features(X)
        C = {}
        for a, b in itertools.combinations(nums[:10], 2):
            C[f'{a}_times_{b}'] = X[a] * X[b]
            C[f'{a}_over_{b}'] = X[a] / (X[b] + 1e-8)
            C[f'{a}_minus_{b}'] = (X[a] - X[b]).abs()
        base = mutual_info_classif(X, y).mean()
        for name, vals in C.items():
            mi = mutual_info_classif(vals.values.reshape(-1, 1), y)[0]
            if mi > base + self.min_mi_improvement:
                self.generated_features[name] = vals
                self.feature_importance[name] = mi
        if len(self.generated_features) > self.max_features:
            top = sorted(self.feature_importance.items(),
                         key=lambda x: x[1], reverse=True)[:self.max_features]
            self.generated_features = {
                k: self.generated_features[k] for k, _ in top}
        return self

    def transform(self, X):
        Xn = X.copy()
        for name, _ in self.generated_features.items():
            if '_times_' in name:
                a, b = name.split('_times_')
                Xn[name] = X[a] * X[b]
            elif '_over_' in name:
                a, b = name.split('_over_')
                Xn[name] = X[a] / (X[b] + 1e-8)
            elif '_minus_' in name:
                a, b = name.split('_minus_')
                Xn[name] = (X[a] - X[b]).abs()
            else:
                feats = self._create_financial_features(X)
                if name in feats:
                    Xn[name] = feats[name]
        return Xn


# --- Small-Data Pipeline (with SHAP support) ---
class SmallDataCreditPipeline:
    def __init__(self, use_all_techniques=True):
        self.use_all_techniques = use_all_techniques
        self.feature_selector = StabilityAwareFeatureSelector(
            n_features=20, n_bootstrap=30, required_features=[]
        )
        self.feature_engineer = CompositionalFeatureEngineer(max_features=8)
        self.ensemble = DataEfficientEnsemble(confidence_threshold=0.85)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(n_models=10)
        self.scaler = StandardScaler()

        # attrs needed by eval.py SHAP path
        self.explainer = None
        self.is_fitted = False
        self.selected_features = None
        self.feature_names = None
        self.numeric_cols = None

    def fit(self, Xtr, ytr, Xunl=None):
        # 1) Feature engineering
        if self.use_all_techniques:
            self.feature_engineer.fit(Xtr, ytr)
            Xtr = self.feature_engineer.transform(Xtr)
            if Xunl is not None:
                Xunl = self.feature_engineer.transform(Xunl)

        # 2) Feature selection
        self.feature_selector.fit(Xtr, ytr)
        Xtr_sel = self.feature_selector.transform(Xtr)
        self.selected_features = list(Xtr_sel.columns)
        self.feature_names = list(self.selected_features)

        # 3) Scaling
        nums = Xtr_sel.select_dtypes(include=[np.number]).columns
        self.numeric_cols = list(nums)
        Xtr_scaled = Xtr_sel.copy()
        Xtr_scaled[nums] = self.scaler.fit_transform(Xtr_sel[nums])

        # 4) Train ensemble (+ pseudo-labeling if provided)
        if Xunl is not None and self.use_all_techniques:
            Xunl_sel = self.feature_selector.transform(Xunl)
            Xunl_sel = Xunl_sel.copy()
            Xunl_sel[nums] = self.scaler.transform(Xunl_sel[nums])
            self.ensemble.fit(Xtr_scaled, ytr, Xunl_sel)
        else:
            self.ensemble.fit(Xtr_scaled, ytr)

        # 5) (Optional) Train uncertainty estimator
        if self.use_all_techniques:
            self.uncertainty_estimator.fit(Xtr_scaled, ytr)

        # 6) Build SHAP explainer if final model is tree-based
        try:
            if getattr(self.ensemble, "final_model", None) is not None:
                self.explainer = shap.TreeExplainer(self.ensemble.final_model)
            else:
                self.explainer = None
        except Exception:
            self.explainer = None

        self.is_fitted = True
        return self

    # --- internal prep used by predict paths ---
    def _prep(self, X):
        Xp = X.copy()
        if self.use_all_techniques:
            Xp = self.feature_engineer.transform(Xp)
        Xp = self.feature_selector.transform(Xp)
        Xp[self.numeric_cols] = self.scaler.transform(Xp[self.numeric_cols])
        return Xp

    # --- required by eval.ResultVisualizer ---
    def _align_features(self, X):
        """
        Align columns to training feature set order and fill any missing columns with zeros.
        """
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Pipeline not fitted or feature_names missing.")
        Xp = self._prep(X)
        for col in self.feature_names:
            if col not in Xp.columns:
                Xp[col] = 0
        return Xp[self.feature_names]

    def get_shap_values(self, X):
        """
        Return SHAP values for aligned features, compatible with eval.py.
        """
        if self.explainer is None:
            raise ValueError(
                "SHAP explainer not available (model not tree-based or not fitted).")
        X_aligned = self._align_features(X)
        # TreeExplainer in some versions provides .shap_values; in others, __call__
        try:
            vals = self.explainer.shap_values(X_aligned)
        except Exception:
            vals = self.explainer(X_aligned).values
        return vals

    # --- inference API ---
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("fit first")
        return self.ensemble.predict(self._prep(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("fit first")
        return self.ensemble.predict_proba(self._prep(X))

    def predict_with_uncertainty(self, X):
        if not (self.is_fitted and self.use_all_techniques):
            raise ValueError("Uncertainty estimation not available")
        return self.uncertainty_estimator.predict_with_uncertainty(self._prep(X))


# --- Baseline model factory (kept for compatibility with your imports) ---
def get_models_smalldata():
    return {
        'Small-Data Pipeline (Full)': SmallDataCreditPipeline(use_all_techniques=True),
        'Small-Data Pipeline (Basic)': SmallDataCreditPipeline(use_all_techniques=False),
        'XGBoost Baseline': XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                          random_state=SEED, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    }
