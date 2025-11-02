# configs.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
SEED = 3407
DATA_PATH = r"data/german_credit.csv"
XGB_KW = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
              subsample=0.8, eval_metric='logloss', random_state=SEED)
METRICS_PNG = "metrics_comparison.png"
RESULTS_CSV = "table1_results.csv"
RESULTS_TEX = "table1_results.tex"
BEST_MODEL_PKL = "best_risk_model.pkl"
