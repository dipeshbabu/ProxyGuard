# Reliability First Evaluation of Tabular Foundation Models for Credit Risk

## Overview

This repository supports a short workshop paper on reliability-first evaluation of tabular foundation models for credit risk. The paper asks a practical question: when credit-risk models are judged by calibrated probabilities instead of ranking metrics alone, do newer tabular foundation models such as TabPFN change the baseline story?

The problem is important because credit-risk systems often use predicted probabilities for thresholding, portfolio monitoring, and downstream decisions. A model can have acceptable AUC while still producing probabilities that are poorly calibrated for decision making. This benchmark therefore evaluates both discrimination and reliability across four public credit-related datasets.

The paper's main finding is that XGBoost and compact XGBoost remain the strongest overall baselines, especially on larger datasets, while TabPFN is competitive on smaller datasets and should be included as a first-class baseline in modern tabular evaluations. The work is not proposing a new model; it is making the empirical case that calibration-aware evaluation changes how model quality should be interpreted in credit-risk settings.

## What Is In Scope

The benchmark compares:

- Logistic regression
- XGBoost
- Compact XGBoost
- TabPFN

on four credit-related tabular datasets:

- `australian_credit`
- `german_credit`
- `give_me_some_credit` / GMSC
- `taiwan_default`

The final paper reports:

- AUC
- AUPRC
- Brier score
- expected calibration error
- calibration slope
- weak-label sensitivity on German Credit

## Repository Layout

```text
risk-models/
  README.md
  requirements.txt
  main.py
  risk_models/
  scripts/
    run_fmsd_experiments.py
    build_paper_assets.py
  tests/
    test_experiment_contracts.py
```

## Setup

The experiments use Python 3.13 and the exact package versions pinned in `requirements.txt`.

The commands below use `uv`; activating the virtual environment is optional because project commands are run through `uv run`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv venv --python 3.13
uv pip install -r requirements.txt
```

TabPFN local inference also requires Prior Labs license acceptance and a runtime token:

```bash
export TABPFN_TOKEN="paste_your_token_here"
```

## Run The Classical Benchmark

This regenerates the full classical benchmark used for the main table.

```bash
python main.py --mode benchmark --dataset all --repeats 20 --output-root outputs --calibration-method temperature
```

## Run Weak-Label Sensitivity

This regenerates the German Credit weak-label sensitivity outputs used for the weak-label figure.

```bash
python main.py --mode weak_label --dataset german_credit --repeats 20 --output-root outputs --calibration-method temperature
```

## Run TabPFN Outputs Used In The Final Draft

The final draft uses completed TabPFN rows for Australian, German, and Taiwan. GMSC TabPFN is not included because it did not complete under the CPU budget.

Australian and German use 20 splits:

```bash
python scripts/run_fmsd_experiments.py --datasets australian_credit,german_credit --repeats 20 --output-root outputs/fmsd_tabpfn --model-set tabpfn --skip-ablation --skip-weak-label
```

Taiwan uses 5 splits:

```bash
python scripts/run_fmsd_experiments.py --datasets taiwan_default --repeats 5 --output-root outputs/fmsd_tabpfn --model-set tabpfn --skip-ablation --skip-weak-label
```

## Build Local Paper Assets

This writes the final local tables and figures under `paper_assets/fmsd_tabpfn_mixed/`.

```bash
python scripts/build_paper_assets.py --output-root outputs/fmsd_tabpfn --asset-root paper_assets/fmsd_tabpfn_mixed --include-tabpfn
```

The final draft uses:

```text
paper_assets/fmsd_tabpfn_mixed/main_results_table.tex
paper_assets/fmsd_tabpfn_mixed/auc_ece_tradeoff.png
paper_assets/fmsd_tabpfn_mixed/weak_label_sensitivity.png
paper_assets/fmsd_tabpfn_mixed/calibration_delta_table.csv
```

## Validate Code

```bash
python -m compileall main.py risk_models scripts tests
uv run pytest -q
```

## Method Details Implemented In Code

The implemented evaluation uses:

- stratified train/validation/test splits
- validation-fitted temperature scaling
- AUC, AUPRC, Brier, ECE, log loss, calibration slope/intercept
- repeated split aggregation with confidence intervals
- German Credit weak-label perturbations

See:

- `risk_models/eval.py`
- `risk_models/cv_runner.py`
- `risk_models/model.py`
