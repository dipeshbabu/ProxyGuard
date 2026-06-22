# Reliability First Evaluation of Tabular Foundation Models for Credit Risk

## Overview

This repository supports a reliability-first study of tabular models for credit risk. The paper asks a practical question: when credit-risk models are judged by calibrated probabilities and decision cost instead of ranking metrics alone, do newer tabular foundation models such as TabPFN change the baseline story?

The problem is important because credit-risk systems often use predicted probabilities for thresholding, portfolio monitoring, and downstream decisions. A model can have acceptable AUC while still producing probabilities that are poorly calibrated for decision making. This benchmark therefore evaluates both discrimination and reliability across four public credit-related datasets.

The current workshop-style finding is that XGBoost and compact XGBoost remain the strongest overall baselines, especially on larger datasets, while TabPFN is competitive on smaller datasets and should be included as a first-class baseline in modern tabular evaluations. The spotlight-oriented revision adds a reliability-constrained ensemble (RCE) that learns mixture weights across classical tabular models using an inner validation objective with log loss, Brier score, ECE, class-balance calibration, and decision-cost penalties.

## What Is In Scope

The benchmark compares:

- Logistic regression
- XGBoost
- LightGBM
- CatBoost
- Compact XGBoost
- TabPFN
- TabICL
- Reliability-constrained ensemble / RCE

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

The revision also supports a small synthetic-noise robustness probe on real-label datasets. The Australian Credit probe used for the revised robustness table can be regenerated with:

```bash
python main.py --mode weak_label --dataset australian_credit --repeats 20 --output-root outputs --calibration-method temperature --no-reliability
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

## Run The Spotlight-Oriented Model Set

This includes logistic regression, XGBoost, LightGBM, CatBoost, compact XGBoost, TabPFN, TabICL, and the reliability-constrained ensemble. Use this mode for the method-paper revision.

```bash
python scripts/run_fmsd_experiments.py --datasets all --repeats 20 --output-root outputs/spotlight_final --model-set spotlight
```

For a quick contract run of only the new method on one dataset, lower `--repeats` and use a temporary output root.

## Build Local Paper Assets

This writes the final local tables and figures under `paper_assets/fmsd_tabpfn_mixed/`.

```bash
python scripts/build_paper_assets.py --output-root outputs/fmsd_tabpfn --asset-root paper_assets/fmsd_tabpfn_mixed --include-tabpfn
```

If a timing probe exists under `outputs/efficiency_probe`, the asset builder uses it for the practical cost table while keeping the main results tied to the full benchmark. To set this explicitly:

```bash
python scripts/build_paper_assets.py --output-root outputs/fmsd_tabpfn --asset-root paper_assets/fmsd_tabpfn_mixed --efficiency-root outputs/efficiency_probe --include-tabpfn
```

For the spotlight-oriented revision:

```bash
python scripts/build_paper_assets.py --output-root outputs/spotlight_final --asset-root paper_assets/spotlight_final --include-tabpfn
python scripts/build_reviewer_readiness_audit.py --output-root outputs/spotlight_final --asset-root paper_assets/spotlight_readiness
```

The reviewer-readiness audit intentionally reports gaps, including dataset breadth, missing split counts, missing method ablations, or missing subgroup artifacts.

The final draft uses:

```text
paper_assets/fmsd_tabpfn_mixed/main_results_table.tex
paper_assets/fmsd_tabpfn_mixed/main_results_with_ci_table.tex
paper_assets/fmsd_tabpfn_mixed/paired_win_counts_table.tex
paper_assets/fmsd_tabpfn_mixed/efficiency_table.tex
paper_assets/fmsd_tabpfn_mixed/auc_ece_tradeoff.png
paper_assets/fmsd_tabpfn_mixed/weak_label_sensitivity.png
paper_assets/fmsd_tabpfn_mixed/weak_label_sensitivity_australian.png
paper_assets/fmsd_tabpfn_mixed/weak_label_two_dataset_table.tex
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
- ECE sensitivity across 10, 15, 20, and adaptive bins
- validation-selected F1 and asymmetric decision-cost thresholds
- repeated split aggregation with confidence intervals
- subgroup reliability artifacts
- reliability-constrained ensemble with log-loss, Brier, ECE, calibration-balance, and cost penalties
- German Credit weak-label perturbations

See:

- `risk_models/eval.py`
- `risk_models/cv_runner.py`
- `risk_models/model.py`
