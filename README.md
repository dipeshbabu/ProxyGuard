# Risk Models

Reliable credit risk benchmarking for small and medium scale tabular finance datasets.

This repository is built for finance-focused workshop-style experiments on credit risk modeling, with emphasis on:

- calibration and probability quality
- repeated train/validation/test evaluation
- compact feature selection
- interpretable feature engineering
- optional segmentation
- weak-label sensitivity analysis
- subgroup reliability checks

## Supported datasets

The current benchmark supports:

- `taiwan_default`
  Real default benchmark based on the UCI Default of Credit Card Clients dataset.
- `give_me_some_credit`
  Real default benchmark based on the Give Me Some Credit dataset.
- `german_credit`
  Small-data benchmark with explicit weak-label construction and sensitivity sweeps.
- `australian_credit`
  Optional small credit approval benchmark.

Dataset loaders can download raw files on first use if they are not already present under `data/`.

## Repository structure

```text
risk-models/
  README.md
  requirements.txt
  .gitignore
  main.py
  data/
  outputs/
  scripts/
    smoke_test.py
  risk_models/
    __init__.py
    configs.py
    dataset.py
    model.py
    eval.py
    cv_runner.py
    reporting.py
    diagnostics.py
    datasets/
  tests/
    test_smoke.py
```

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main experiment modes

Quick debug run:

```bash
python main.py --mode debug --dataset german_credit --output-root outputs
```

Main benchmark:

```bash
python main.py --mode benchmark --dataset all --repeats 20 --output-root outputs --calibration-method temperature
```

Ablation suite:

```bash
python main.py --mode ablation --dataset german_credit --repeats 20 --output-root outputs --calibration-method temperature
```

Weak-label sensitivity:

```bash
python main.py --mode weak_label --dataset german_credit --repeats 20 --output-root outputs --calibration-method temperature
```

Calibration comparison:

```bash
python main.py --mode benchmark --dataset all --repeats 20 --output-root outputs_no_cal --calibration-method none
```

## Output files

Each run writes outputs under `outputs/`:

- `outputs/benchmark/<dataset>/`
- `outputs/ablations/<dataset>/`
- `outputs/weak_label/<variant>/<dataset>/`
- `outputs/debug/<dataset>/`

Common output files include:

- `split_metrics.csv`
- `aggregate_metrics.csv`
- `aggregate_metrics.tex`
- `subgroup_metrics.csv`
- `feature_stability.csv`

Depending on flags, runs may also save reliability diagrams. SHAP explainer hooks exist in the model layer, but runner-level SHAP artifact export is not enabled yet.

## Smoke test

Run a fast pipeline smoke test:

```bash
python scripts/smoke_test.py --dataset all
```

## Recommended paper workflow

For a finance-focused workshop paper, run experiments in this order:

1. Debug run on `german_credit`
2. Full benchmark on `german_credit`
3. Ablation suite on `german_credit`
4. Weak-label sensitivity on `german_credit`
5. Full benchmark on `taiwan_default`
6. Full benchmark on `give_me_some_credit`
7. Optional benchmark on `australian_credit`

## Notes

- `german_credit` is a weak-label benchmark and should not be the only dataset in the final paper.
- `taiwan_default` and `give_me_some_credit` should be the main real-label finance benchmarks.
- Calibration is controlled at run time through `--calibration-method`.
