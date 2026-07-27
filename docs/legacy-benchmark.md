# Legacy tabular reliability benchmark

The repository began as a repeated-split benchmark for discrimination,
calibration, decision cost, and subgroup reliability on public tabular
datasets. ProxyGuard still uses its dataset loaders and learning pipelines.

Run the classical benchmark:

```bash
python main.py \
  --mode benchmark \
  --dataset all \
  --repeats 20 \
  --output-root outputs \
  --calibration-method temperature
```

Run weak-label sensitivity:

```bash
python main.py \
  --mode weak_label \
  --dataset german_credit \
  --repeats 20 \
  --output-root outputs \
  --calibration-method temperature
```

The implementation is under `risk_models/`. Legacy experiment and artifact
drivers remain directly under `scripts/`. Their generated datasets, tables,
figures, and model artifacts are excluded from version control.
