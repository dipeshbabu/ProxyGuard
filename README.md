# ProxyGuard: Direct Reliability Inference for Randomized Data Release Mechanisms with Shared Targets

This repository contains the ProxyGuard Python package, registered audit
configurations, executable study drivers, and automated tests.

## Repository layout

```text
proxyguard/                # public statistical library
scripts/proxyguard/        # studies, audits, planners, and asset builders
scripts/proxyguard/tabular/ # tabular data loaders and evaluation helpers
registries/                # frozen study declarations and SHA-256 digests
tests/proxyguard/           # method and experiment-driver tests
docs/                       # study and workflow notes
outputs/                    # generated experiment results (ignored)
```

See [`scripts/README.md`](scripts/README.md),
[`registries/README.md`](registries/README.md), and
[`tests/README.md`](tests/README.md) for the conventions within those
directories. Generated outputs and local datasets are excluded from version
control; see [`.gitignore`](.gitignore).

## Installation

The library supports Python 3.11 or newer. The recorded experiments use Python
3.13 and the exact versions in `requirements.txt`.

```bash
uv python install 3.13
uv venv --python 3.13
uv pip install -e .
uv pip install -r requirements.txt
```

## Public API

```python
from proxyguard import (
    RiskRequirement,
    audit_proxy_candidates,
    audit_proxy_mechanisms,
    plan_conditional_shared_target,
    recommend_cost_normalized_audit,
    shared_target_block_witness_lower_bound,
    shared_target_conditional_mean_lower_bound,
    shared_target_conditional_witness_lower_bound,
)
```

`proxyguard.core` also exposes the component confidence bounds, p-values,
Holm adjustment, Clopper-Pearson bounds, and sequential alpha-spending
utilities used by the experiments.

## Run the controlled studies

```bash
python scripts/proxyguard/run_proxyguard_calibration_study.py --repetitions 5000
python scripts/proxyguard/run_proxyguard_target_reuse_study.py
python scripts/proxyguard/run_proxyguard_out_of_mechanism_study.py
python scripts/proxyguard/run_proxyguard_mechanism_study.py
python scripts/proxyguard/run_proxyguard_mechanism_revision_study.py
python -m scripts.proxyguard.run_proxyguard_collective_extension
python -m scripts.proxyguard.run_proxyguard_conditional_shared_target \
  --registry registries/proxyguard_conditional_shared_target_confirmatory.json \
  --output-root outputs/proxyguard_conditional_shared_target_confirmatory
python -m scripts.proxyguard.run_proxyguard_direct_multirequirement \
  --registry registries/proxyguard_direct_multirequirement_moderate_confirmatory.json \
  --output-root outputs/proxyguard_direct_multirequirement_moderate_confirmatory
python -m scripts.proxyguard.run_proxyguard_false_pass_diagnostic \
  --registry registries/proxyguard_false_pass_diagnostic_confirmatory.json \
  --output-root outputs/proxyguard_false_pass_diagnostic_confirmatory
python -m scripts.proxyguard.build_direct_multirequirement_assets
python -m scripts.proxyguard.run_proxyguard_stratified_subgroup_study
```

## Run the real-data audits

The drivers read their declarations from `registries/` and write generated
artifacts under `outputs/`.

```bash
python scripts/proxyguard/run_proxyguard_shift_audits.py
python scripts/proxyguard/run_proxyguard_bootstrap_mechanism.py
python scripts/proxyguard/run_proxyguard_magic_sealed_mechanism.py prepare
python scripts/proxyguard/run_proxyguard_magic_sealed_mechanism.py audit
python -m scripts.proxyguard.run_proxyguard_spambase_aim_mechanism prepare
python -m scripts.proxyguard.run_proxyguard_spambase_aim_mechanism prepare-amendment
uv run --with smartnoise-synth \
  python -m scripts.proxyguard.run_proxyguard_spambase_aim_mechanism audit \
  --registry registries/proxyguard_spambase_aim_audit_v2.json --jobs 4
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct prepare
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct amend-domain
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct amend-mechanism
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct pilot \
  --partition-registry registries/proxyguard_secondary_mushroom_partition_v3.json
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct freeze \
  --partition-registry registries/proxyguard_secondary_mushroom_partition_v3.json
python -m scripts.proxyguard.run_proxyguard_secondary_mushroom_direct audit
python -m scripts.proxyguard.run_proxyguard_neural_direct prepare \
  --partition-registry registries/proxyguard_covertype_neural_partition.json
python -m scripts.proxyguard.run_proxyguard_neural_direct pilot \
  --partition-registry registries/proxyguard_covertype_neural_partition.json \
  --output-root outputs/proxyguard_covertype_neural_direct/pilot
python -m scripts.proxyguard.run_proxyguard_neural_direct freeze \
  --partition-registry registries/proxyguard_covertype_neural_partition.json \
  --pilot-root outputs/proxyguard_covertype_neural_direct/pilot \
  --audit-registry registries/proxyguard_covertype_neural_audit.json
python -m scripts.proxyguard.run_proxyguard_neural_direct generate \
  --audit-registry registries/proxyguard_covertype_neural_audit.json \
  --output-root outputs/proxyguard_covertype_neural_direct/generated
python -m scripts.proxyguard.run_proxyguard_neural_direct audit \
  --audit-registry registries/proxyguard_covertype_neural_audit.json \
  --generation-root outputs/proxyguard_covertype_neural_direct/generated \
  --output-root outputs/proxyguard_covertype_neural_direct/audit_stratified \
  --sampling-correction registries/proxyguard_covertype_neural_sampling_correction.json
python -m scripts.proxyguard.run_proxyguard_cost_planning
python -m scripts.proxyguard.run_proxyguard_cost_mode_phase
python -m scripts.proxyguard.run_proxyguard_cost_planning_sensitivity
uv run --with ctgan python -m scripts.proxyguard.run_proxyguard_rice_tvae \
  --audit-registry registries/proxyguard_rice_tvae_audit.json \
  --generation-root outputs/proxyguard_rice_tvae/releases_v3 \
  --audit-root outputs/proxyguard_rice_tvae/audit audit
python scripts/proxyguard/audit_proxyguard_target_lineage.py
```

## Optional runtimes

Install SmartNoise Synth when running AIM drivers:

```bash
uv run --with smartnoise-synth \
  python scripts/proxyguard/run_proxyguard_repeated_aim.py
```

Provide a TabDDPM checkout when running its audit driver:

```bash
python scripts/proxyguard/run_proxyguard_tabddpm_audit.py \
  --tabddpm-repo /path/to/tab-ddpm
```

Run the empirical attack driver with:

```bash
python scripts/proxyguard/run_proxyguard_privacy_attacks.py
```

## Validation

```bash
uv run ruff check proxyguard scripts tests
uv run pytest -q
```
