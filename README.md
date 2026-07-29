# ProxyGuard

ProxyGuard tests whether a proxy dataset supports prespecified downstream
claims without treating a favorable point estimate as validation. It handles
two questions separately:

1. Does one realized proxy release satisfy every registered requirement?
2. Does a frozen randomized mechanism produce valid releases often enough?

The release audit uses bounded target losses, an intersection-union test, and
Holm correction across screened candidates. The mechanism audit applies a
conservative binomial tail to statistically recognized releases and corrects
across mechanisms. Results are `Validated`, `Violation detected`, or
`Unresolved`.

The guarantee is conditional on the registered source, target population,
learning procedures, losses, limits, candidates, and release mechanisms. It
does not certify privacy or approve a data release.

## Repository layout

```text
proxyguard/
  core.py                 # release- and mechanism-level risk control
  shared_target.py        # direct reliability bounds for one shared target
  attacks.py              # fixed empirical attack-suite utilities
scripts/
  proxyguard/             # ProxyGuard studies and artifact builders
  *.py                    # legacy benchmark drivers
registries/               # frozen study declarations and SHA-256 files
risk_models/              # tabular learning and evaluation support
tests/
  proxyguard/             # method and experiment-driver tests
  test_experiment_contracts.py
```

Generated outputs, local datasets, and manuscript files are intentionally not
version controlled. See `.gitignore`.

## Installation

The recorded experiments use Python 3.13 and the versions in
`requirements.txt`.

```bash
uv python install 3.13
uv venv --python 3.13
uv pip install -r requirements.txt
```

## Public API

```python
from proxyguard import (
    RiskRequirement,
    audit_proxy_candidates,
    audit_proxy_mechanisms,
    plan_conditional_shared_target,
    shared_target_conditional_mean_lower_bound,
    shared_target_conditional_witness_lower_bound,
)
```

`proxyguard.core` also exposes the component confidence bounds, p-values,
Holm adjustment, Clopper-Pearson bounds, and sequential alpha-spending
utilities used by the experiments.

## Reproduce the controlled studies

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
python scripts/proxyguard/audit_proxyguard_target_lineage.py
```

By default, mechanism audits use Holm-certified release counts. The direct
shared-target function above provides a mechanism-only lower bound without
certifying named releases. The planning helper reports necessary best-case
target and release counts before an audit is opened. If both named and direct
bounds are inspected, divide the declared error budget between them before
target access; taking their uncorrected maximum is not valid.

The Simes partial-conjunction mode is an independent-batch baseline. It must
not be enabled merely because releases share one target. When the registered
design supplies independent audit batches or a justified PRDS argument, pass
both flags:

```bash
python scripts/proxyguard/run_proxyguard_mechanism_audit.py ... \
  --mechanism-count-mode simes --collective-dependence-verified
```

The MAGIC audit is a nonprivate fidelity control. Its `prepare` command
selects and hashes an unlabeled reserve, fixes the requirements and release
seeds from the remaining development records, and writes a frozen registry.
The `audit` command verifies those hashes before reading the reserve.

AIM requires SmartNoise Synth:

```bash
uv run --with smartnoise-synth \
  python scripts/proxyguard/run_proxyguard_repeated_aim.py
```

TabDDPM requires the authors' runtime and repository:

```bash
python scripts/proxyguard/run_proxyguard_tabddpm_audit.py \
  --tabddpm-repo /path/to/tab-ddpm
```

The empirical attack study is conditional on its fixed attack suite:

```bash
python scripts/proxyguard/run_proxyguard_privacy_attacks.py
```

## Validation

```bash
uv run ruff check proxyguard scripts/proxyguard tests
uv run pytest -q
```

The older tabular reliability benchmark remains available in `risk_models/`
and the legacy scripts. Its short usage guide is in
[`docs/legacy-benchmark.md`](docs/legacy-benchmark.md).
