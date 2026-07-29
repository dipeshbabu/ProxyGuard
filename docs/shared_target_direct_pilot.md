# Shared-target direct reliability pilot

This note records the first exploratory attempt to replace release-by-release
certification with a direct reliability bound from one shared target sample.
The block and tensor constructions below were not used as confirmatory
evidence. A later conditional-score construction addressed the main power
problem and was evaluated in a separate frozen confirmation.

## Statistical constructions

The pilot implements two finite-sample lower bounds.

The block-witness bound partitions one sealed target into disjoint blocks. A
release receives a score in \([0,1]\) only when every block-average loss falls
below a registered cutoff. For any invalid release, a bounded-loss
Bernoulli-KL Chernoff bound limits the expected score by \(\kappa\). The
complete release-by-block average is a two-sample U-statistic. A matching
argument reduces its concentration analysis to
\(\min(R,B)\) independent bounded observations, where \(R\) is the number of
releases and \(B\) is the number of target blocks.

The polynomial bound constructs a tensor-Bernstein polynomial that is at most
one everywhere and at most zero whenever any registered requirement is
invalid. Its expectation is therefore no larger than mechanism reliability.
Distinct target positions provide unbiased mixed-moment estimates while all
releases use the same target records. The coefficient tensor is obtained from
a sparse linear program before audit outcomes are read.

Neither construction assumes independence or PRDS among release-level
\(p\)-values.

## Exploratory design

The registry is
`registries/proxyguard_shared_target_pilot.json`. It uses three mechanisms,
50 releases per mechanism, three bounded Bernoulli requirements, and one
shared target of 5,000 records. Independent-batch partial-conjunction
baselines receive 100 records per release, so every method has the same total
target-label budget. The pilot has only 20 repetitions and cannot estimate a
0.05 error rate precisely.

## Result

The block witness was calibrated in this small run but did not improve on
named-release Holm certification. At true reliability 0.95, its per-mechanism
validation rate was 0.333, compared with 0.767 for Holm. Replacing the original
Hoeffding ceiling with the sharper Bernoulli-KL ceiling improved the witness
substantially, but not enough to close the gap.

The degree-8 tensor polynomial had no validation power in the same setting.
Its certified floor over the registered three-dimensional valid box was
0.279, and its mean reliability lower bound at true reliability 0.95 was
0.197. A fixed-mechanism sensitivity check increased the degree to 12 and the
coefficient range to \([-1,1]\); the resulting lower bound was still only
0.531.

The numerical optimization completed normally. Low-degree polynomials did not
separate the joint valid region sharply enough. Raising the degree makes the
coefficient tensor grow as \((d+1)^J\), and widening the coefficient range
weakens the finite-sample concentration bound.

## Follow-up decision

These exploratory bounds did not support a paper claim. The follow-up method
therefore abandoned the tensor approximation. It conditions on the shared
target, obtains an exact binomial lower bound for the mean release score, and
subtracts a finite-sample allowance for invalid releases that score well by
chance. The pilot registry is
`registries/proxyguard_conditional_shared_target_pilot.json`.

The selected setting was rerun for 1,000 repetitions under the frozen registry
`registries/proxyguard_conditional_shared_target_confirmatory.json`, whose
SHA-256 is
`5496f46f21b203895d7b0ae46d58523bf7e3c8ed0bd9e54a221a78168539b087`.
At the reliability boundary 0.80, observed false validation was 0.0% for the
conditional method and 2.2% for named-release Holm. At reliability 0.95,
power was 89.2% and 80.1%, respectively. At reliability 0.90, Holm was
stronger, 79.9% versus 69.0%. The confirmatory result therefore supports a
selective shared-target gain, not uniform dominance.
