# Frozen study registries

This directory contains the registered designs used by the ProxyGuard studies
and audits. Each immutable JSON declaration is accompanied by a `.sha256`
digest when the workflow requires integrity verification.

Treat a frozen registry as an experimental input: do not edit it in place.
Create a new version and digest for a prospective amendment, then record the
reason for the amendment before opening the corresponding target data.

The smooth target-concentration study retains both exploratory pilots. The
first found no eligible separation at reliability 0.95; the second registered
the revised 0.99 question before the confirmatory seed was run.

The 20 Newsgroups audit retains its original 2,000-target registry and a
pre-audit feasibility amendment to 4,000 i.i.d. target draws. The amendment
changed no outcome-dependent quantity and was frozen before the test subset
was loaded.
