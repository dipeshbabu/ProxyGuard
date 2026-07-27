from __future__ import annotations

import numpy as np

from proxyguard.core import empirical_bernstein_badness_pvalue
from scripts.proxyguard.run_proxyguard_out_of_mechanism_study import (
    _empirical_bernstein_badness_pvalues,
)


def test_vectorized_empirical_bernstein_inversion_matches_audit_code() -> None:
    rng = np.random.default_rng(23)
    draws = np.clip(rng.normal(-0.08, 0.20, size=(3, 2, 250)), -1.0, 1.0)
    vectorized = _empirical_bernstein_badness_pvalues(draws, tolerance=0.01)
    expected = np.asarray(
        [
            [
                empirical_bernstein_badness_pvalue(
                    draws[candidate, requirement],
                    tolerance=0.01,
                    lower=-1.0,
                    upper=1.0,
                )
                for requirement in range(draws.shape[1])
            ]
            for candidate in range(draws.shape[0])
        ]
    )
    np.testing.assert_allclose(vectorized, expected, atol=1e-12, rtol=1e-12)
