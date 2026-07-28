from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import beta, binom


@dataclass(frozen=True)
class RiskRequirement:
    """A bounded mean requirement declared before audit evaluation.

    ``estimand`` records what the supplied observations mean. The core audit
    accepts any bounded per-record quantity, including a proxy-minus-source
    regret or the proxy loss itself. ``lower`` and ``upper`` bound that
    supplied quantity. For losses in [0, 1], relative regrets lie in [-1, 1]
    and absolute risks lie in [0, 1].
    """

    name: str
    tolerance: float
    lower: float = -1.0
    upper: float = 1.0
    estimand: str = "relative_regret"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Requirement names must be non-empty.")
        if not np.isfinite(self.tolerance):
            raise ValueError(f"Tolerance for {self.name!r} must be finite.")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError(f"Bounds for {self.name!r} must be finite.")
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound must be smaller than upper bound for {self.name!r}.")
        if not self.estimand.strip():
            raise ValueError(f"Estimand for {self.name!r} must be non-empty.")


@dataclass(frozen=True)
class ProxyAuditResult:
    candidate_summary: pd.DataFrame
    requirement_detail: pd.DataFrame


@dataclass(frozen=True)
class MechanismAuditResult:
    release_audit: ProxyAuditResult
    mechanism_summary: pd.DataFrame


@dataclass(frozen=True)
class SequentialAuditResult:
    round_summary: pd.DataFrame
    requirement_detail: pd.DataFrame


def _bounded_values(values: Sequence[float], lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("At least one audit observation is required.")
    if not np.isfinite(array).all():
        raise ValueError("Audit observations must be finite.")
    tolerance = 1e-12
    if np.any(array < lower - tolerance) or np.any(array > upper + tolerance):
        raise ValueError(f"Audit observations must lie in [{lower}, {upper}].")
    return array


def hoeffding_badness_pvalue(
    values: Sequence[float],
    tolerance: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Test H0: E[value] >= tolerance using a one-sided Hoeffding bound."""

    array = _bounded_values(values, lower, upper)
    gap = float(tolerance - array.mean())
    if gap <= 0.0:
        return 1.0
    value_range = upper - lower
    return float(min(1.0, exp(-2.0 * array.size * gap * gap / (value_range * value_range))))


def hoeffding_violation_pvalue(
    values: Sequence[float],
    tolerance: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Test H0: E[value] <= tolerance using a one-sided Hoeffding bound."""

    array = _bounded_values(values, lower, upper)
    gap = float(array.mean() - tolerance)
    if gap <= 0.0:
        return 1.0
    value_range = upper - lower
    return float(min(1.0, exp(-2.0 * array.size * gap * gap / (value_range * value_range))))


def hoeffding_upper_bound(
    values: Sequence[float],
    error_rate: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    array = _bounded_values(values, lower, upper)
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    radius = (upper - lower) * sqrt(log(1.0 / error_rate) / (2.0 * array.size))
    return float(min(upper, array.mean() + radius))


def hoeffding_lower_bound(
    values: Sequence[float],
    error_rate: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    array = _bounded_values(values, lower, upper)
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    radius = (upper - lower) * sqrt(log(1.0 / error_rate) / (2.0 * array.size))
    return float(max(lower, array.mean() - radius))


def empirical_bernstein_upper_bound(
    values: Sequence[float],
    error_rate: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Maurer--Pontil empirical Bernstein upper confidence bound."""

    array = _bounded_values(values, lower, upper)
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    if array.size < 2:
        return hoeffding_upper_bound(array, error_rate, lower, upper)
    log_term = log(2.0 / error_rate)
    variance = float(array.var(ddof=1))
    radius = sqrt(2.0 * variance * log_term / array.size)
    radius += 7.0 * (upper - lower) * log_term / (3.0 * (array.size - 1))
    return float(min(upper, array.mean() + radius))


def empirical_bernstein_lower_bound(
    values: Sequence[float],
    error_rate: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Maurer--Pontil empirical Bernstein lower confidence bound."""

    array = _bounded_values(values, lower, upper)
    if array.size < 2:
        return hoeffding_lower_bound(array, error_rate, lower, upper)
    reflected = lower + upper - array
    reflected_upper = empirical_bernstein_upper_bound(
        reflected,
        error_rate,
        lower,
        upper,
    )
    return float(max(lower, lower + upper - reflected_upper))


def _invert_confidence_bound(
    predicate,
    iterations: int = 80,
) -> float:
    almost_one = 1.0 - np.finfo(float).eps
    if not predicate(almost_one):
        return 1.0
    low = np.finfo(float).tiny
    high = almost_one
    for _ in range(iterations):
        midpoint = (low + high) / 2.0
        if predicate(midpoint):
            high = midpoint
        else:
            low = midpoint
    return float(high)


def empirical_bernstein_badness_pvalue(
    values: Sequence[float],
    tolerance: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Invert an empirical Bernstein UCB for H0: E[value] >= tolerance."""

    array = _bounded_values(values, lower, upper)
    if float(array.mean()) >= tolerance:
        return 1.0
    return _invert_confidence_bound(
        lambda error_rate: empirical_bernstein_upper_bound(
            array,
            error_rate,
            lower,
            upper,
        )
        < tolerance
    )


def empirical_bernstein_violation_pvalue(
    values: Sequence[float],
    tolerance: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Invert an empirical Bernstein LCB for H0: E[value] <= tolerance."""

    array = _bounded_values(values, lower, upper)
    if float(array.mean()) <= tolerance:
        return 1.0
    return _invert_confidence_bound(
        lambda error_rate: empirical_bernstein_lower_bound(
            array,
            error_rate,
            lower,
            upper,
        )
        > tolerance
    )


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values while preserving input keys."""

    if not pvalues:
        return {}
    checked: list[tuple[str, float]] = []
    for name, raw_value in pvalues.items():
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {name!r}: {raw_value!r}")
        checked.append((name, value))

    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, (total - rank) * value)
        adjusted[name] = float(min(1.0, running_max))
    return {name: adjusted[name] for name in pvalues}


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    error_rate: float,
) -> float:
    """One-sided exact lower confidence bound for a binomial probability."""

    _validate_binomial_inputs(successes, trials, error_rate)
    if successes == 0:
        return 0.0
    return float(beta.ppf(error_rate, successes, trials - successes + 1))


def clopper_pearson_upper_bound(
    successes: int,
    trials: int,
    error_rate: float,
) -> float:
    """One-sided exact upper confidence bound for a binomial probability."""

    _validate_binomial_inputs(successes, trials, error_rate)
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - error_rate, successes + 1, trials - successes))


def _validate_binomial_inputs(
    successes: int,
    trials: int,
    error_rate: float,
) -> None:
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be integers.")
    if int(successes) != successes or int(trials) != trials:
        raise ValueError("successes and trials must be integers.")
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")


def mechanism_validity_pvalue(
    validated_releases: int,
    releases: int,
    minimum_reliability: float,
) -> float:
    """Test H0: mechanism reliability <= minimum_reliability.

    The count may be a conservative lower bound on the number of truly valid
    releases. The upper-tail binomial p-value remains conservative in that
    case.
    """

    _validate_binomial_inputs(validated_releases, releases, 0.5)
    if not 0.0 < minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie strictly between zero and one.")
    return float(binom.sf(validated_releases - 1, releases, minimum_reliability))


def mechanism_violation_pvalue(
    detected_violations: int,
    releases: int,
    minimum_reliability: float,
) -> float:
    """Test H0: mechanism reliability >= minimum_reliability.

    A detected violation is a conservative lower bound on the number of bad
    releases. Therefore ``releases - detected_violations`` is a conservative
    upper bound on the number of good releases.
    """

    _validate_binomial_inputs(detected_violations, releases, 0.5)
    if not 0.0 < minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie strictly between zero and one.")
    maximum_good = releases - detected_violations
    return float(binom.cdf(maximum_good, releases, minimum_reliability))


def mechanism_release_lower_count(
    candidate_frame: pd.DataFrame,
    mode: str,
    release_alpha: float,
) -> int:
    """Convert release outcomes into a conservative lower bound on valid releases.

    Parameters
    ----------
    mode:
      - ``"holm"``: count releases that pass individual Holm validation.
      - ``"simes"``: compute a collective lower bound on the number of valid
        releases using Simes-style partial-conjunction scores.

    The Simes mode can be less conservative and is intended to provide stronger
    mechanism-only evidence under the standard independence/PRDS regime for
    p-values.
    """

    if not 0.0 < release_alpha <= 1.0:
        raise ValueError("release_alpha must be in (0, 1].")
    frame = candidate_frame.copy()
    if frame.empty:
        return 0
    if "CandidatePValue" not in frame:
        raise ValueError("candidate_frame must include CandidatePValue.")

    release_pvalues = frame["CandidatePValue"].to_numpy(dtype=float)
    releases = int(release_pvalues.size)
    if mode == "holm":
        return int(frame["Validated"].sum())
    if mode != "simes":
        raise ValueError(
            "mechanism_count_mode must be 'holm' or 'simes'."
        )

    if releases == 0:
        return 0
    ranked = np.sort(release_pvalues)
    index = np.arange(1, releases + 1, dtype=float)
    scaled = np.minimum(1.0, (releases / index) * ranked)
    scaled = np.maximum.accumulate(scaled)
    return int(np.sum(scaled <= release_alpha))


def audit_proxy_mechanisms(
    candidate_regrets: Mapping[str, Mapping[str, Sequence[float]]],
    release_to_mechanism: Mapping[str, str],
    requirements: Sequence[RiskRequirement],
    minimum_reliability: float = 0.8,
    total_alpha: float = 0.05,
    release_error_share: float = 0.5,
    violation_total_alpha: float | None = None,
    violation_release_error_share: float | None = None,
    mechanism_count_mode: str = "holm",
    bound_method: str = "empirical_bernstein",
) -> MechanismAuditResult:
    """Audit realized releases and the mechanisms that generated them.

    The total false-validation budget is split between two layers. The inner
    release audit controls the chance of validating any invalid realized
    release. The outer binomial upper-tail tests use conservatively
    recognized releases to ask whether each mechanism produces a valid
    release with probability greater than
    ``minimum_reliability``. Holm adjustment covers the registered mechanism
    configurations. The reverse decision, detecting an unreliable mechanism,
    uses a separate error budget. By default it has the same numerical split
    as validation, but the two guarantees remain direction-specific.

    Releases within a mechanism must be independent draws from that
    mechanism and must be generated without target-audit feedback. They may
    share target records because the inner release audit already controls
    arbitrary dependence across its p-values.

    Parameters
    ----------
    mechanism_count_mode:
      - ``"holm"`` (default): count only individually validated releases
        before applying the mechanism binomial test (guaranteed under arbitrary
        dependence from shared targets).
      - ``"simes"``: use a collective lower-bound count on the number of
        valid releases, which is typically less conservative under
        independence/PRDS assumptions on release p-values.
    """

    if not 0.0 < total_alpha < 1.0:
        raise ValueError("total_alpha must lie strictly between zero and one.")
    if not 0.0 < release_error_share < 1.0:
        raise ValueError("release_error_share must lie strictly between zero and one.")
    if violation_total_alpha is None:
        violation_total_alpha = total_alpha
    if not 0.0 < violation_total_alpha < 1.0:
        raise ValueError("violation_total_alpha must lie strictly between zero and one.")
    if violation_release_error_share is None:
        violation_release_error_share = release_error_share
    if not 0.0 < violation_release_error_share < 1.0:
        raise ValueError(
            "violation_release_error_share must lie strictly between zero and one."
        )
    if not 0.0 < minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie strictly between zero and one.")
    if set(candidate_regrets) != set(release_to_mechanism):
        missing = sorted(set(candidate_regrets) - set(release_to_mechanism))
        extra = sorted(set(release_to_mechanism) - set(candidate_regrets))
        raise ValueError(
            "release_to_mechanism must name every candidate exactly once; "
            f"missing={missing}, extra={extra}."
        )
    if any(not str(mechanism).strip() for mechanism in release_to_mechanism.values()):
        raise ValueError("Mechanism names must be non-empty.")

    release_alpha = total_alpha * release_error_share
    mechanism_alpha = total_alpha - release_alpha
    violation_release_alpha = (
        violation_total_alpha * violation_release_error_share
    )
    violation_mechanism_alpha = (
        violation_total_alpha - violation_release_alpha
    )
    release_audit = audit_proxy_candidates(
        candidate_regrets,
        requirements=requirements,
        alpha=release_alpha,
        violation_alpha=violation_release_alpha,
        bound_method=bound_method,
    )
    release_summary = release_audit.candidate_summary.copy()
    release_summary["Mechanism"] = release_summary["Candidate"].map(release_to_mechanism)

    grouped = list(release_summary.groupby("Mechanism", sort=True))
    mechanism_count = len(grouped)
    lower_interval_error = mechanism_alpha / mechanism_count
    upper_interval_error = violation_mechanism_alpha / mechanism_count
    validity_pvalues: dict[str, float] = {}
    violation_pvalues: dict[str, float] = {}
    counts: dict[str, tuple[int, int, int, int]] = {}
    for mechanism, frame in grouped:
        releases = int(len(frame))
        validated = mechanism_release_lower_count(
            frame,
            mode=mechanism_count_mode,
            release_alpha=release_alpha,
        )
        individual_validated = int(frame["Validated"].sum())
        violations = int(frame["ViolationDetected"].sum())
        # Store both collective and individually validated counts; the
        # mechanism-level decision can use either form depending on the
        # selected reporting mode.
        counts[str(mechanism)] = (releases, validated, violations, individual_validated)
        validity_pvalues[str(mechanism)] = mechanism_validity_pvalue(
            validated,
            releases,
            minimum_reliability,
        )
        violation_pvalues[str(mechanism)] = mechanism_violation_pvalue(
            violations,
            releases,
            minimum_reliability,
        )

    validity_adjusted = holm_adjust(validity_pvalues)
    violation_adjusted = holm_adjust(violation_pvalues)
    rows: list[dict[str, float | int | str | bool]] = []
    for mechanism in sorted(counts):
        releases, validated, violations, individual_validated = counts[mechanism]
        validity_rejected = validity_adjusted[mechanism] <= mechanism_alpha
        violation_rejected = (
            violation_adjusted[mechanism] <= violation_mechanism_alpha
        )
        if validity_rejected:
            status = "Mechanism validated"
        elif violation_rejected:
            status = "Reliability violation detected"
        else:
            status = "Unresolved"
        rows.append(
            {
                "Mechanism": mechanism,
                "Releases": releases,
                "ValidatedReleases": validated,
                "IndividuallyValidatedReleases": individual_validated,
                "DetectedReleaseViolations": violations,
                "MinimumReliability": minimum_reliability,
                "ReliabilityLCB": clopper_pearson_lower_bound(
                    validated,
                    releases,
                    lower_interval_error,
                ),
                "ReliabilityUCB": clopper_pearson_upper_bound(
                    releases - violations,
                    releases,
                    upper_interval_error,
                ),
                "LowerBoundSimultaneousCoverage": 1.0 - total_alpha,
                "UpperBoundSimultaneousCoverage": 1.0 - violation_total_alpha,
                "DisplayedPairJointCoverageLowerBound": max(
                    0.0,
                    1.0 - total_alpha - violation_total_alpha,
                ),
                "ValidityPValue": validity_pvalues[mechanism],
                "ValidityHolmAdjustedPValue": validity_adjusted[mechanism],
                "ViolationPValue": violation_pvalues[mechanism],
                "ViolationHolmAdjustedPValue": violation_adjusted[mechanism],
                "Validated": validity_rejected,
                "ViolationDetected": violation_rejected,
                "Status": status,
                "ReleaseErrorRate": release_alpha,
                "MechanismErrorRate": mechanism_alpha,
                "TotalErrorRate": total_alpha,
                "ViolationReleaseErrorRate": violation_release_alpha,
                "ViolationMechanismErrorRate": violation_mechanism_alpha,
                "ViolationTotalErrorRate": violation_total_alpha,
                "MechanismCountMode": mechanism_count_mode,
            }
        )

    return MechanismAuditResult(
        release_audit=ProxyAuditResult(
            candidate_summary=release_summary,
            requirement_detail=release_audit.requirement_detail,
        ),
        mechanism_summary=pd.DataFrame(rows),
    )


def quadratic_alpha_schedule(total_alpha: float, rounds: int) -> np.ndarray:
    """Return the first terms of a summable alpha-spending schedule."""

    if not 0.0 < total_alpha < 1.0:
        raise ValueError("total_alpha must lie strictly between zero and one.")
    if isinstance(rounds, bool) or int(rounds) != rounds or rounds <= 0:
        raise ValueError("rounds must be a positive integer.")
    indices = np.arange(1, int(rounds) + 1, dtype=float)
    return 6.0 * total_alpha / (np.pi**2 * indices**2)


def audit_adaptive_candidate_stream(
    candidate_regrets: Mapping[str, Mapping[str, Sequence[float]]],
    requirements: Sequence[RiskRequirement],
    total_alpha: float = 0.05,
    violation_total_alpha: float | None = None,
    bound_method: str = "empirical_bernstein",
) -> SequentialAuditResult:
    """Audit a predictable candidate stream using fresh target batches.

    Candidate ``t`` may depend on results from earlier rounds, but its audit
    observations must be conditionally independent of that history. The
    quadratic schedule spends at most ``total_alpha`` over an unbounded
    sequence, so the chance of ever validating an invalid candidate is
    controlled by a union bound.
    """

    if not candidate_regrets:
        raise ValueError("At least one candidate proxy is required.")
    if violation_total_alpha is None:
        violation_total_alpha = total_alpha
    if not 0.0 < violation_total_alpha < 1.0:
        raise ValueError("violation_total_alpha must lie strictly between zero and one.")

    candidate_names = list(candidate_regrets)
    validation_spending = quadratic_alpha_schedule(total_alpha, len(candidate_names))
    violation_spending = quadratic_alpha_schedule(
        violation_total_alpha,
        len(candidate_names),
    )
    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for index, candidate in enumerate(candidate_names):
        round_number = index + 1
        result = audit_proxy_candidates(
            {candidate: candidate_regrets[candidate]},
            requirements=requirements,
            alpha=float(validation_spending[index]),
            violation_alpha=float(violation_spending[index]),
            bound_method=bound_method,
        )
        summary = result.candidate_summary.copy()
        summary.insert(0, "Round", round_number)
        summary["RoundAlpha"] = float(validation_spending[index])
        summary["CumulativeAlpha"] = float(validation_spending[: round_number].sum())
        detail = result.requirement_detail.copy()
        detail.insert(0, "Round", round_number)
        detail["RoundAlpha"] = float(validation_spending[index])
        summary_frames.append(summary)
        detail_frames.append(detail)

    return SequentialAuditResult(
        round_summary=pd.concat(summary_frames, ignore_index=True),
        requirement_detail=pd.concat(detail_frames, ignore_index=True),
    )


def audit_proxy_candidates(
    candidate_regrets: Mapping[str, Mapping[str, Sequence[float]]],
    requirements: Sequence[RiskRequirement],
    alpha: float = 0.05,
    violation_alpha: float | None = None,
    bound_method: str = "empirical_bernstein",
) -> ProxyAuditResult:
    """Audit a fixed candidate set on one independent target sample.

    Within a candidate, validity is an intersection claim: every declared
    bounded mean must be below its tolerance. The supplied observations may
    be relative regrets, absolute proxy losses, or another registered
    per-record audit quantity. The maximum requirement-level badness p-value
    is therefore a valid intersection-union p-value for the candidate. Holm
    adjustment across candidates controls the probability of validating at
    least one invalid candidate.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    if violation_alpha is None:
        violation_alpha = alpha
    if not 0.0 < violation_alpha < 1.0:
        raise ValueError("violation_alpha must lie strictly between zero and one.")
    if not candidate_regrets:
        raise ValueError("At least one candidate proxy is required.")
    if bound_method == "empirical_bernstein":
        badness_pvalue = empirical_bernstein_badness_pvalue
        violation_pvalue = empirical_bernstein_violation_pvalue
        lower_bound = empirical_bernstein_lower_bound
        upper_bound = empirical_bernstein_upper_bound
    elif bound_method == "hoeffding":
        badness_pvalue = hoeffding_badness_pvalue
        violation_pvalue = hoeffding_violation_pvalue
        lower_bound = hoeffding_lower_bound
        upper_bound = hoeffding_upper_bound
    else:
        raise ValueError("bound_method must be 'empirical_bernstein' or 'hoeffding'.")

    requirement_map = {requirement.name: requirement for requirement in requirements}
    if len(requirement_map) != len(requirements):
        raise ValueError("Requirement names must be unique.")
    if not requirement_map:
        raise ValueError("At least one risk requirement is required.")

    total_intervals = len(candidate_regrets) * len(requirement_map)
    # Both endpoints are reported simultaneously, so Bonferroni must cover
    # two tails for every candidate--requirement pair.
    interval_error = alpha / (2.0 * total_intervals)
    detail_rows: list[dict[str, float | int | str | bool]] = []
    candidate_pvalues: dict[str, float] = {}
    violation_pvalues: dict[str, float] = {}

    for candidate, regret_map in candidate_regrets.items():
        missing = set(requirement_map) - set(regret_map)
        extra = set(regret_map) - set(requirement_map)
        if missing or extra:
            raise ValueError(
                f"Candidate {candidate!r} has a mismatched requirement set; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

        badness_values: list[float] = []
        for requirement_name, requirement in requirement_map.items():
            values = _bounded_values(
                regret_map[requirement_name],
                requirement.lower,
                requirement.upper,
            )
            badness_p = badness_pvalue(
                values,
                requirement.tolerance,
                requirement.lower,
                requirement.upper,
            )
            violation_p = violation_pvalue(
                values,
                requirement.tolerance,
                requirement.lower,
                requirement.upper,
            )
            key = f"{candidate}::{requirement_name}"
            violation_pvalues[key] = violation_p
            badness_values.append(badness_p)
            detail_rows.append(
                {
                    "Candidate": candidate,
                    "Requirement": requirement_name,
                    "Estimand": requirement.estimand,
                    "N": int(values.size),
                    "MeanValue": float(values.mean()),
                    "MeanRegret": float(values.mean()),
                    "Tolerance": float(requirement.tolerance),
                    "LowerBound": float(requirement.lower),
                    "UpperBound": float(requirement.upper),
                    "BoundMethod": bound_method,
                    "BadnessPValue": badness_p,
                    "ViolationPValue": violation_p,
                    "SimultaneousLCB": lower_bound(
                        values,
                        interval_error,
                        requirement.lower,
                        requirement.upper,
                    ),
                    "SimultaneousUCB": upper_bound(
                        values,
                        interval_error,
                        requirement.lower,
                        requirement.upper,
                    ),
                }
            )
        candidate_pvalues[candidate] = max(badness_values)

    candidate_adjusted = holm_adjust(candidate_pvalues)
    violation_adjusted = holm_adjust(violation_pvalues)
    detail = pd.DataFrame(detail_rows)
    detail["ViolationAdjustedPValue"] = [
        violation_adjusted[f"{row.Candidate}::{row.Requirement}"]
        for row in detail.itertuples(index=False)
    ]
    detail["ViolationDetected"] = detail["ViolationAdjustedPValue"] <= violation_alpha

    summary_rows: list[dict[str, float | int | str | bool]] = []
    for candidate in candidate_regrets:
        candidate_detail = detail[detail["Candidate"] == candidate]
        validated = candidate_adjusted[candidate] <= alpha
        violation_detected = bool(candidate_detail["ViolationDetected"].any())
        if validated:
            status = "Validated"
        elif violation_detected:
            status = "Violation detected"
        else:
            status = "Unresolved"
        summary_rows.append(
            {
                "Candidate": candidate,
                "Requirements": len(requirement_map),
                "AuditNMin": int(candidate_detail["N"].min()),
                "CandidatePValue": candidate_pvalues[candidate],
                "HolmAdjustedPValue": candidate_adjusted[candidate],
                "Validated": validated,
                "ViolationDetected": violation_detected,
                "Status": status,
                "WorstUpperBound": float(candidate_detail["SimultaneousUCB"].max()),
                "WorstUpperRegretBound": float(candidate_detail["SimultaneousUCB"].max()),
            }
        )

    return ProxyAuditResult(
        candidate_summary=pd.DataFrame(summary_rows),
        requirement_detail=detail,
    )


def paired_prediction_losses(
    y_true: Sequence[int],
    source_probability: Sequence[float],
    proxy_probability: Sequence[float],
    source_thresholds: Mapping[float, float],
    proxy_thresholds: Mapping[float, float],
    record_ids: Sequence[object] | None = None,
    probability_clip: float = 1e-6,
) -> pd.DataFrame:
    """Build per-record source, proxy, and paired-regret losses."""

    labels = np.asarray(y_true, dtype=int).reshape(-1)
    source = np.asarray(source_probability, dtype=float).reshape(-1)
    proxy = np.asarray(proxy_probability, dtype=float).reshape(-1)
    if not (labels.size == source.size == proxy.size):
        raise ValueError("Labels and probability arrays must have the same length.")
    if labels.size == 0:
        raise ValueError("At least one audit record is required.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain binary labels.")
    if not np.isfinite(source).all() or not np.isfinite(proxy).all():
        raise ValueError("Probabilities must be finite.")
    if np.any((source < 0.0) | (source > 1.0)) or np.any((proxy < 0.0) | (proxy > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if set(source_thresholds) != set(proxy_thresholds):
        raise ValueError("Source and proxy threshold cost ratios must match.")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie between zero and one half.")

    if record_ids is None:
        identifiers = np.arange(labels.size)
    else:
        identifiers = np.asarray(record_ids, dtype=object).reshape(-1)
        if identifiers.size != labels.size:
            raise ValueError("record_ids must match the number of labels.")

    frame = pd.DataFrame(
        {
            "record_id": identifiers,
            "y_true": labels,
            "source_probability": source,
            "proxy_probability": proxy,
        }
    )

    source_brier = np.square(source - labels)
    proxy_brier = np.square(proxy - labels)
    frame["brier_source"] = source_brier
    frame["brier_proxy"] = proxy_brier
    frame["brier_regret"] = proxy_brier - source_brier

    clipped_source = np.clip(source, probability_clip, 1.0 - probability_clip)
    clipped_proxy = np.clip(proxy, probability_clip, 1.0 - probability_clip)
    source_log = -(labels * np.log(clipped_source) + (1 - labels) * np.log(1.0 - clipped_source))
    proxy_log = -(labels * np.log(clipped_proxy) + (1 - labels) * np.log(1.0 - clipped_proxy))
    max_log = -log(probability_clip)
    frame["logloss_source"] = source_log / max_log
    frame["logloss_proxy"] = proxy_log / max_log
    frame["logloss_regret"] = (proxy_log - source_log) / max_log

    for fn_cost in sorted(source_thresholds):
        source_threshold = float(source_thresholds[fn_cost])
        proxy_threshold = float(proxy_thresholds[fn_cost])
        if not 0.0 <= source_threshold <= 1.0 or not 0.0 <= proxy_threshold <= 1.0:
            raise ValueError("Decision thresholds must lie in [0, 1].")
        normalizer = max(float(fn_cost), 1.0)
        source_prediction = source >= source_threshold
        proxy_prediction = proxy >= proxy_threshold
        source_cost = (
            float(fn_cost) * ((labels == 1) & ~source_prediction)
            + ((labels == 0) & source_prediction)
        ) / normalizer
        proxy_cost = (
            float(fn_cost) * ((labels == 1) & ~proxy_prediction)
            + ((labels == 0) & proxy_prediction)
        ) / normalizer
        suffix = f"cost{int(fn_cost)}x"
        frame[f"{suffix}_source"] = source_cost
        frame[f"{suffix}_proxy"] = proxy_cost
        frame[f"{suffix}_regret"] = proxy_cost - source_cost

    return frame
