from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, floor, log
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.interpolate import BPoly, PPoly
from scipy.optimize import brentq, linprog
from scipy.stats import beta


@dataclass(frozen=True)
class SharedTargetReliabilityResult:
    """A one-sided reliability certificate from a shared target audit.

    ``certificate_mean`` is the observed two-sample U-statistic before the
    Bernstein approximation correction. ``soft_reliability_lower_bound``
    lower-bounds the mechanism probability of satisfying every registered
    requirement.
    """

    soft_reliability_lower_bound: float
    bernstein_mean_lower_bound: float
    certificate_mean: float
    approximation_correction: float
    releases: int
    target_records: int
    target_records_used: int
    target_blocks: int
    effective_sample_size: int
    requirements: int
    degree: int
    error_rate: float
    block_seed: int


@dataclass(frozen=True)
class SharedTargetWitnessResult:
    """Direct reliability inference from a bounded shared-target witness."""

    reliability_lower_bound: float
    witness_mean_lower_bound: float
    witness_mean: float
    invalid_release_witness_ceiling: float
    releases: int
    target_records: int
    target_records_used: int
    target_blocks: int
    effective_sample_size: int
    requirements: int
    block_size: int
    error_rate: float
    block_seed: int


@dataclass(frozen=True)
class SharedTargetPolynomialResult:
    """A direct shared-target certificate from polynomial minorants."""

    reliability_lower_bound: float
    certificate_mean_lower_bound: float
    certificate_mean: float
    releases: int
    target_records: int
    target_records_used: int
    target_blocks: int
    effective_sample_size: int
    requirements: int
    degree: int
    coefficient_floor: float
    valid_region_floors: tuple[float, ...]
    error_rate: float
    block_seed: int


@dataclass(frozen=True)
class SharedTargetTensorPolynomialResult:
    """A joint polynomial reliability certificate on a shared target."""

    reliability_lower_bound: float
    certificate_mean_lower_bound: float
    certificate_mean: float
    releases: int
    target_records: int
    target_records_used: int
    target_blocks: int
    effective_sample_size: int
    requirements: int
    degree: int
    coefficient_floor: float
    valid_region_floor: float
    error_rate: float
    block_seed: int


@dataclass(frozen=True)
class ConditionalSharedTargetResult:
    """A conditional-release reliability certificate on one shared target."""

    reliability_lower_bound: float
    conditional_score_lower_bound: float
    conditional_score_mean: float
    invalid_release_score_ceiling: float
    target_contamination_allowance: float
    releases: int
    target_records: int
    requirements: int
    error_rate: float
    release_error_rate: float
    target_error_rate: float


@dataclass(frozen=True)
class ConditionalSharedTargetPlan:
    """Best-case resolution limits for the binary shared-target score."""

    invalid_release_score_ceiling: float
    target_contamination_allowance: float
    minimum_target_records: int
    minimum_releases: int | None
    release_error_rate: float
    target_error_rate: float


def _as_requirement_vector(
    values: float | Sequence[float],
    requirements: int,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), requirements)
    array = array.reshape(-1)
    if array.size != requirements:
        raise ValueError(f"{name} must have one entry per requirement.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _soft_step(grid: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    return np.clip((threshold - grid) / margin, 0.0, 1.0)


def bernstein_soft_step_overshoot(
    *,
    threshold: float,
    margin: float,
    degree: int,
) -> float:
    """Return a conservative one-sided Bernstein approximation error.

    For ``psi(u) = clip((threshold - u) / margin, 0, 1)`` and its degree
    ``degree`` Bernstein polynomial ``B_degree psi``, the returned value is
    at least ``sup_u (B_degree psi(u) - psi(u))`` on ``[0, 1]``.

    The maximum is found from the endpoints, breakpoints, and all stationary
    points of the polynomial difference on each linear segment. A small
    upward numerical guard keeps the value conservative in floating-point
    arithmetic.
    """

    if degree < 1:
        raise ValueError("degree must be positive.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("normalized threshold must lie strictly between zero and one.")
    if not margin > 0.0:
        raise ValueError("margin must be positive.")

    grid = np.linspace(0.0, 1.0, degree + 1)
    coefficients = _soft_step(grid, threshold, margin)
    bernstein = BPoly(coefficients[:, None], [0.0, 1.0])
    derivative = PPoly.from_bernstein_basis(bernstein.derivative())

    breakpoints = np.unique(
        np.clip(
            np.asarray([0.0, threshold - margin, threshold, 1.0]),
            0.0,
            1.0,
        )
    )
    candidates: list[float] = breakpoints.tolist()
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        if right <= left:
            continue
        midpoint = float((left + right) / 2.0)
        slope = -1.0 / margin if threshold - margin < midpoint < threshold else 0.0
        shifted_coefficients = derivative.c.copy()
        shifted_coefficients[-1, :] -= slope
        stationary = PPoly(
            shifted_coefficients,
            derivative.x,
            extrapolate=False,
        ).roots(extrapolate=False)
        for root in stationary:
            root_value = float(np.real(root))
            if abs(float(np.imag(root))) <= 1e-10 and left < root_value < right:
                candidates.append(root_value)

    points = np.asarray(candidates, dtype=float)
    differences = np.asarray(bernstein(points), dtype=float).reshape(-1)
    differences -= _soft_step(points, threshold, margin)
    maximum = max(0.0, float(differences.max(initial=0.0)))
    numerical_guard = 128.0 * np.finfo(float).eps * (degree + 1)
    return float(min(1.0, maximum + numerical_guard))


def _poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0], dtype=float)
    for probability in probabilities:
        updated = np.zeros(pmf.size + 1, dtype=float)
        updated[:-1] += pmf * (1.0 - probability)
        updated[1:] += pmf * probability
        pmf = updated
    return pmf


def _poisson_binomial_pmfs(probabilities: np.ndarray) -> np.ndarray:
    """Vectorized Poisson-binomial PMFs along the last probability axis."""

    array = np.asarray(probabilities, dtype=float)
    if array.ndim < 1:
        raise ValueError("probabilities must have at least one dimension.")
    degree = array.shape[-1]
    pmf = np.zeros((*array.shape[:-1], degree + 1), dtype=float)
    pmf[..., 0] = 1.0
    for index in range(degree):
        probability = array[..., index, None]
        previous = pmf[..., : index + 1].copy()
        pmf[..., : index + 1] *= 1.0 - probability
        pmf[..., 1 : index + 2] += previous * probability
    return pmf


def _split_bernstein_coefficients(
    coefficients: np.ndarray,
    position: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split Bernstein coefficients at ``position`` by de Casteljau's rule."""

    if not 0.0 <= position <= 1.0:
        raise ValueError("position must lie in [0, 1].")
    level = np.asarray(coefficients, dtype=float)
    if level.ndim != 2 or level.shape[0] < 1:
        raise ValueError("coefficients must be a non-empty matrix.")
    degree = level.shape[0] - 1
    levels = [level]
    for _ in range(degree):
        previous = levels[-1]
        levels.append(
            (1.0 - position) * previous[:-1] + position * previous[1:]
        )
    left = np.stack([levels[index][0] for index in range(degree + 1)])
    right = np.stack(
        [levels[degree - index][index] for index in range(degree + 1)]
    )
    return left, right


def bernstein_restriction_matrix(
    degree: int,
    left: float,
    right: float,
) -> np.ndarray:
    """Map degree-``degree`` Bernstein coefficients from ``[0,1]`` to an interval."""

    if degree < 1:
        raise ValueError("degree must be positive.")
    if not 0.0 <= left < right <= 1.0:
        raise ValueError("The restriction interval must lie inside [0, 1].")
    coefficients = np.eye(degree + 1, dtype=float)
    if left > 0.0:
        _, coefficients = _split_bernstein_coefficients(coefficients, left)
    if right < 1.0:
        relative_right = (right - left) / (1.0 - left)
        coefficients, _ = _split_bernstein_coefficients(
            coefficients,
            relative_right,
        )
    return coefficients


def bernstein_step_minorant(
    *,
    threshold: float,
    margin: float,
    degree: int,
    coefficient_floor: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Construct a certified polynomial minorant of ``1{u < threshold}``.

    The returned Bernstein polynomial is at most one on ``[0,1]`` and at
    most zero on ``[threshold,1]``. It therefore lies below the hard validity
    indicator everywhere. The linear program maximizes a certified lower
    floor on ``[0, threshold - margin]`` while keeping every Bernstein
    coefficient in ``[-coefficient_floor, 1]``.
    """

    if degree < 1:
        raise ValueError("degree must be positive.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one.")
    if not 0.0 < margin < threshold:
        raise ValueError("margin must lie strictly between zero and threshold.")
    if coefficient_floor <= 0.0:
        raise ValueError("coefficient_floor must be positive.")

    valid_matrix = bernstein_restriction_matrix(
        degree,
        0.0,
        threshold - margin,
    )
    invalid_matrix = bernstein_restriction_matrix(
        degree,
        threshold,
        1.0,
    )
    variables = degree + 2
    objective = np.zeros(variables, dtype=float)
    objective[-1] = -1.0

    invalid_constraints = np.zeros(
        (invalid_matrix.shape[0], variables),
        dtype=float,
    )
    invalid_constraints[:, : degree + 1] = invalid_matrix
    valid_constraints = np.zeros(
        (valid_matrix.shape[0], variables),
        dtype=float,
    )
    valid_constraints[:, : degree + 1] = -valid_matrix
    valid_constraints[:, -1] = 1.0
    constraints = np.vstack([invalid_constraints, valid_constraints])
    upper = np.zeros(constraints.shape[0], dtype=float)
    bounds = [
        (-coefficient_floor, 1.0) for _ in range(degree + 1)
    ] + [(-coefficient_floor, 1.0)]
    solution = linprog(
        objective,
        A_ub=constraints,
        b_ub=upper,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Minorant optimization failed: {solution.message}")
    coefficients = np.asarray(solution.x[: degree + 1], dtype=float)
    valid_floor = float(solution.x[-1])
    return coefficients, valid_floor


def _tensor_axis_transform(
    transformation: np.ndarray,
    *,
    axis: int,
    dimensions: int,
) -> sparse.csr_matrix:
    size = transformation.shape[0]
    operator: sparse.spmatrix = sparse.csr_matrix([[1.0]])
    identity = sparse.eye(size, format="csr")
    transformed = sparse.csr_matrix(transformation)
    for index in range(dimensions):
        operator = sparse.kron(
            operator,
            transformed if index == axis else identity,
            format="csr",
        )
    return sparse.csr_matrix(operator)


def tensor_bernstein_validity_minorant(
    *,
    thresholds: Sequence[float],
    margins: Sequence[float],
    degree: int,
    coefficient_floor: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Construct a joint minorant of a rectangular validity indicator.

    The tensor Bernstein polynomial is at most zero whenever any coordinate
    reaches its invalid interval ``[threshold_j, 1]`` and at most one
    everywhere. It is therefore a global minorant of
    ``1{u_j < threshold_j for every j}``.
    """

    thresholds_array = np.asarray(thresholds, dtype=float).reshape(-1)
    margins_array = np.asarray(margins, dtype=float).reshape(-1)
    requirements = thresholds_array.size
    if requirements < 1 or margins_array.size != requirements:
        raise ValueError("thresholds and margins must have the same positive length.")
    if degree < 1:
        raise ValueError("degree must be positive.")
    if coefficient_floor <= 0.0:
        raise ValueError("coefficient_floor must be positive.")
    if np.any(thresholds_array <= 0.0) or np.any(thresholds_array >= 1.0):
        raise ValueError("Every threshold must lie strictly between zero and one.")
    if np.any(margins_array <= 0.0) or np.any(
        margins_array >= thresholds_array
    ):
        raise ValueError("Every margin must lie strictly between zero and its threshold.")
    coefficients, valid_floor = _tensor_bernstein_validity_minorant_cached(
        tuple(float(value) for value in thresholds_array),
        tuple(float(value) for value in margins_array),
        degree,
        float(coefficient_floor),
    )
    return coefficients.copy(), valid_floor


def _bernstein_basis_values(value: float, degree: int) -> np.ndarray:
    indices = np.arange(degree + 1)
    return np.asarray(
        [
            comb(degree, int(index))
            * value ** int(index)
            * (1.0 - value) ** (degree - int(index))
            for index in indices
        ],
        dtype=float,
    )


def tensor_bernstein_design_minorant(
    *,
    thresholds: Sequence[float],
    degree: int,
    design_points: np.ndarray,
    design_weights: Sequence[float] | None = None,
    coefficient_floor: float = 0.25,
) -> np.ndarray:
    """Optimize a valid joint minorant for a pre-target design distribution."""

    thresholds_array = np.asarray(thresholds, dtype=float).reshape(-1)
    points = np.asarray(design_points, dtype=float)
    requirements = thresholds_array.size
    if requirements < 1:
        raise ValueError("thresholds must be non-empty.")
    if points.ndim != 2 or points.shape[1] != requirements or points.shape[0] < 1:
        raise ValueError("design_points must have shape (points, requirements).")
    if not np.isfinite(points).all() or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("design_points must lie in [0, 1].")
    if np.any(thresholds_array <= 0.0) or np.any(thresholds_array >= 1.0):
        raise ValueError("Every threshold must lie strictly between zero and one.")
    if degree < 1:
        raise ValueError("degree must be positive.")
    if coefficient_floor <= 0.0:
        raise ValueError("coefficient_floor must be positive.")
    if design_weights is None:
        weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        weights = np.asarray(design_weights, dtype=float).reshape(-1)
        if weights.size != points.shape[0]:
            raise ValueError("design_weights must have one entry per design point.")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("design_weights must be finite and nonnegative.")
        if weights.sum() <= 0.0:
            raise ValueError("design_weights must have positive total weight.")
        weights = weights / weights.sum()
    coefficients = _tensor_bernstein_design_minorant_cached(
        tuple(float(value) for value in thresholds_array),
        degree,
        tuple(tuple(float(value) for value in row) for row in points),
        tuple(float(value) for value in weights),
        float(coefficient_floor),
    )
    return coefficients.copy()


@lru_cache(maxsize=128)
def _tensor_bernstein_design_minorant_cached(
    thresholds: tuple[float, ...],
    degree: int,
    design_points: tuple[tuple[float, ...], ...],
    design_weights: tuple[float, ...],
    coefficient_floor: float,
) -> np.ndarray:
    thresholds_array = np.asarray(thresholds, dtype=float)
    points = np.asarray(design_points, dtype=float)
    weights = np.asarray(design_weights, dtype=float)
    requirements = thresholds_array.size
    side = degree + 1
    coefficients_count = side**requirements

    invalid_constraints: list[sparse.csr_matrix] = []
    for requirement_index in range(requirements):
        restriction = bernstein_restriction_matrix(
            degree,
            float(thresholds_array[requirement_index]),
            1.0,
        )
        invalid_constraints.append(
            _tensor_axis_transform(
                restriction,
                axis=requirement_index,
                dimensions=requirements,
            )
        )
    constraints = sparse.vstack(invalid_constraints, format="csr")

    objective_weights = np.zeros((side,) * requirements, dtype=float)
    for point, weight in zip(points, weights, strict=True):
        point_basis: np.ndarray | float = np.asarray(1.0)
        for requirement_index in range(requirements):
            point_basis = np.multiply.outer(
                point_basis,
                _bernstein_basis_values(
                    float(point[requirement_index]),
                    degree,
                ),
            )
        objective_weights += weight * np.asarray(point_basis).reshape(
            objective_weights.shape
        )

    solution = linprog(
        -objective_weights.reshape(-1),
        A_ub=constraints,
        b_ub=np.zeros(constraints.shape[0], dtype=float),
        bounds=[
            (-coefficient_floor, 1.0)
            for _ in range(coefficients_count)
        ],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(
            f"Design minorant optimization failed: {solution.message}"
        )
    return np.asarray(solution.x, dtype=float).reshape(
        (side,) * requirements
    )


@lru_cache(maxsize=128)
def _tensor_bernstein_validity_minorant_cached(
    thresholds: tuple[float, ...],
    margins: tuple[float, ...],
    degree: int,
    coefficient_floor: float,
) -> tuple[np.ndarray, float]:
    thresholds_array = np.asarray(thresholds, dtype=float)
    margins_array = np.asarray(margins, dtype=float)
    requirements = thresholds_array.size
    side = degree + 1
    coefficients_count = side**requirements
    invalid_operators: list[sparse.csr_matrix] = []
    valid_operator: sparse.spmatrix = sparse.csr_matrix([[1.0]])
    for requirement_index in range(requirements):
        invalid_restriction = bernstein_restriction_matrix(
            degree,
            float(thresholds_array[requirement_index]),
            1.0,
        )
        invalid_operators.append(
            _tensor_axis_transform(
                invalid_restriction,
                axis=requirement_index,
                dimensions=requirements,
            )
        )
        valid_restriction = bernstein_restriction_matrix(
            degree,
            0.0,
            float(
                thresholds_array[requirement_index]
                - margins_array[requirement_index]
            ),
        )
        valid_operator = sparse.kron(
            valid_operator,
            sparse.csr_matrix(valid_restriction),
            format="csr",
        )

    zero_column = sparse.csr_matrix((coefficients_count, 1))
    invalid_constraints = [
        sparse.hstack([operator, zero_column], format="csr")
        for operator in invalid_operators
    ]
    valid_constraint = sparse.hstack(
        [
            -sparse.csr_matrix(valid_operator),
            sparse.csr_matrix(np.ones((coefficients_count, 1))),
        ],
        format="csr",
    )
    constraints = sparse.vstack(
        [*invalid_constraints, valid_constraint],
        format="csr",
    )
    objective = np.zeros(coefficients_count + 1, dtype=float)
    objective[-1] = -1.0
    bounds = [
        (-coefficient_floor, 1.0) for _ in range(coefficients_count)
    ] + [(-coefficient_floor, 1.0)]
    solution = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.zeros(constraints.shape[0], dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(
            f"Joint minorant optimization failed: {solution.message}"
        )
    shape = (side,) * requirements
    coefficients = np.asarray(
        solution.x[:coefficients_count],
        dtype=float,
    ).reshape(shape)
    valid_floor = float(solution.x[-1])
    return coefficients, valid_floor


def bounded_kl_lower_bound(
    sample_mean: float,
    sample_size: int,
    error_rate: float,
) -> float:
    """Invert the Bernoulli-KL lower tail for observations in ``[0, 1]``."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    mean = float(sample_mean)
    if not 0.0 <= mean <= 1.0:
        raise ValueError("sample_mean must lie in [0, 1].")
    if mean == 0.0:
        return 0.0
    if mean == 1.0:
        return float(error_rate ** (1.0 / sample_size))

    target = log(1.0 / error_rate) / sample_size

    def binary_kl(candidate: float) -> float:
        return mean * log(mean / candidate) + (1.0 - mean) * log(
            (1.0 - mean) / (1.0 - candidate)
        )

    lower_endpoint = np.finfo(float).tiny
    return float(
        brentq(
            lambda candidate: binary_kl(candidate) - target,
            lower_endpoint,
            mean,
        )
    )


def _bernoulli_kl(left: float, right: float) -> float:
    if not 0.0 <= left <= right <= 1.0:
        raise ValueError("Bernoulli means must satisfy 0 <= left <= right <= 1.")
    if left == right:
        return 0.0
    if right == 1.0:
        return float("inf")
    first = 0.0 if left == 0.0 else left * log(left / right)
    second = (
        0.0
        if left == 1.0
        else (1.0 - left) * log((1.0 - left) / (1.0 - right))
    )
    return first + second


def shared_target_polynomial_lower_bound(
    losses: np.ndarray,
    *,
    tolerances: float | Sequence[float],
    margins: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    degree: int = 12,
    coefficient_floor: float = 1.0,
    error_rate: float = 0.05,
    mechanisms: int = 1,
    block_seed: int = 0,
) -> SharedTargetPolynomialResult:
    """Lower-bound reliability from polynomial moments on one shared target.

    Each requirement receives a polynomial ``q_j`` satisfying
    ``q_j(u) <= 1{u < tolerance_j}``. Hence

    ``sum_j q_j(mu_j(Q)) - (requirements - 1)``

    is a pointwise minorant of the release-validity indicator. Distinct
    records within each target block give an unbiased Bernstein-moment
    estimator. The complete release-by-block average is a bounded two-sample
    U-statistic, so a matching argument yields a Bernoulli-KL lower bound
    with effective sample size ``min(releases, target_blocks)``.
    """

    array = np.asarray(losses, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "losses must have shape (releases, target_records, requirements)."
        )
    releases, target_records, requirements = array.shape
    if releases < 1 or target_records < 1 or requirements < 1:
        raise ValueError("losses must have non-empty release, target, and requirement axes.")
    if not np.isfinite(array).all():
        raise ValueError("losses must contain only finite values.")
    if degree < 1:
        raise ValueError("degree must be positive.")
    if coefficient_floor <= 0.0:
        raise ValueError("coefficient_floor must be positive.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    margins_array = _as_requirement_vector(
        margins,
        requirements,
        name="margins",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(margins_array <= 0.0):
        raise ValueError("Every margin must be positive.")
    tolerance = 1e-12
    if np.any(array < lower_array.reshape(1, 1, -1) - tolerance) or np.any(
        array > upper_array.reshape(1, 1, -1) + tolerance
    ):
        raise ValueError("Observed losses fall outside their registered bounds.")

    widths = upper_array - lower_array
    normalized = (array - lower_array.reshape(1, 1, -1)) / widths.reshape(
        1,
        1,
        -1,
    )
    normalized_thresholds = (tolerances_array - lower_array) / widths
    normalized_margins = margins_array / widths
    if np.any(normalized_thresholds <= 0.0) or np.any(normalized_thresholds >= 1.0):
        raise ValueError("Every normalized tolerance must lie strictly inside [0, 1].")
    if np.any(normalized_margins >= normalized_thresholds):
        raise ValueError("Every normalized margin must be smaller than its tolerance.")

    target_blocks = target_records // degree
    if target_blocks < 1:
        raise ValueError("Not enough target records for one polynomial block.")
    rng = np.random.default_rng(block_seed)
    positions = rng.permutation(target_records)[: target_blocks * degree]
    blocks = positions.reshape(target_blocks, degree)

    coefficient_vectors: list[np.ndarray] = []
    valid_region_floors: list[float] = []
    for requirement_index in range(requirements):
        coefficients, valid_floor = bernstein_step_minorant(
            threshold=float(normalized_thresholds[requirement_index]),
            margin=float(normalized_margins[requirement_index]),
            degree=degree,
            coefficient_floor=coefficient_floor,
        )
        coefficient_vectors.append(coefficients)
        valid_region_floors.append(valid_floor)

    certificate = np.full(
        (releases, target_blocks),
        -(requirements - 1.0),
        dtype=float,
    )
    for release_index in range(releases):
        for block_index in range(target_blocks):
            for requirement_index in range(requirements):
                probabilities = normalized[
                    release_index,
                    blocks[block_index],
                    requirement_index,
                ]
                pmf = _poisson_binomial_pmf(probabilities)
                certificate[release_index, block_index] += float(
                    np.dot(pmf, coefficient_vectors[requirement_index])
                )

    certificate_mean = float(certificate.mean())
    certificate_min = -requirements * coefficient_floor - (requirements - 1.0)
    certificate_max = 1.0
    certificate_range = certificate_max - certificate_min
    normalized_mean = (certificate_mean - certificate_min) / certificate_range
    effective_sample_size = min(releases, target_blocks)
    normalized_lower = bounded_kl_lower_bound(
        normalized_mean,
        effective_sample_size,
        error_rate / mechanisms,
    )
    certificate_lower = certificate_min + certificate_range * normalized_lower
    reliability_lower = max(0.0, certificate_lower)
    return SharedTargetPolynomialResult(
        reliability_lower_bound=float(min(1.0, reliability_lower)),
        certificate_mean_lower_bound=certificate_lower,
        certificate_mean=certificate_mean,
        releases=releases,
        target_records=target_records,
        target_records_used=target_blocks * degree,
        target_blocks=target_blocks,
        effective_sample_size=effective_sample_size,
        requirements=requirements,
        degree=degree,
        coefficient_floor=coefficient_floor,
        valid_region_floors=tuple(valid_region_floors),
        error_rate=error_rate,
        block_seed=block_seed,
    )


def shared_target_tensor_polynomial_lower_bound(
    losses: np.ndarray,
    *,
    tolerances: float | Sequence[float],
    margins: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    degree: int = 8,
    coefficient_floor: float = 0.25,
    design_points: np.ndarray | None = None,
    design_weights: Sequence[float] | None = None,
    error_rate: float = 0.05,
    mechanisms: int = 1,
    block_seed: int = 0,
) -> SharedTargetTensorPolynomialResult:
    """Infer mechanism reliability with a joint polynomial moment certificate."""

    array = np.asarray(losses, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "losses must have shape (releases, target_records, requirements)."
        )
    releases, target_records, requirements = array.shape
    if releases < 1 or target_records < 1 or requirements < 1:
        raise ValueError("losses must have non-empty release, target, and requirement axes.")
    if not np.isfinite(array).all():
        raise ValueError("losses must contain only finite values.")
    if degree < 1:
        raise ValueError("degree must be positive.")
    if coefficient_floor <= 0.0:
        raise ValueError("coefficient_floor must be positive.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    margins_array = _as_requirement_vector(
        margins,
        requirements,
        name="margins",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(margins_array <= 0.0):
        raise ValueError("Every margin must be positive.")
    numerical_tolerance = 1e-12
    if np.any(array < lower_array.reshape(1, 1, -1) - numerical_tolerance) or np.any(
        array > upper_array.reshape(1, 1, -1) + numerical_tolerance
    ):
        raise ValueError("Observed losses fall outside their registered bounds.")

    widths = upper_array - lower_array
    normalized = (array - lower_array.reshape(1, 1, -1)) / widths.reshape(
        1,
        1,
        -1,
    )
    normalized_thresholds = (tolerances_array - lower_array) / widths
    normalized_margins = margins_array / widths
    if design_points is None:
        coefficients, valid_floor = tensor_bernstein_validity_minorant(
            thresholds=normalized_thresholds,
            margins=normalized_margins,
            degree=degree,
            coefficient_floor=coefficient_floor,
        )
    else:
        original_points = np.asarray(design_points, dtype=float)
        if original_points.ndim != 2 or original_points.shape[1] != requirements:
            raise ValueError(
                "design_points must have shape (points, requirements)."
            )
        normalized_points = (
            original_points - lower_array.reshape(1, -1)
        ) / widths.reshape(1, -1)
        coefficients = tensor_bernstein_design_minorant(
            thresholds=normalized_thresholds,
            degree=degree,
            design_points=normalized_points,
            design_weights=design_weights,
            coefficient_floor=coefficient_floor,
        )
        valid_transformation: sparse.spmatrix = sparse.csr_matrix([[1.0]])
        for requirement_index in range(requirements):
            valid_transformation = sparse.kron(
                valid_transformation,
                sparse.csr_matrix(
                    bernstein_restriction_matrix(
                        degree,
                        0.0,
                        float(
                            normalized_thresholds[requirement_index]
                            - normalized_margins[requirement_index]
                        ),
                    )
                ),
                format="csr",
            )
        valid_floor = float(
            np.asarray(
                valid_transformation @ coefficients.reshape(-1)
            ).min()
        )

    block_width = degree * requirements
    target_blocks = target_records // block_width
    if target_blocks < 1:
        raise ValueError("Not enough target records for one joint polynomial block.")
    rng = np.random.default_rng(block_seed)
    positions = rng.permutation(target_records)[: target_blocks * block_width]
    blocks = positions.reshape(target_blocks, requirements, degree)

    pmfs = [
        _poisson_binomial_pmfs(
            normalized[:, blocks[:, requirement_index, :], requirement_index]
        )
        for requirement_index in range(requirements)
    ]
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if requirements > len(labels):
        raise ValueError("Too many requirements for tensor contraction.")
    coefficient_labels = labels[:requirements]
    expression = ",".join(
        [f"...{label}" for label in coefficient_labels]
        + [coefficient_labels]
    )
    certificate = np.einsum(
        f"{expression}->...",
        *pmfs,
        coefficients,
        optimize=True,
    )

    certificate_mean = float(certificate.mean())
    normalized_mean = (
        certificate_mean + coefficient_floor
    ) / (1.0 + coefficient_floor)
    effective_sample_size = min(releases, target_blocks)
    normalized_lower = bounded_kl_lower_bound(
        normalized_mean,
        effective_sample_size,
        error_rate / mechanisms,
    )
    certificate_lower = (
        -coefficient_floor
        + (1.0 + coefficient_floor) * normalized_lower
    )
    reliability_lower = max(0.0, certificate_lower)
    return SharedTargetTensorPolynomialResult(
        reliability_lower_bound=float(min(1.0, reliability_lower)),
        certificate_mean_lower_bound=certificate_lower,
        certificate_mean=certificate_mean,
        releases=releases,
        target_records=target_records,
        target_records_used=target_blocks * block_width,
        target_blocks=target_blocks,
        effective_sample_size=effective_sample_size,
        requirements=requirements,
        degree=degree,
        coefficient_floor=coefficient_floor,
        valid_region_floor=valid_floor,
        error_rate=error_rate,
        block_seed=block_seed,
    )


def shared_target_reliability_lower_bound(
    losses: np.ndarray,
    *,
    tolerances: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    margins: float | Sequence[float] = 0.1,
    degree: int = 8,
    error_rate: float = 0.05,
    mechanisms: int = 1,
    block_seed: int = 0,
) -> SharedTargetReliabilityResult:
    """Lower-bound mechanism reliability from one shared target sample.

    Parameters
    ----------
    losses:
        Array with shape ``(releases, target_records, requirements)``. Every
        release is evaluated on the same target records.
    tolerances, lower_bounds, upper_bounds:
        Registered bounded-mean requirements. They are normalized to
        ``[0, 1]`` before the polynomial certificate is evaluated.
    margins:
        Width of the soft validity transition in normalized risk units.
    degree:
        Bernstein degree per requirement. One target block contains
        ``degree * requirements`` distinct target records.
    error_rate:
        Familywise error allocated to the direct lower bounds.
    mechanisms:
        Number of mechanisms covered simultaneously by equal allocation.
    block_seed:
        Seed for an outcome-independent permutation of target positions.

    Notes
    -----
    The estimator is a complete cross-average over independent releases and
    disjoint target blocks. Its bounded-kernel lower confidence bound uses
    effective sample size ``min(releases, target_blocks)``. Mechanisms may
    share the target blocks; equal allocation gives simultaneous coverage.
    """

    array = np.asarray(losses, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "losses must have shape (releases, target_records, requirements)."
        )
    releases, target_records, requirements = array.shape
    if releases < 1 or target_records < 1 or requirements < 1:
        raise ValueError("losses must have non-empty release, target, and requirement axes.")
    if not np.isfinite(array).all():
        raise ValueError("losses must contain only finite values.")
    if degree < 1:
        raise ValueError("degree must be positive.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    margins_array = _as_requirement_vector(
        margins,
        requirements,
        name="margins",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(margins_array <= 0.0):
        raise ValueError("Every margin must be positive.")
    tolerance = 1e-12
    if np.any(array < lower_array.reshape(1, 1, -1) - tolerance) or np.any(
        array > upper_array.reshape(1, 1, -1) + tolerance
    ):
        raise ValueError("Observed losses fall outside their registered bounds.")

    widths = upper_array - lower_array
    normalized = (array - lower_array.reshape(1, 1, -1)) / widths.reshape(
        1,
        1,
        -1,
    )
    normalized_thresholds = (tolerances_array - lower_array) / widths
    normalized_margins = margins_array / widths
    if np.any(normalized_thresholds <= 0.0) or np.any(normalized_thresholds >= 1.0):
        raise ValueError("Every normalized tolerance must lie strictly inside [0, 1].")

    block_width = degree * requirements
    target_blocks = target_records // block_width
    if target_blocks < 1:
        raise ValueError(
            "Not enough target records for one disjoint certificate block."
        )
    rng = np.random.default_rng(block_seed)
    positions = rng.permutation(target_records)[: target_blocks * block_width]
    blocks = positions.reshape(target_blocks, requirements, degree)

    grid = np.linspace(0.0, 1.0, degree + 1)
    soft_values = [
        _soft_step(grid, normalized_thresholds[index], normalized_margins[index])
        for index in range(requirements)
    ]
    approximation_correction = float(
        sum(
            bernstein_soft_step_overshoot(
                threshold=float(normalized_thresholds[index]),
                margin=float(normalized_margins[index]),
                degree=degree,
            )
            for index in range(requirements)
        )
    )

    certificate_scores = np.ones((releases, target_blocks), dtype=float)
    for release_index in range(releases):
        for block_index in range(target_blocks):
            score = 1.0
            for requirement_index in range(requirements):
                probabilities = normalized[
                    release_index,
                    blocks[block_index, requirement_index],
                    requirement_index,
                ]
                pmf = _poisson_binomial_pmf(probabilities)
                score *= float(np.dot(pmf, soft_values[requirement_index]))
            certificate_scores[release_index, block_index] = score

    certificate_mean = float(certificate_scores.mean())
    effective_sample_size = min(releases, target_blocks)
    allocated_error = error_rate / mechanisms
    bernstein_mean_lower = bounded_kl_lower_bound(
        certificate_mean,
        effective_sample_size,
        allocated_error,
    )
    reliability_lower = max(
        0.0,
        bernstein_mean_lower - approximation_correction,
    )
    return SharedTargetReliabilityResult(
        soft_reliability_lower_bound=float(min(1.0, reliability_lower)),
        bernstein_mean_lower_bound=bernstein_mean_lower,
        certificate_mean=certificate_mean,
        approximation_correction=approximation_correction,
        releases=releases,
        target_records=target_records,
        target_records_used=target_blocks * block_width,
        target_blocks=target_blocks,
        effective_sample_size=effective_sample_size,
        requirements=requirements,
        degree=degree,
        error_rate=error_rate,
        block_seed=block_seed,
    )


def shared_target_witness_lower_bound(
    losses: np.ndarray,
    *,
    tolerances: float | Sequence[float],
    slacks: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    block_size: int = 100,
    ramp_widths: float | Sequence[float] = 0.0,
    error_rate: float = 0.05,
    mechanisms: int = 1,
    block_seed: int = 0,
) -> SharedTargetWitnessResult:
    """Infer reliability directly from one shared target sample.

    Target records are partitioned, before outcomes are used, into disjoint
    blocks. For each release and block, the witness is positive only when
    every block-average loss falls at least ``slack`` below its registered
    limit. If a release is invalid, the bounded-loss Bernoulli-KL Chernoff
    bound limits its expected witness by ``kappa``. Therefore

    ``E[witness] <= kappa + (1 - kappa) * reliability``.

    The observed release-by-block cross-average is a two-sample U-statistic.
    A Bernoulli-KL inversion with effective sample size
    ``min(releases, blocks)`` lower-bounds its expectation without assuming
    independence or PRDS among release-level p-values.

    Positive ``ramp_widths`` replace the hard block indicator by a continuous
    score. The score is one below ``limit - slack - ramp``, decreases linearly
    to zero, and remains zero at or above ``limit - slack``.
    """

    array = np.asarray(losses, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "losses must have shape (releases, target_records, requirements)."
        )
    releases, target_records, requirements = array.shape
    if releases < 1 or target_records < 1 or requirements < 1:
        raise ValueError("losses must have non-empty release, target, and requirement axes.")
    if not np.isfinite(array).all():
        raise ValueError("losses must contain only finite values.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    slacks_array = _as_requirement_vector(
        slacks,
        requirements,
        name="slacks",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    ramps_array = _as_requirement_vector(
        ramp_widths,
        requirements,
        name="ramp_widths",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(slacks_array <= 0.0):
        raise ValueError("Every slack must be positive.")
    if np.any(ramps_array < 0.0):
        raise ValueError("Ramp widths cannot be negative.")
    cutoffs = tolerances_array - slacks_array
    if np.any(cutoffs < lower_array) or np.any(cutoffs > upper_array):
        raise ValueError("Every tolerance minus slack must lie within its loss bounds.")
    tolerance = 1e-12
    if np.any(array < lower_array.reshape(1, 1, -1) - tolerance) or np.any(
        array > upper_array.reshape(1, 1, -1) + tolerance
    ):
        raise ValueError("Observed losses fall outside their registered bounds.")

    target_blocks = target_records // block_size
    if target_blocks < 1:
        raise ValueError("Not enough target records for one witness block.")
    rng = np.random.default_rng(block_seed)
    positions = rng.permutation(target_records)[: target_blocks * block_size]
    blocks = positions.reshape(target_blocks, block_size)
    block_means = array[:, blocks, :].mean(axis=2)

    factors = np.empty_like(block_means)
    for requirement_index in range(requirements):
        if ramps_array[requirement_index] == 0.0:
            factors[:, :, requirement_index] = (
                block_means[:, :, requirement_index]
                <= cutoffs[requirement_index]
            )
        else:
            factors[:, :, requirement_index] = np.clip(
                (
                    cutoffs[requirement_index]
                    - block_means[:, :, requirement_index]
                )
                / ramps_array[requirement_index],
                0.0,
                1.0,
            )
    witness = factors.prod(axis=2)
    witness_mean = float(witness.mean())

    widths = upper_array - lower_array
    normalized_cutoffs = (cutoffs - lower_array) / widths
    normalized_tolerances = (tolerances_array - lower_array) / widths
    if np.any(normalized_tolerances > 1.0):
        raise ValueError("Every tolerance must not exceed its upper loss bound.")
    invalid_ceiling = float(
        np.max(
            [
                np.exp(
                    -block_size
                    * _bernoulli_kl(
                        float(normalized_cutoffs[index]),
                        float(normalized_tolerances[index]),
                    )
                )
                for index in range(requirements)
            ]
        )
    )
    (
        reliability_lower,
        witness_mean_lower,
        effective_sample_size,
    ) = witness_reliability_lower_from_mean(
        witness_mean,
        releases=releases,
        target_blocks=target_blocks,
        invalid_release_witness_ceiling=invalid_ceiling,
        error_rate=error_rate,
        mechanisms=mechanisms,
    )
    return SharedTargetWitnessResult(
        reliability_lower_bound=float(min(1.0, reliability_lower)),
        witness_mean_lower_bound=witness_mean_lower,
        witness_mean=witness_mean,
        invalid_release_witness_ceiling=invalid_ceiling,
        releases=releases,
        target_records=target_records,
        target_records_used=target_blocks * block_size,
        target_blocks=target_blocks,
        effective_sample_size=effective_sample_size,
        requirements=requirements,
        block_size=block_size,
        error_rate=error_rate,
        block_seed=block_seed,
    )


def shared_target_conditional_witness_lower_bound(
    losses: np.ndarray,
    *,
    tolerances: float | Sequence[float],
    slacks: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    ramp_widths: float | Sequence[float] = 0.0,
    error_rate: float = 0.05,
    target_error_fraction: float = 0.5,
    mechanisms: int = 1,
) -> ConditionalSharedTargetResult:
    """Lower-bound reliability by conditioning on one shared target audit.

    Conditional on the realized target records, independent releases give
    independent bounded scores. A Bernoulli-KL inversion lower-bounds their
    conditional mean. An invalid release can score positively only after a
    target lower-tail deviation. A bounded-loss Chernoff bound controls its
    expected contribution, and Markov's inequality converts that expectation
    into a simultaneous target-contamination allowance.
    """

    array = np.asarray(losses, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "losses must have shape (releases, target_records, requirements)."
        )
    releases, target_records, requirements = array.shape
    if releases < 1 or target_records < 1 or requirements < 1:
        raise ValueError("losses must have non-empty release, target, and requirement axes.")
    if not np.isfinite(array).all():
        raise ValueError("losses must contain only finite values.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    if not 0.0 < target_error_fraction < 1.0:
        raise ValueError("target_error_fraction must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    slacks_array = _as_requirement_vector(
        slacks,
        requirements,
        name="slacks",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    ramps_array = _as_requirement_vector(
        ramp_widths,
        requirements,
        name="ramp_widths",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(slacks_array <= 0.0):
        raise ValueError("Every slack must be positive.")
    if np.any(ramps_array < 0.0):
        raise ValueError("Ramp widths cannot be negative.")
    cutoffs = tolerances_array - slacks_array
    if (
        np.any(cutoffs <= lower_array)
        or np.any(cutoffs >= tolerances_array)
        or np.any(tolerances_array >= upper_array)
    ):
        raise ValueError(
            "Each requirement must satisfy lower < tolerance - slack "
            "< tolerance < upper."
        )
    numerical_tolerance = 1e-12
    if np.any(array < lower_array.reshape(1, 1, -1) - numerical_tolerance) or np.any(
        array > upper_array.reshape(1, 1, -1) + numerical_tolerance
    ):
        raise ValueError("Observed losses fall outside their registered bounds.")

    return shared_target_conditional_mean_lower_bound(
        array.mean(axis=1),
        target_records=target_records,
        tolerances=tolerances_array,
        slacks=slacks_array,
        lower_bounds=lower_array,
        upper_bounds=upper_array,
        ramp_widths=ramps_array,
        error_rate=error_rate,
        target_error_fraction=target_error_fraction,
        mechanisms=mechanisms,
    )


def shared_target_conditional_mean_lower_bound(
    release_means: np.ndarray,
    *,
    target_records: int,
    tolerances: float | Sequence[float],
    slacks: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    ramp_widths: float | Sequence[float] = 0.0,
    error_rate: float = 0.05,
    target_error_fraction: float = 0.5,
    mechanisms: int = 1,
) -> ConditionalSharedTargetResult:
    """Evaluate the conditional certificate from registered release means.

    This is equivalent to :func:`shared_target_conditional_witness_lower_bound`
    once the shared-target means have been computed.
    """

    means = np.asarray(release_means, dtype=float)
    if means.ndim != 2:
        raise ValueError("release_means must have shape (releases, requirements).")
    releases, requirements = means.shape
    if releases < 1 or requirements < 1:
        raise ValueError("release_means must have non-empty axes.")
    if target_records < 1:
        raise ValueError("target_records must be positive.")
    if not np.isfinite(means).all():
        raise ValueError("release_means must contain only finite values.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    if not 0.0 < target_error_fraction < 1.0:
        raise ValueError("target_error_fraction must lie strictly between zero and one.")

    tolerances_array = _as_requirement_vector(
        tolerances,
        requirements,
        name="tolerances",
    )
    slacks_array = _as_requirement_vector(
        slacks,
        requirements,
        name="slacks",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    ramps_array = _as_requirement_vector(
        ramp_widths,
        requirements,
        name="ramp_widths",
    )
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any(slacks_array <= 0.0):
        raise ValueError("Every slack must be positive.")
    if np.any(ramps_array < 0.0):
        raise ValueError("Ramp widths cannot be negative.")
    cutoffs = tolerances_array - slacks_array
    if (
        np.any(cutoffs <= lower_array)
        or np.any(cutoffs >= tolerances_array)
        or np.any(tolerances_array >= upper_array)
    ):
        raise ValueError(
            "Each requirement must satisfy lower < tolerance - slack "
            "< tolerance < upper."
        )
    numerical_tolerance = 1e-12
    if np.any(means < lower_array.reshape(1, -1) - numerical_tolerance) or np.any(
        means > upper_array.reshape(1, -1) + numerical_tolerance
    ):
        raise ValueError("Release means fall outside their registered bounds.")

    release_means = means
    factors = np.empty_like(release_means)
    for requirement_index in range(requirements):
        if ramps_array[requirement_index] == 0.0:
            factors[:, requirement_index] = (
                release_means[:, requirement_index]
                <= cutoffs[requirement_index]
            )
        else:
            factors[:, requirement_index] = np.clip(
                (
                    cutoffs[requirement_index]
                    - release_means[:, requirement_index]
                )
                / ramps_array[requirement_index],
                0.0,
                1.0,
            )
    scores = factors.prod(axis=1)
    score_mean = float(scores.mean())

    widths = upper_array - lower_array
    normalized_cutoffs = (cutoffs - lower_array) / widths
    normalized_tolerances = (tolerances_array - lower_array) / widths
    if np.any(normalized_tolerances > 1.0):
        raise ValueError("Every tolerance must not exceed its upper loss bound.")
    invalid_ceiling = float(
        np.max(
            [
                np.exp(
                    -target_records
                    * _bernoulli_kl(
                        float(normalized_cutoffs[index]),
                        float(normalized_tolerances[index]),
                    )
                )
                for index in range(requirements)
            ]
        )
    )

    target_error_rate = error_rate * target_error_fraction / mechanisms
    release_error_rate = (
        error_rate * (1.0 - target_error_fraction) / mechanisms
    )
    if np.all(ramps_array == 0.0):
        successes = int(scores.sum())
        score_lower = (
            0.0
            if successes == 0
            else float(
                beta.ppf(
                    release_error_rate,
                    successes,
                    releases - successes + 1,
                )
            )
        )
    else:
        score_lower = bounded_kl_lower_bound(
            score_mean,
            releases,
            release_error_rate,
        )
    contamination_allowance = min(
        1.0,
        invalid_ceiling / target_error_rate,
    )
    reliability_lower = max(
        0.0,
        score_lower - contamination_allowance,
    )
    return ConditionalSharedTargetResult(
        reliability_lower_bound=float(min(1.0, reliability_lower)),
        conditional_score_lower_bound=score_lower,
        conditional_score_mean=score_mean,
        invalid_release_score_ceiling=invalid_ceiling,
        target_contamination_allowance=contamination_allowance,
        releases=releases,
        target_records=target_records,
        requirements=requirements,
        error_rate=error_rate,
        release_error_rate=release_error_rate,
        target_error_rate=target_error_rate,
    )


def plan_conditional_shared_target(
    *,
    target_records: int,
    minimum_reliability: float,
    tolerances: float | Sequence[float],
    slacks: float | Sequence[float],
    lower_bounds: float | Sequence[float] = 0.0,
    upper_bounds: float | Sequence[float] = 1.0,
    error_rate: float = 0.05,
    target_error_fraction: float = 0.5,
    mechanisms: int = 1,
) -> ConditionalSharedTargetPlan:
    """Compute necessary best-case sample sizes for the binary direct bound.

    The calculation assumes that every sampled release scores one. It therefore
    gives a necessary planning threshold, not a power guarantee.
    """

    if target_records < 1:
        raise ValueError("target_records must be positive.")
    if not 0.0 < minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie strictly between zero and one.")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly between zero and one.")
    if not 0.0 < target_error_fraction < 1.0:
        raise ValueError("target_error_fraction must lie strictly between zero and one.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")

    tolerances_array = np.asarray(tolerances, dtype=float).reshape(-1)
    requirements = tolerances_array.size
    if requirements < 1:
        raise ValueError("tolerances must contain at least one requirement.")
    slacks_array = _as_requirement_vector(
        slacks,
        requirements,
        name="slacks",
    )
    lower_array = _as_requirement_vector(
        lower_bounds,
        requirements,
        name="lower_bounds",
    )
    upper_array = _as_requirement_vector(
        upper_bounds,
        requirements,
        name="upper_bounds",
    )
    cutoffs = tolerances_array - slacks_array
    if (
        np.any(lower_array >= upper_array)
        or np.any(cutoffs <= lower_array)
        or np.any(cutoffs >= tolerances_array)
        or np.any(tolerances_array >= upper_array)
    ):
        raise ValueError(
            "Each requirement must satisfy lower < tolerance - slack "
            "< tolerance < upper."
        )

    widths = upper_array - lower_array
    normalized_cutoffs = (cutoffs - lower_array) / widths
    normalized_tolerances = (tolerances_array - lower_array) / widths
    divergences = np.asarray(
        [
            _bernoulli_kl(float(left), float(right))
            for left, right in zip(
                normalized_cutoffs,
                normalized_tolerances,
                strict=True,
            )
        ],
        dtype=float,
    )
    minimum_divergence = float(divergences.min())
    target_error_rate = error_rate * target_error_fraction / mechanisms
    release_error_rate = (
        error_rate * (1.0 - target_error_fraction) / mechanisms
    )
    minimum_target_records = (
        floor(
            log(1.0 / (target_error_rate * (1.0 - minimum_reliability)))
            / minimum_divergence
        )
        + 1
    )
    invalid_ceiling = float(np.exp(-target_records * minimum_divergence))
    contamination_allowance = min(1.0, invalid_ceiling / target_error_rate)
    best_case_threshold = minimum_reliability + contamination_allowance
    minimum_releases: int | None
    if best_case_threshold >= 1.0:
        minimum_releases = None
    else:
        minimum_releases = (
            floor(log(release_error_rate) / log(best_case_threshold)) + 1
        )
    return ConditionalSharedTargetPlan(
        invalid_release_score_ceiling=invalid_ceiling,
        target_contamination_allowance=contamination_allowance,
        minimum_target_records=minimum_target_records,
        minimum_releases=minimum_releases,
        release_error_rate=release_error_rate,
        target_error_rate=target_error_rate,
    )


def witness_reliability_lower_from_mean(
    witness_mean: float,
    *,
    releases: int,
    target_blocks: int,
    invalid_release_witness_ceiling: float,
    error_rate: float,
    mechanisms: int = 1,
) -> tuple[float, float, int]:
    """Convert a shared-target witness mean into a reliability lower bound."""

    if releases < 1 or target_blocks < 1:
        raise ValueError("releases and target_blocks must be positive.")
    if mechanisms < 1:
        raise ValueError("mechanisms must be positive.")
    if not 0.0 <= witness_mean <= 1.0:
        raise ValueError("witness_mean must lie in [0, 1].")
    if not 0.0 <= invalid_release_witness_ceiling <= 1.0:
        raise ValueError(
            "invalid_release_witness_ceiling must lie in [0, 1]."
        )
    effective_sample_size = min(releases, target_blocks)
    witness_mean_lower = bounded_kl_lower_bound(
        witness_mean,
        effective_sample_size,
        error_rate / mechanisms,
    )
    if invalid_release_witness_ceiling >= 1.0:
        reliability_lower = 0.0
    else:
        reliability_lower = max(
            0.0,
            (witness_mean_lower - invalid_release_witness_ceiling)
            / (1.0 - invalid_release_witness_ceiling),
        )
    return (
        float(min(1.0, reliability_lower)),
        witness_mean_lower,
        effective_sample_size,
    )


def hybrid_reliability_lower_bound(
    identified_release_lower_bound: float,
    shared_target_lower_bound: float,
    *,
    identified_error_rate: float,
    shared_target_error_rate: float,
    total_error_rate: float,
) -> float:
    """Combine lower bounds whose directional error budgets are jointly valid."""

    values = np.asarray(
        [identified_release_lower_bound, shared_target_lower_bound],
        dtype=float,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Reliability lower bounds must lie in [0, 1].")
    error_rates = np.asarray(
        [identified_error_rate, shared_target_error_rate, total_error_rate],
        dtype=float,
    )
    if (
        not np.isfinite(error_rates).all()
        or np.any(error_rates <= 0.0)
        or np.any(error_rates >= 1.0)
    ):
        raise ValueError("Error rates must lie strictly between zero and one.")
    if identified_error_rate + shared_target_error_rate > total_error_rate + 1e-15:
        raise ValueError(
            "The component error rates must sum to at most total_error_rate."
        )
    return float(values.max())
