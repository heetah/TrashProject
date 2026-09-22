"""Research-only primitives for release-state hypothesis marginalization.

This module has no production caller.  It supplies strict numerical building
blocks only: it does not construct hypotheses from tracker history, select a
route, modify NULL behavior, or claim calibrated probabilities/accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


def gaussian_nll(residual: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    """Return ``-log N(residual; 0, covariance)`` using Cholesky solves.

    Residual components and covariance axes must describe the same quantity and
    units; for example, a pixel residual requires covariance in pixels squared.
    No jitter or covariance repair is applied.
    """

    vector = np.asarray(residual, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("residual must be a non-empty one-dimensional vector")
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("covariance shape must match residual dimension")
    if not np.isfinite(vector).all() or not np.isfinite(matrix).all():
        raise ValueError("residual and covariance must be finite")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("covariance must be exactly symmetric")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    whitened = np.linalg.solve(factor, vector)
    log_determinant = 2.0 * float(np.log(np.diag(factor)).sum())
    return 0.5 * (
        vector.size * math.log(2.0 * math.pi)
        + log_determinant
        + float(whitened @ whitened)
    )


@dataclass(frozen=True)
class ReleasePriorMass:
    """Prior mass assigned to one release-time grid point."""

    release_time_seconds: float
    cell_start_seconds: float
    cell_end_seconds: float
    density: float
    mass: float
    log_mass: float


def uniform_prior_density(times_seconds: Sequence[float]) -> np.ndarray:
    """Return an explicit constant density for caller-selected time points."""

    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("times_seconds must be a non-empty finite vector")
    return np.ones(times.shape, dtype=np.float64)


def release_time_prior_masses(
    times_seconds: Sequence[float],
    prior_density: Sequence[float],
    *,
    domain_start_seconds: float,
    domain_end_seconds: float,
) -> tuple[ReleasePriorMass, ...]:
    """Discretize a caller-supplied density with midpoint cell boundaries.

    For sorted unique grid points ``t_i``, interior boundaries are
    ``b_i=(t_(i-1)+t_i)/2``.  The caller supplies the two outer domain
    boundaries.  Point mass is ``density_i * (b_(i+1)-b_i)`` followed by one
    global normalization. Duplicate times are merged before cells are built and
    must carry the same density value.
    """

    times = np.asarray(times_seconds, dtype=np.float64)
    density = np.asarray(prior_density, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_seconds must be a non-empty one-dimensional vector")
    if density.shape != times.shape:
        raise ValueError("prior_density must have one value per time point")
    if not np.isfinite(times).all() or not np.isfinite(density).all():
        raise ValueError("times and prior density must be finite")
    if np.any(density < 0.0):
        raise ValueError("prior density must be non-negative")
    start = float(domain_start_seconds)
    end = float(domain_end_seconds)
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("prior domain must be finite with start < end")

    merged: dict[float, float] = {}
    for time_value, density_value in zip(times.tolist(), density.tolist()):
        if time_value in merged and merged[time_value] != density_value:
            raise ValueError("duplicate times must have identical density")
        merged[time_value] = density_value
    unique_times = np.asarray(sorted(merged), dtype=np.float64)
    unique_density = np.asarray(
        [merged[time_value] for time_value in unique_times.tolist()],
        dtype=np.float64,
    )
    if unique_times[0] < start or unique_times[-1] > end:
        raise ValueError("all grid times must lie within the supplied domain")

    boundaries = np.empty(unique_times.size + 1, dtype=np.float64)
    boundaries[0] = start
    boundaries[-1] = end
    if unique_times.size > 1:
        boundaries[1:-1] = 0.5 * (unique_times[:-1] + unique_times[1:])
    widths = np.diff(boundaries)
    if np.any(widths <= 0.0):
        raise ValueError("time grid cells must have positive width")
    unnormalized = unique_density * widths
    total = float(unnormalized.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("prior density must induce positive finite total mass")
    masses = unnormalized / total
    return tuple(
        ReleasePriorMass(
            release_time_seconds=float(time_value),
            cell_start_seconds=float(boundaries[index]),
            cell_end_seconds=float(boundaries[index + 1]),
            density=float(unique_density[index]),
            mass=float(mass),
            log_mass=(math.log(float(mass)) if mass > 0.0 else -math.inf),
        )
        for index, (time_value, mass) in enumerate(zip(unique_times, masses))
    )


@dataclass(frozen=True)
class HypothesisEvidence:
    """One auditable state/identity/visibility/release hypothesis."""

    hypothesis_id: str
    route_id: str
    release_time_seconds: float
    log_likelihood: float
    log_prior_mass: float

    def validate(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if not isinstance(self.route_id, str) or not self.route_id:
            raise ValueError("route_id is required")
        if not math.isfinite(float(self.release_time_seconds)):
            raise ValueError("release_time_seconds must be finite")
        likelihood = float(self.log_likelihood)
        if math.isnan(likelihood) or likelihood == math.inf:
            raise ValueError(
                "log_likelihood must be finite or negative infinity"
            )
        prior = float(self.log_prior_mass)
        if math.isnan(prior) or prior == math.inf:
            raise ValueError("log_prior_mass must be finite or negative infinity")


@dataclass(frozen=True)
class RouteMarginalEvidence:
    route_id: str
    hypothesis_ids: tuple[str, ...]
    conditional_log_evidence: float
    route_log_prior: float
    joint_log_evidence: float
    posterior: float
    is_null: bool


@dataclass(frozen=True)
class MarginalEvidence:
    routes: tuple[RouteMarginalEvidence, ...]
    total_log_evidence: float


def _logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.isnan(array).any():
        raise ValueError("log-sum-exp requires a non-empty vector without NaN")
    maximum = float(np.max(array))
    if maximum == -math.inf:
        return -math.inf
    if maximum == math.inf:
        raise ValueError("positive infinite log weight is invalid")
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def _require_normalized_log_masses(values: Sequence[float], label: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.isnan(array).any():
        raise ValueError(f"{label} must be a non-empty vector without NaN")
    if np.any(array == math.inf) or np.any(array > 0.0):
        raise ValueError(f"{label} cannot contain positive log mass")
    if array.size == 1:
        # A singleton normalized mass is exactly log(1) == 0 in float64; no
        # accumulation or transcendental roundoff allowance is needed.
        if float(array[0]) != 0.0:
            raise ValueError(f"{label} must sum to one in probability space")
        return

    probability_total = math.fsum(math.exp(float(value)) for value in array)
    # Standard floating-point model: unit roundoff u = ulp(1)/2. Allow one
    # relative-rounding contribution for each exp evaluation plus the final
    # accurately rounded fsum result. Their forward-error bound is
    # gamma_k = k*u/(1-k*u), where k=N+1. This is input-size-derived numerical
    # allowance, not a probability/model calibration threshold.
    unit_roundoff = math.ulp(1.0) / 2.0
    operation_count = int(array.size) + 1
    accumulated_roundoff = operation_count * unit_roundoff
    if accumulated_roundoff >= 1.0:
        raise ValueError(f"{label} has too many terms for a finite error bound")
    tolerance = accumulated_roundoff / (1.0 - accumulated_roundoff)
    if not math.isclose(
        probability_total, 1.0, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(f"{label} must sum to one in probability space")


def marginalize_route_hypotheses(
    hypotheses: Sequence[HypothesisEvidence],
    route_log_priors: Mapping[str, float],
    *,
    null_route_id: str,
) -> MarginalEvidence:
    """Marginalize supplied hypotheses within routes, then across routes.

    Hypothesis log prior masses must normalize within each route, and route log
    priors must normalize across routes. The function does not multiply history
    observations, infer lineage independence, or select a top-1 route.
    """

    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    if not route_log_priors:
        raise ValueError("route_log_priors are required")
    if not isinstance(null_route_id, str):
        raise ValueError("null_route_id must be a string")
    if any(
        not isinstance(route_id, str) or not route_id
        for route_id in route_log_priors
    ):
        raise ValueError("route prior IDs must be non-empty strings")
    if not null_route_id or null_route_id not in route_log_priors:
        raise ValueError("the caller-specified NULL route must have a prior")

    grouped: dict[str, list[HypothesisEvidence]] = {}
    seen_ids = set()
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, HypothesisEvidence):
            raise ValueError("every hypothesis must be HypothesisEvidence")
        hypothesis.validate()
        if hypothesis.hypothesis_id in seen_ids:
            raise ValueError("hypothesis_id values must be globally unique")
        seen_ids.add(hypothesis.hypothesis_id)
        if hypothesis.route_id not in route_log_priors:
            raise ValueError("every hypothesis route must have a route prior")
        grouped.setdefault(hypothesis.route_id, []).append(hypothesis)

    prior_routes = set(route_log_priors)
    if prior_routes != set(grouped):
        raise ValueError("every prior route must have at least one hypothesis")
    route_prior_values = []
    for route_id in sorted(prior_routes):
        value = float(route_log_priors[route_id])
        if math.isnan(value) or value == math.inf:
            raise ValueError("route log priors must be finite or negative infinity")
        route_prior_values.append(value)
    _require_normalized_log_masses(route_prior_values, "route priors")

    intermediate = []
    for route_id in sorted(grouped):
        route_hypotheses = grouped[route_id]
        hypothesis_log_priors = [
            float(item.log_prior_mass) for item in route_hypotheses
        ]
        _require_normalized_log_masses(
            hypothesis_log_priors, f"hypothesis priors for route {route_id}"
        )
        conditional = _logsumexp([
            float(item.log_likelihood) + float(item.log_prior_mass)
            for item in route_hypotheses
        ])
        route_prior = float(route_log_priors[route_id])
        intermediate.append((
            route_id,
            tuple(item.hypothesis_id for item in route_hypotheses),
            conditional,
            route_prior,
            conditional + route_prior,
        ))
    total = _logsumexp([item[4] for item in intermediate])
    if total == -math.inf:
        raise ValueError("total route evidence is zero")
    routes = tuple(
        RouteMarginalEvidence(
            route_id=route_id,
            hypothesis_ids=hypothesis_ids,
            conditional_log_evidence=conditional,
            route_log_prior=route_prior,
            joint_log_evidence=joint,
            posterior=float(math.exp(joint - total)),
            is_null=route_id == null_route_id,
        )
        for route_id, hypothesis_ids, conditional, route_prior, joint
        in intermediate
    )
    return MarginalEvidence(routes=routes, total_log_evidence=total)


__all__ = [
    "HypothesisEvidence",
    "MarginalEvidence",
    "ReleasePriorMass",
    "RouteMarginalEvidence",
    "gaussian_nll",
    "marginalize_route_hypotheses",
    "release_time_prior_masses",
    "uniform_prior_density",
]
