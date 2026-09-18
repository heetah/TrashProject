import math

import numpy as np
import pytest

from pipeline.backtrack.release_marginalization import (
    HypothesisEvidence,
    gaussian_nll,
    marginalize_route_hypotheses,
    release_time_prior_masses,
    uniform_prior_density,
)


def test_gaussian_nll_known_one_dimensional_value():
    assert gaussian_nll([0.0], [[4.0]]) == pytest.approx(
        0.5 * math.log(2.0 * math.pi * 4.0)
    )


def test_gaussian_nll_known_two_dimensional_value():
    expected = 0.5 * (
        2.0 * math.log(2.0 * math.pi) + math.log(4.0) + 2.0
    )
    assert gaussian_nll([1.0, 2.0], [[1.0, 0.0], [0.0, 4.0]]) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "residual,covariance,match",
    [
        ([1.0], [[1.0, 0.0], [0.0, 1.0]], "shape"),
        ([1.0, 1.0], [[1.0, 0.1], [0.0, 1.0]], "symmetric"),
        ([1.0, 1.0], [[1.0, 2.0], [2.0, 1.0]], "positive definite"),
        ([math.nan], [[1.0]], "finite"),
        ([1.0], [[math.nan]], "finite"),
    ],
)
def test_gaussian_nll_rejects_invalid_input(residual, covariance, match):
    with pytest.raises(ValueError, match=match):
        gaussian_nll(residual, covariance)


def test_irregular_prior_masses_sum_to_one_with_explicit_boundaries():
    rows = release_time_prior_masses(
        [0.0, 1.0, 3.0],
        [1.0, 2.0, 1.0],
        domain_start_seconds=-0.5,
        domain_end_seconds=4.0,
    )
    assert sum(row.mass for row in rows) == pytest.approx(1.0)
    assert [(row.cell_start_seconds, row.cell_end_seconds) for row in rows] == [
        (-0.5, 0.5), (0.5, 2.0), (2.0, 4.0)
    ]


def test_duplicate_time_is_merged_without_adding_prior_mass():
    unique = release_time_prior_masses(
        [0.0, 1.0], [1.0, 1.0],
        domain_start_seconds=-0.5, domain_end_seconds=1.5,
    )
    duplicate = release_time_prior_masses(
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        domain_start_seconds=-0.5, domain_end_seconds=1.5,
    )
    assert [(row.release_time_seconds, row.mass) for row in duplicate] == pytest.approx(
        [(row.release_time_seconds, row.mass) for row in unique]
    )


def _one_route_evidence(times):
    prior = release_time_prior_masses(
        times,
        uniform_prior_density(times),
        domain_start_seconds=0.0,
        domain_end_seconds=2.0,
    )
    hypotheses = [
        HypothesisEvidence(
            hypothesis_id=f"h{index}",
            route_id="NULL",
            release_time_seconds=row.release_time_seconds,
            log_likelihood=-7.0,
            log_prior_mass=row.log_mass,
        )
        for index, row in enumerate(prior)
    ]
    return marginalize_route_hypotheses(
        hypotheses, {"NULL": 0.0}, null_route_id="NULL"
    )


def test_grid_refinement_does_not_change_constant_likelihood_evidence():
    coarse = _one_route_evidence([0.5, 1.5])
    fine = _one_route_evidence([0.25, 0.75, 1.25, 1.75])
    assert coarse.total_log_evidence == pytest.approx(-7.0)
    assert fine.total_log_evidence == pytest.approx(coarse.total_log_evidence)


def test_log_sum_exp_is_stable_and_route_posteriors_sum_to_one():
    half = math.log(0.5)
    hypotheses = [
        HypothesisEvidence("a0", "actor", 0.0, -10_000.0, half),
        HypothesisEvidence("a1", "actor", 1.0, -10_001.0, half),
        HypothesisEvidence("n0", "NULL", 0.0, -10_002.0, 0.0),
    ]
    result = marginalize_route_hypotheses(
        hypotheses,
        {"actor": half, "NULL": half},
        null_route_id="NULL",
    )
    assert math.isfinite(result.total_log_evidence)
    assert sum(route.posterior for route in result.routes) == pytest.approx(1.0)
    assert any(route.is_null and route.route_id == "NULL" for route in result.routes)


def test_zero_likelihood_route_has_zero_posterior_when_total_is_finite():
    result = marginalize_route_hypotheses(
        [
            HypothesisEvidence("a0", "actor", 0.0, -1.0, 0.0),
            HypothesisEvidence("n0", "NULL", 0.0, -math.inf, 0.0),
        ],
        {"actor": math.log(0.5), "NULL": math.log(0.5)},
        null_route_id="NULL",
    )
    by_route = {route.route_id: route for route in result.routes}
    assert by_route["NULL"].conditional_log_evidence == -math.inf
    assert by_route["NULL"].posterior == 0.0
    assert by_route["actor"].posterior == pytest.approx(1.0)


def test_zero_route_prior_has_zero_posterior_when_total_is_finite():
    result = marginalize_route_hypotheses(
        [
            HypothesisEvidence("a0", "actor", 0.0, -1.0, 0.0),
            HypothesisEvidence("n0", "NULL", 0.0, -1.0, 0.0),
        ],
        {"actor": 0.0, "NULL": -math.inf},
        null_route_id="NULL",
    )
    by_route = {route.route_id: route for route in result.routes}
    assert by_route["NULL"].joint_log_evidence == -math.inf
    assert by_route["NULL"].posterior == 0.0
    assert by_route["actor"].posterior == pytest.approx(1.0)


def test_all_zero_route_evidence_fails_instead_of_returning_nan():
    with pytest.raises(ValueError, match="total route evidence is zero"):
        marginalize_route_hypotheses(
            [
                HypothesisEvidence("a0", "actor", 0.0, -math.inf, 0.0),
                HypothesisEvidence("n0", "NULL", 0.0, -math.inf, 0.0),
            ],
            {"actor": math.log(0.5), "NULL": math.log(0.5)},
            null_route_id="NULL",
        )


@pytest.mark.parametrize("invalid_log_likelihood", [math.inf, math.nan])
def test_positive_infinite_or_nan_likelihood_fails_closed(invalid_log_likelihood):
    with pytest.raises(ValueError, match="log_likelihood"):
        marginalize_route_hypotheses(
            [
                HypothesisEvidence(
                    "n0", "NULL", 0.0, invalid_log_likelihood, 0.0
                )
            ],
            {"NULL": 0.0},
            null_route_id="NULL",
        )


def test_duplicate_hypothesis_id_fails_closed():
    rows = [
        HypothesisEvidence("same", "actor", 0.0, -1.0, 0.0),
        HypothesisEvidence("same", "NULL", 0.0, -1.0, 0.0),
    ]
    with pytest.raises(ValueError, match="globally unique"):
        marginalize_route_hypotheses(
            rows, {"actor": math.log(0.5), "NULL": math.log(0.5)},
            null_route_id="NULL",
        )


@pytest.mark.parametrize(
    "hypotheses,route_priors,null_route,match",
    [
        ([], {"NULL": 0.0}, "NULL", "hypothesis"),
        ([HypothesisEvidence("h", "", 0.0, -1.0, 0.0)], {"NULL": 0.0}, "NULL", "route_id"),
        ([HypothesisEvidence("h", "actor", 0.0, -1.0, 0.0)], {"NULL": 0.0}, "NULL", "route prior"),
        ([HypothesisEvidence("h", "actor", 0.0, -1.0, 0.0)], {"actor": 0.0}, "NULL", "NULL route"),
    ],
)
def test_missing_hypothesis_or_route_fails_closed(
    hypotheses, route_priors, null_route, match
):
    with pytest.raises(ValueError, match=match):
        marginalize_route_hypotheses(
            hypotheses, route_priors, null_route_id=null_route
        )


def test_prior_rejects_conflicting_duplicate_density():
    with pytest.raises(ValueError, match="identical density"):
        release_time_prior_masses(
            [0.5, 0.5], [1.0, 2.0],
            domain_start_seconds=0.0, domain_end_seconds=1.0,
        )


def test_hypothesis_prior_masses_must_be_normalized_within_route():
    hypotheses = [
        HypothesisEvidence("a0", "actor", 0.0, -1.0, math.log(0.8)),
        HypothesisEvidence("a1", "actor", 1.0, -1.0, math.log(0.8)),
        HypothesisEvidence("n0", "NULL", 0.0, -1.0, 0.0),
    ]
    with pytest.raises(ValueError, match="hypothesis priors"):
        marginalize_route_hypotheses(
            hypotheses,
            {"actor": math.log(0.5), "NULL": math.log(0.5)},
            null_route_id="NULL",
        )


def test_route_prior_masses_must_be_normalized():
    hypotheses = [
        HypothesisEvidence("a0", "actor", 0.0, -1.0, 0.0),
        HypothesisEvidence("n0", "NULL", 0.0, -1.0, 0.0),
    ]
    with pytest.raises(ValueError, match="route priors"):
        marginalize_route_hypotheses(
            hypotheses,
            {"actor": math.log(0.8), "NULL": math.log(0.8)},
            null_route_id="NULL",
        )


def _equal_mass_hypotheses(count, scale=1.0):
    masses = np.full(count, scale / count, dtype=np.float64)
    return [
        HypothesisEvidence(
            f"h{index}", "NULL", float(index), -1.0, math.log(float(mass))
        )
        for index, mass in enumerate(masses)
    ]


def test_normalization_accepts_roundoff_with_derived_gamma_bound():
    count = 10
    # Under the standard model, N exp roundings plus the final fsum rounding
    # are bounded by gamma_(N+1). Ten rounded 0.1 masses have a nonzero ULP-
    # scale residual, so this exercises the allowance rather than exact zero.
    unit_roundoff = math.ulp(1.0) / 2.0
    operations = count + 1
    gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    scale = 1.0 + gamma / 2.0
    hypotheses = _equal_mass_hypotheses(count, scale)
    assert math.fsum(math.exp(row.log_prior_mass) for row in hypotheses) != 1.0
    result = marginalize_route_hypotheses(
        hypotheses, {"NULL": 0.0}, null_route_id="NULL"
    )
    assert result.routes[0].posterior == pytest.approx(1.0)


def test_normalization_rejects_error_beyond_derived_gamma_bound():
    count = 10
    unit_roundoff = math.ulp(1.0) / 2.0
    operations = count + 1
    gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    hypotheses = _equal_mass_hypotheses(count, 1.0 + 2.0 * gamma)
    with pytest.raises(ValueError, match="hypothesis priors"):
        marginalize_route_hypotheses(
            hypotheses, {"NULL": 0.0}, null_route_id="NULL"
        )


def test_singleton_prior_requires_exact_log_one_without_roundoff_allowance():
    hypotheses = [
        HypothesisEvidence(
            "h0", "NULL", 0.0, -1.0, -math.ulp(1.0)
        )
    ]
    with pytest.raises(ValueError, match="hypothesis priors"):
        marginalize_route_hypotheses(
            hypotheses, {"NULL": 0.0}, null_route_id="NULL"
        )
