# 2026-09-14 Codex — release-state marginalization Phase 1A

- Date: 2026-09-14
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `a149c5975898643e5131caf0579bd0aefe228d75`
- Commit: uncommitted working-tree change
- Scope: research-only mathematical primitives

## Problem

The planned release-state and multi-hypothesis research needs numerically sound
likelihood, irregular-time prior and marginal evidence operations before any
prototype can be compared with production Smart Backtrack. Existing production
trajectory/cost/flow code constructs and selects routes; it intentionally does
not provide a probability model and must not be modified for this phase.

## Change

Added `scripts/pipeline/backtrack/release_marginalization.py` with no production
caller:

- `gaussian_nll` implements multivariate Gaussian negative log likelihood by
  Cholesky factorization and solve. It rejects empty/dimension-mismatched,
  non-finite, non-symmetric and non-positive-definite inputs. It adds no jitter.
- `release_time_prior_masses` merges duplicate seconds, forms midpoint/Voronoi
  cells inside caller-supplied outer boundaries, integrates caller-supplied
  densities and normalizes once. Conflicting densities at the same time fail
  closed. `uniform_prior_density` is an explicit test/research helper.
- `marginalize_route_hypotheses` performs stable log-sum-exp first within each
  route and then across routes. It requires unique auditable hypothesis IDs,
  normalized conditional hypothesis priors, normalized route priors, and a
  caller-specified NULL route with hypotheses. It returns evidence/posteriors
  for every route without selecting one.
- Normalization validation works in probability space and derives its float64
  forward-error tolerance as `gamma_(N+1)=((N+1)u)/(1-(N+1)u)`, with
  `u=ulp(1.0)/2`, for N exponentiations plus the final accurately rounded sum.
  A singleton prior requires exact `log(1)=0`; there is no model tolerance.
- `log_likelihood=-inf` and zero prior mass represent mathematically valid zero
  evidence. Individual routes may therefore receive posterior zero; an all-zero
  route evidence set fails explicitly instead of producing NaN.

## Mathematical contract

For residual `r` in units `U` and covariance `Sigma` in `U^2`:

```text
NLL = 1/2 [d log(2*pi) + log det(Sigma) + r^T Sigma^-1 r]
```

For unique release times and caller boundaries `b_i`:

```text
q_i = density(t_i) * (b_(i+1) - b_i)
p_i = q_i / sum_j q_j
```

For hypothesis `h` belonging to route `r`:

```text
log E_r = logsumexp_h(log L_h + log p(h|r))
log J_r = log p(r) + log E_r
p(r|evidence) = exp(log J_r - logsumexp_k(log J_k))
```

The returned posterior is conditional on caller-supplied model assumptions; it
is neither a calibrated correctness probability nor an accuracy estimate.

## Interface and configuration

No production API, environment variable, resolver behavior, route schema,
cost, gate, release window, bandwidth or default was changed. All model inputs
and boundaries are caller supplied. `history_lineage` is deliberately not read:
dependence between observations has not yet been modeled, so the module cannot
automatically multiply per-observation likelihoods.

## Test evidence

Focused tests cover known one/two-dimensional Gaussian values; SPD, symmetry,
finite and dimension failures; prior normalization; duplicate-grid invariance;
constant-likelihood grid-refinement invariance; extreme negative logs; route
posterior normalization; NULL preservation; duplicate hypothesis IDs; missing
routes/hypotheses; and non-normalized prior rejection.

- Focused Phase 1A tests: 27 passed.
- Full `tests/pipeline`: 429 passed, 12 skipped.
- `git diff --check`: passed.
- Search confirmed no Python file under `scripts/` imports the new module.

## Limitations

- No likelihood model for identity, visibility, lineage dependence or release
  dynamics is defined yet.
- No camera-grouped calibration or reviewed 58-case attribution comparison has
  been run for these primitives.
- A posterior can be numerically correct while its supplied priors or likelihood
  model are scientifically wrong.
- The normalization bound assumes the standard floating-point model for the
  runtime `exp` and `fsum`; Python does not promise a cross-platform,
  correctly-rounded formal guarantee for every system libm implementation.
- Production integration and top-1 route selection are explicitly out of scope.

## Rollback

Remove the standalone module, its focused test file and these documentation
sections. No production rollback, output migration or configuration change is
required because the module has no production caller.
