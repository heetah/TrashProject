# Dynamic pseudo-homography Phase 6 bounded optimization

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added bounded deterministic candidate-H coordinate search using five
  dimensionless pseudo-ground perturbation parameters.
- Added robust local-Jacobian perspective regularization and normalized
  previous-H temporal displacement prior.
- Added a ten-component auditable confidence score plus independent hard
  evidence gates for tracks, flows, coverage, duration and improvement.
- Added theta-space confidence-gated small-step proposals with post-update
  geometric validation and objective re-scoring.
- Added explicit freeze reasons, candidate counts, selected parameters,
  improvement, confidence components and proposed matrices to diagnostics.

## Parameterization

`H_candidate=P(theta)H_previous`, where theta contains log anisotropy, x/y
shear and x/y perspective terms. Translation and rotation are not searched
because the current traffic consistency losses cannot identify them. Searching
bounded parameters avoids unconstrained singular or exploding matrices.

## Confidence and safe update

Confidence is the mean of separately reported support, coverage, quality,
stability, residual and improvement components. It does not override hard
evidence gates. If all gates pass:

`alpha=max_update_alpha*confidence*improvement_score`.

The optimizer applies `alpha*theta` to the previous H, validates the result and
requires its complete objective to remain better than baseline. Otherwise the
previous matrix is returned unchanged with a freeze reason.

## Evidence boundary

All default search steps, objective weights, confidence targets and gates are
explicit initial engineering priors. Synthetic tests establish algorithmic
behavior, not real-camera calibration accuracy. Reviewed camera-grouped replay
is required before tuning or promotion.

## Safety

`DYNAMIC_HOMOGRAPHY_OPTIMIZE=0` is the default. Enabling it only emits a
proposal with `applied_to_production=false` and `affects_attribution=false`.
There is no persistent H update or backtrack integration in Phase 6.

## Validation

- Phase 1-6 calibration tests: 41 passed.
- Affected pipeline selection: 164 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 285 passed, 12 skipped, 2 failed. The two known
  suite-order detector-render SHA mismatches remain in scenarios A/B; their
  characterization module passes in isolation, and no golden was changed.

## Rollback

Keep `DYNAMIC_HOMOGRAPHY_OPTIMIZE=0`, or remove `optimizer.py` and the optional
`optimization` summary field to return to Phase-5-only diagnostics.
