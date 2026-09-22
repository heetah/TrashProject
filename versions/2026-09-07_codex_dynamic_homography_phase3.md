# Dynamic pseudo-homography Phase 3 safe initial transform

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added a deterministic normalized-image initial H rather than random or
  moving-object point correspondence initialization.
- Added projective-scale normalization without mutating caller matrices.
- Added vectorized `(N,2)` image-to-ground projection with row-level safety
  masks and NaN output for unsafe rows.
- Added complete-image validation covering determinant, condition number,
  homogeneous denominator magnitude/variation/sign, projected bounds,
  orientation and ground-region collapse.
- Added an auditable version-0, confidence-0, relative-scale-only snapshot to
  the opt-in calibration analysis summary.

## Mathematical baseline

For image width `W` and height `H`, the initial relative transform is
`diag(1/W, 1/H, 1)`. It maps the image rectangle to `[0,1]x[0,1]` while
preserving orientation. It does not remove perspective and is not metric.

## Evidence boundary

The snapshot reports `affects_attribution=false`. No candidate optimization,
confidence-gated update, event freeze or backtrack distance migration occurs in
this phase. Validation means numerically/geometrically safe, not accurate.

## Configuration

Matrix determinant, condition number, denominator, output-bound, projected-area
and orientation checks are configurable in `.env.example`.

## Validation

- Phase 1/2/3 synthetic and integration tests: 25 passed.
- Affected detector/litter selection: 78 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 269 passed, 12 skipped, 2 failed. The same
  suite-order detector-render SHA mismatches remain; the characterization
  tests pass inside the isolated affected selection. Golden images were not
  changed.

## Rollback

`DYNAMIC_HOMOGRAPHY=0` remains the default. Removing `transform.py` and the
snapshot field restores Phase-2-only diagnostics without changing attribution.
