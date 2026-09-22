# Dynamic pseudo-homography Phase 2 motion field and flow clustering

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added a configurable spatial traffic motion field.
- Added robust circular direction estimation using a circular medoid followed
  by angular-residual trimming; speed uses a quality-weighted median.
- Added deterministic grid-indexed traffic-flow clustering using normalized
  position, local direction and same-track segment continuity.
- Required independent track support before a flow becomes valid.
- Added non-mutating debug rendering for local segments, cell arrows and flow
  cluster colors.
- Extended the opt-in analysis summary with coverage, flow count, noise count,
  direction variance and flow confidence diagnostics.

## Evidence boundary

Phase 2 still reports `affects_attribution=false`. Flow confidence summarizes
support, quality and direction concentration; it is not classification
probability or calibration accuracy. No homography is estimated or promoted.

## Configuration

Grid, robust trimming, spatial radius, direction angles, continuity bounds and
minimum sample/track support are all exposed in `.env.example`. Defaults are
engineering baselines pending multi-camera calibration data.

## Validation

- Phase 1/2 synthetic and integration tests: 16 passed.
- Affected detector/litter selection: 69 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 260 passed, 12 skipped, 2 failed. As in Phase 1,
  the failures are suite-order detector-render SHA mismatches; the same
  characterization tests pass in the isolated affected selection. Golden
  images were not changed.

## Rollback

`DYNAMIC_HOMOGRAPHY=0` remains the default and avoids all observation work.
Removing `motion_field.py` and its summary calls restores Phase-1-only behavior
without changing litter or attribution logic.
