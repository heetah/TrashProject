# Dynamic pseudo-homography Phase 8 static-background stabilization

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added optional ORB static-background current-to-reference stabilization.
- Required explicit dynamic-object exclusion masks and added a helper that
  combines dense masks, polygons and boxes for vehicle/person/litter regions.
- Added ratio filtering, RANSAC, inlier support and bounded translation,
  rotation, scale and orientation validation.
- Added `H_runtime=H_calibration@G_current_to_reference` composition while
  freezing the last valid G after transient failures.
- Added immutable event-level H/G/runtime-H snapshots and a reserved
  `meters_per_unit` field for future external metric calibration.
- Added complete copy-only debug rendering for image tracks, ground points,
  flows, pseudo-ground tracks and calibration statistics.

## Memory and runtime

The stabilizer stores one reference keypoint/descriptor set and one 3x3 G, not
a video or frame history. ORB runs only when explicitly called. Long-term
calibration continues to store bounded lightweight trajectory metadata and
evaluate by configurable windows rather than every frame.

## Safety

`DYNAMIC_HOMOGRAPHY_STABILIZE=0` remains the default. Missing exclusion masks,
weak matches or excessive motion produce an invalid diagnostic result and no
new G. Phase 8 still reports `affects_attribution=false`; the snapshot API is
not consumed by production backtracking yet.

## Validation

- Phase 1-8 calibration unit/integration/synthetic tests: 66 passed.
- Full production pipeline suite: 312 passed, 12 skipped.
- Required production-module compilation and `git diff --check`: passed.
- Repository-wide suite: 337 passed, 12 skipped, 2 failed. Both failures are
  pre-existing litter-confirmation policy assertions in
  `tests/test_litter_regression.py` (Case 10 and Case 115/116), outside this
  isolated calibration change; no attribution or event-confirmation threshold
  was changed to hide them.
- Static audit found no moving-vehicle previous/current correspondence solve in
  `scripts/pipeline/calibration/`. The only transform estimator is the optional
  background ORB/RANSAC partial affine, which receives a required dynamic-object
  exclusion mask.
- Synthetic coverage includes straight convergence with increasing persistent
  confidence and decreasing update magnitude, smooth curvature, multiple flows,
  sparse-evidence freeze, segmentation jitter, ID-switch/teleport rejection,
  persistent camera-drift recalibration, and curved-road lane-change downweighting.

## Rollback

Keep `DYNAMIC_HOMOGRAPHY_STABILIZE=0` or call `StaticBackgroundStabilizer.reset()`.
Removing `stabilization.py`, `debug.py` and their optional state APIs restores
Phase-7-only behavior.
