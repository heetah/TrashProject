# Dynamic pseudo-homography Phase 7 persistent state machine

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added `DynamicHomographyCalibrator` with the required collection, update,
  projection, H/state/confidence access and reset API.
- Added UNCALIBRATED, COLLECTING, ESTIMATING, WARMING_UP, LOCKED,
  LOW_CONFIDENCE, DRIFT_DETECTED and RECALIBRATING states.
- Added interval-based evaluation, stable-window locking and persistent rather
  than single-window drift transitions.
- Added residual, normalized traffic centroid/spread, direction-tensor and
  optional background-motion drift signals.
- Added bounded rollback snapshots and the required per-window calibration
  metrics without retaining video frames.
- Integrated the calibrator into the existing YOLO-Seg actor-observation path
  while retaining a compatibility collector alias.

## State and safety behavior

Successful Phase-6 proposals update only the isolated calibrator H and increment
its version. LOCKED state monitors evidence without continuously optimizing.
Low evidence and invalid/non-improving proposals freeze the previous H. Drift
requires consecutive abnormal windows. Image-shape changes freeze and require
an explicit reset so observations from incompatible coordinate systems cannot
be combined.

The state remains marked `affects_attribution=false`; backtrack, event
confirmation and NULL routing do not read this H in Phase 7.

## Configuration

Evaluation interval, lock counts, drift persistence and thresholds, residual
EMA, rollback depth and metric history are all exposed in `.env.example`.
Defaults are conservative engineering priors pending reviewed camera-grouped
replay.

## Validation

- Phase 1-7 calibration tests: 48 passed.
- Affected pipeline selection: 171 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 292 passed, 12 skipped, 2 failed. The two known
  suite-order detector-render SHA mismatches remain in scenarios A/B; their
  characterization module passes in isolation, and no golden was changed.

## Rollback

Set `DYNAMIC_HOMOGRAPHY=0` to disable collection and state evaluation. Runtime
`rollback()` restores the most recent isolated H snapshot; `reset()` clears the
camera calibration state and lightweight evidence.
