# Dynamic pseudo-homography Phase 1 observation layer

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Problem

The legacy BEV experiment accumulates vehicle bbox bottom centres and constructs
a simple projection. A scalable self-calibrator instead needs robust mask
contact points and long-term tracked local-motion observations before any H
optimization is allowed to influence attribution.

## Change

- Added a standalone calibration package and Phase-1 configuration.
- Added robust 98-percentile bottom-band median extraction for dense masks and
  Ultralytics segmentation polygons.
- Added explicit rejection for cached observations, small/invalid masks, image
  borders, low confidence, non-monotonic tracks and teleport motion.
- Added bounded rolling per-track metadata and curved local motion segments.
- Connected observed vehicle/scooter masks to the collector through
  `GlobalLitterTracker`.
- Added opt-in analysis summary with `affects_attribution=false`.

## Interface

Set `DYNAMIC_HOMOGRAPHY=1` to collect evidence. All thresholds and memory/window
limits are listed in `.env.example`. The default is disabled.

## Evidence boundary

This phase does not estimate or update H. It therefore cannot improve or reduce
attribution accuracy. Its output measures data availability and quality only;
it is not ground truth.

## Validation

- Phase-1 unit/integration tests: 9 passed.
- Affected detector/litter regression selection: 62 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 253 passed, 12 skipped, 2 failed. The two failures
  are detector-render SHA mismatches that pass when the same characterization
  tests run independently, indicating pre-existing suite-order/global-state
  contamination outside the Phase-1 observation path. Golden images were not
  rewritten.

## Rollback

Leave `DYNAMIC_HOMOGRAPHY=0` (default), or remove the calibration package and
the observation hook without changing legacy litter/backtrack behavior.
