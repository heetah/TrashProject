# Dynamic Homography event integration and 58-clip validation

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs / validation

## Problem

Phase 1-8 produced a validated calibration state and event snapshot API, but
Smart Backtrack did not consume it. Product readiness also needed a fixed
58-clip denominator so missed events, NULL routes and failures could not be
removed from the reported accuracy.

## Change

- Added event-frozen pseudo-ground projection helpers.
- Wired the same eligible snapshot into C_BA, C_BC and C_AC spatial features.
- Kept Kalman/RTS and time evidence in image/frame coordinates.
- Required `LOCKED`, confidence at least 0.65 and matrix validation; every
  ineligible event records an image-space fallback reason and retains NULL.
- Added full-denominator readiness reporting and nested batch-output support.
- Made malformed joined metrics fail closed: no confirmed event plus emitted
  vehicle route can never count as end-to-end correct.

## Runtime validation

- 58/58 reviewed usable positive clips completed; zero process failures.
- 58 analysis files, sidecars, logs and annotated MP4 files were produced.
- All 58 MP4 files passed `ffprobe` video-stream decoding.
- 62 event snapshots were evaluated; 0 applied H and 62 safely fell back
  because calibration never reached `LOCKED` in isolated short clips.
- Baseline and opt-in batch clip/event metric CSV files are byte-identical.
- Event sensitivity: 41/58 = 70.69%.
- End-to-end correct vehicle: 33/58 = 56.90%.
- Conditional correct vehicle given a confirmed event: 33/41 = 80.49%.
- The 85% point gate requires 50/58, leaving a 17-clip end-to-end gap.
- False-positive release gate remains blocked because no reviewed negative set
  was supplied. Extra event records are warnings, not verified false positives.

This is fallback/no-regression evidence, not proof that a learned H improves
real attribution. Continuous per-camera footage is required to reach `LOCKED`
and perform a paired applied-H comparison.

## Tests

- `tests/pipeline`: 319 passed, 12 skipped.
- Production compilation and `git diff --check`: passed.
- Whole repository: 344 passed, 12 skipped, 2 failed. The two pre-existing
  tracker-only Case 10 and Case 115/116 expectations conflict with the reviewed
  8/27 two-observation recovery profile and remain unresolved; no threshold was
  changed to hide them.

## Rollback

Keep `SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=0` (default), or remove the spatial
transform argument from resolver cost calls. Calibration collection and all
legacy image-space attribution remain available independently.

## Subsequent metric correction (2026-09-07)

The 41/58, 33/58 and 33/41 figures above used a numeric actor-ID table copied
from a different inference run. Tracker IDs are run-local, so those values must
not be used as product-readiness accuracy. The homography fallback/no-change
finding remains valid because the compared files were byte-identical.

The replacement protocol maps manual actor boxes to tracklets from the exact
evaluated run and accepts only strict/moderate event matches. Its baseline is
32/58 accepted events and a provisional 23/58 exact end-to-end routes. See
`versions/2026-09-07_codex_same_run_readiness_protocol.md`; the labels remain
unreviewed and the negative/cross-camera gates remain blocked.
