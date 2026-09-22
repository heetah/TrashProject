# Litter horizontal-streak probation research candidate

- Date: 2026-09-12
- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree based on `a149c5975898643e5131caf0579bd0aefe228d75`
- Type: fix / test / research validation

## Problem

The frozen 58-positive baseline missed 17 annotated events. Same-run candidate
tracing localized six misses to the horizontal-streak prefilter: early rising
observations were rejected before a ballistic apex/descent could be observed.

## Implementation

- Add an opt-in minimum observation count before horizontal-streak rejection.
- Bound deferred observations by bbox-scale- and source-frame-normalized centroid
  displacement, preventing large jumps from bridging different objects.
- Record both research controls in the litter-candidate sidecar run record.
- Preserve production defaults (`minimum observations=2`, dedup disabled).
- Extend the postprocess calibrator to discover isolated per-case sidecars
  recursively.

## Config and schema

- `LITTER_FP_STREAK_MIN_OBSERVATIONS` defaults to `2`.
- `LITTER_FP_STREAK_DEFER_MAX_STEP_DIAGONALS_PER_FRAME` defaults to `1.1` and is
  consequential only when the observation minimum is greater than `2`.
- Litter candidate run sidecars add `fp_streak_min_observations` and
  `fp_streak_defer_max_step_diagonals_per_frame`.

## Evidence

Final research configuration: observation minimum 6, normalized jump maximum
1.1, same-frame IoU dedup enabled at 0.5.

- Baseline accepted events: 36/58; candidate: 41/58.
- Baseline provisional end-to-end correct routes: 27/58; candidate: 31/58.
- Paired correctness gains/losses: 4/0.
- Paired event-match gains/losses: 5/0.
- Exact two-sided paired p-value: 0.125; not statistically significant.
- Confirmed candidate records: 65 baseline, 64 candidate.
- Dynamic pseudo-homography applied: 0/64 candidate records; gains are not
  attributable to depth/world-coordinate scoring.
- Final batch: 58/58 completed; 58/58 MP4 streams probe successfully.
- Compile smoke passed; complete pipeline suite: 338 passed, 12 skipped.

Complete evidence is under `artifacts/missed_event_triage_20260912/final/`.

## Limitations

The 1.1 value was selected on development cases. Event annotations are
unreviewed, the set is positive-only, and camera-group holdout and reviewed
negative/OCR truth are absent. The candidate is not production-promoted and does
not establish 85% enforcement accuracy.

## Rollback

Keep the default `LITTER_FP_STREAK_MIN_OBSERVATIONS=2` and
`LITTER_CANDIDATE_DEDUP=0`; this reproduces the prior production behavior.
Removing the two new research controls and their tests fully removes the opt-in
candidate path. Research outputs are isolated and ignored by Git.
