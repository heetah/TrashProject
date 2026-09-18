# 2026-09-13 Codex — reviewed metadata correction and backtrack multiview research

- Date: 2026-09-13
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `a149c5975898643e5131caf0579bd0aefe228d75`
- Commit: uncommitted working-tree change

## Problem

The 58 event annotations had already been reviewed by the single responsible human,
but their `review_state` metadata remained `unreviewed`. Backtrack accuracy also
remained below the 85% deployment target, requiring bounded multi-angle research
without silently changing production defaults.

## Changes

- Corrected the three ground-truth exports to `review_state=reviewed`; no semantic
  event, release, actor, route, bbox, or ID field changed.
- Added an auditable one-time correction manifest and guarded correction script.
- Made readiness limitations and Markdown output conditional on actual review state.
- Evaluated a locked confirmation-actor-hint reranker and rejected it after regressions.
- Evaluated 16 locked one-at-a-time hypotheses across geometry, trajectory, Kalman,
  route topology, release time, and distance/time weighting.
- Retained `direct_vehicle_penalty=1.1` plus
  `release_window_prior_weight=0.2` as a research-only fine-tune candidate after a
  locked combined replay; production defaults remain unchanged.
- Documented remaining failure mechanisms and the next low-compute wrist-distance
  hypothesis that reuses existing YOLO-Pose output.

## Evidence

- Ground-truth event rows: 58/58 reviewed.
- Baseline exact route: 31/58 (53.45%).
- Combined research candidate: 33/58 (56.90%), gain/loss=2/0.
- Correct route given accepted event: 33/41 (80.49%).
- Event sensitivity unchanged: 41/58 (70.69%).
- Paired two-sided exact McNemar p=0.5; not statistically conclusive.
- Event-match tiers changed from 16/25/2/15 to 13/28/2/15
  (strict/moderate/exploratory/unmatched); three strict matches regressed to moderate.
- The event-match metric depends on selected release coordinates and is not a pure
  detector-sensitivity measurement.
- Confirmation actor hint: rejected because even weak penalties caused losses.
- Pipeline regression suite: 347 passed, 12 skipped.
- JSONL, CSV, and XLSX exports each contain 58 reviewed event rows; `git diff --check`
  passed.

## Interface and configuration

No production interface, schema, or default changed. The retained candidate uses
existing `StudyConfig` fields only. The canonical event annotation hash after the
metadata correction is
`c59e4d0f7e99c478d8ef6bbece7047aec4564716d7559391a9a855422731f7bd`.

## Limitations

This is positive-only development characterization on the same cohort used to select
the candidate. Reviewed negatives, independent camera groups, OCR truth, and a locked
holdout remain unavailable. Dynamic pseudo-homography was applied to 0 cases because
no runtime calibration reached `LOCKED`, so it provides no measured gain here.

## Rollback

Production requires no rollback because defaults were not changed. To reverse only
the review-state correction, use the before hashes and legacy seed recorded in
`artifacts/ground_truth_review_correction_20260913/manifest.json`; do not reverse it
unless the human attestation itself is withdrawn.

## Follow-up: wrist-release hypothesis

A bounded, dimensionless single-frame wrist-distance penalty was implemented behind
a zero default and tested on a locked ten-case same-run screen. Twelve grid trials
produced zero target gains; eight regressed correct case 194 and one also degraded
case 42's event-match tier. The runtime wrist field, scoring logic, research config,
and tests were therefore removed, restoring the exact pre-study production/data path.
The protocol, sidecars, and report remain under
`artifacts/wrist_release_study_20260913/` as negative evidence.

## Follow-up: release-hypothesis aggregation

Cases 9 and 67 were separated into different upstream failures. Case 67 has an
accurate litter birth point but loses because candidates minimize over different
release frames; case 9 has only two post-release observations and a 323 px release
point error, so its pre-birth motion is not identifiable without an additional
motion prior.

Two formulas were locked before evaluation on the 43 reviewed events with a mapped
model vehicle: a normalized soft minimum over release frames and a parameter-free
worst-case rule over the zero-cost release window. Both reduced direct-vehicle
top-1 correctness from 38/43 to 36/43. The robust rule repaired cases 168, 67, and
7 but regressed five cases, including mandatory controls 25 and 41. Both formulas
were rejected; production code and defaults remain unchanged. Full derivation and
gain/loss evidence are retained under
`artifacts/release_hypothesis_triage_20260913/`.
