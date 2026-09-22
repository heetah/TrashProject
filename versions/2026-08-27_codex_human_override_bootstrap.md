# Human-adjudicated route override and release/cost bootstrap

- Date: 2026-08-27
- Author: Codex
- Branch: `heetah-dev`
- Scope: research reporting only; no production attribution rule changed.

## Problem

The reviewed table left the vehicle ID unknown for cases 74 and 174, while the
user confirmed both routes visually.  The report also needed to explain and
quantify release estimates later than detector birth.

## Change

- Added an explicit `human_verified_correct` flag for cases 74 and 174.
- Preserved `gt_vehicle_id="?"`; no numeric identity is fabricated.
- Added `adjudicated_route_accuracy` separately from numeric vehicle-ID
  accuracy and the original strict known-ID metric.
- Added regression coverage for the override semantics.

## Evidence

- `conda run -n rtdetr python -m pytest -q tests/pipeline/test_summarize_actor_ground_truth_metrics.py`
  — 3 passed.
- Rebuilt `artifacts/actor_ground_truth_metrics/recovery_horiz1_full_20260826`.

The clip-cluster bootstrap used 10,000 percentile replicates (seed
`20260827`).  For the BC component, 61 event rows form 41 clip clusters;
the clip-mean estimates were `D=0.0162` (95% CI `0.0040–0.0323`),
`T=0.00813 s` (95% CI `0.00163–0.01667`), and the full-production D/T part
`D+0.35T=0.0191` (95% CI `0.00567–0.03578`).  The gate-normalized 0.8/0.2
research cost was `0.0227` (95% CI `0.00755–0.04108`).

For an independent expansion diagnostic, the GT release point was compared
with the unexpanded vehicle box for 38 matched vehicle events.  The required
horizontal expansion had median `0`, Q95 `0.3599` (bootstrap 95% CI
`0.1170–0.7357`), while the required vertical expansion was zero for all 38.
The current `0.18/0.15` expansion contained 33/38 (`86.84%`, Wilson 95% CI
`72.67–94.25%`) of these GT points.  This supports the scale-free expansion
diagnostic, but does not statistically prove `0.18/0.15` as universal values.

For model-derived candidate geometry, the normalized distance gate `0.80`
passed 38/38 correct actors (Wilson `90.82–100%`) and 98/210 distractors
(Wilson `40.04–53.41%`).  The temporal gate `0.25 s` passed 38/38 correct
actors and 186/210 distractors (Wilson `83.56–92.20%`).  These are recall and
physical-validity operating points, not estimates of a unique optimum.

## Current limitation

Case 74 has no confirmed sidecar event, so the manual override is an
adjudicated route label, not evidence that the detector confirmed the litter.
The bootstrap cost values reported separately are descriptive and use a
clip-level resampling unit; they do not prove universal weights.
