# Promote frozen guarded Smart Backtrack policy

- Date: 2026-09-18
- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree only; no commit/push/deploy performed

## Problem

The best frozen research composition reached 48/58 route-correct reviewed
development clips, but production still used the earlier minimum-cost route
selection. The user explicitly requested promotion of that research policy.

## Changes

- Added `pipeline/backtrack/route_reselection.py` and connected it after both
  per-event and global Min-Cost Flow solves.
- Promoted the frozen `{release,+1}` temporal segmentation-mask crossing rule
  and five sequential guarded comparisons: ballistic boundary ambiguity,
  same-vehicle `C_AC` endpoint, `C_BC` quality, direct-vs-person, and same-model
  motion consistency.
- Retained only observed `seg_track`/`seg_predict` vehicle/scooter contours,
  capped at 512 vertices. Cache/Kalman observations never become mask evidence.
- Kept event confirmation, candidate hard gates, route costs, route generation,
  actor capacities and the complete NULL route unchanged. Missing/non-finite
  rule inputs fail closed to the current route.
- Changed the frozen direct-vehicle route penalty from `0.9` to `1.1`; kept
  `SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=0.0`. Dynamic pseudo-homography
  remains disabled and is not part of this promotion.
- Every re-selection stores an auditable ordered `rule/from/to` trace.

For point `p`, vehicle mask polygon `M`, and vehicle bbox diagonal `D`, the mask
feature is:

```text
s(p,M) = tanh(signed_polygon_distance(p,M) / D)
```

Positive values are inside the visible silhouette and negative values are
outside. This is dimensionless 2D silhouette evidence, not metric depth. The
release aggregate is the median of available observations at offsets `{0,+1}`.
Each later comparison only accepts an existing valid alternative whose frozen
feature and route-cost-difference guards all pass; it adds no learned score.

## Configuration

```text
SMART_BACKTRACK_GUARDED_RESELECT=1
SMART_BACKTRACK_MASK_RESELECT=1
SMART_BACKTRACK_DIRECT_VEHICLE_COST=1.1
SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=0.0
```

Set `SMART_BACKTRACK_GUARDED_RESELECT=0` to return to Min-Cost Flow selection.
Set only `SMART_BACKTRACK_MASK_RESELECT=0` to disable mask retention and the
mask rule while retaining the other five guards.

## Evidence

- Focused tests: 54 passed.
- Full `tests/pipeline`: 449 passed, 12 skipped.
- Full `tests`: 476 passed, 12 skipped.
- Frozen sidecar equivalence for the mask rule: 93 records, 5 changed
  assignments, 0 route-ID mismatches against the archived target-43 candidate.
- Frozen sidecar equivalence for the five post-mask guards: 93 records, 9
  changed assignments, 0 route-ID mismatches against the archived target-48
  candidate.
- Existing development result: 48/58 route-correct clips (82.76%), five gains
  and zero losses from 43/58; this is not an independent holdout result.

## Limitations

- 48/58 is below the 85% release requirement; at least 50/58 are needed.
- The 58 reviewed positives were used for research selection. There is no
  reviewed camera/source-recording-disjoint holdout and no reviewed negative
  set, so this must not be advertised as calibrated production accuracy.
- Production YOLO-Seg cadence may leave one temporal offset absent. The frozen
  aggregation explicitly supports available observations, but the live online
  path still requires a complete 58-video rerun before claiming it reproduces
  the archived 48/58 point estimate.
- Contours increase bounded task/sidecar memory; no bitmap or CNN feature tensor
  is retained.

## Rollback

Use the two environment switches above for immediate behavioral rollback. A
code rollback removes `route_reselection.py`, its resolver calls, and bounded
`mask_contour_xy` capture, then restores the direct-vehicle penalty to `0.9`.
