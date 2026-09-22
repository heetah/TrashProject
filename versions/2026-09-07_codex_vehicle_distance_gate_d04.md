# Vehicle normalized-distance gate D=0.4

- Date: 2026-09-07
- Author: Codex
- Branch: current working branch
- Commit: uncommitted working-tree change

## Problem

The requested production direct litter-to-vehicle physical gate is `D <= 0.4`,
where `D = dist(release_point, unexpanded_vehicle_bbox) / bbox_diagonal`.

## Change

- Changed `SmartBacktrackConfig.normalized_distance_gate_vehicle` from `0.3`
  to `0.4`.
- Changed the default research `StudyConfig` to match production.
- Kept the vehicle bbox unexpanded and retained the actor-evidence time gate of
  at most 3 frames AND 0.25 seconds.
- No attribution cost weight or release-time policy was changed.

## Existing development evidence

Frozen full-resolver replay previously found that widening `D` from 0.3 to 0.4
did not change any top-1 prediction in the 35-event overlap cohort. The correct
vehicle was already retained in all but one exact-GT-geometry event at D=0.4;
severe overlap mostly creates multiple spatially valid vehicles, so this change
improves tolerance but does not itself solve attribution ambiguity.

## Validation

Targeted config, cost-boundary, release-policy, and overlap replay tests passed:
72 tests. Frozen full replay gave 32/56 numeric-ID correctness (34/58 under the
existing manual-confirmation convention). The 35-event overlap cohort remained
30/35, so widening the gate produced no paired ranking change. A research-only
ablation that preserved exact queued detector boxes at observed frames also
produced no paired overlap correctness change. Historical reports remain
historical and are not rewritten.

## Limitations and rollback

This is a requested operating point, not a statistically identified universal
constant. Roll back by restoring both config defaults to `0.3` and reverting
the corresponding current-state documentation.
