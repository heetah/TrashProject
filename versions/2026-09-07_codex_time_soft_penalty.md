# Hybrid actor-observation time soft penalty

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Problem

The actor evidence timing rule previously rejected a release/actor pair when
either `gap_frames > 3` or `gap_frames/FPS > 0.25 s`. That discontinuity treats
a one-frame boundary crossing as impossible even though release timing,
detector cadence and short occlusion are uncertain.

## Implementation

Production now uses one dimensionless hybrid time variable:

```text
z_seconds = (gap_frames / FPS) / 0.25
z_frames  = gap_frames / 3
z_time    = max(z_seconds, z_frames)
rho(z)    = z + kappa * max(0, z - 1)^2
```

`kappa=4.0` is the declared initial production baseline. It preserves the
former linear ranking for `z<=1` and adds a continuous squared excess penalty
for `z>1`. Seconds and frames are combined with `max`, not addition, because
they are two scales of the same observation gap.

The same time feature is used by `C_BA`, `C_BC` and `C_AC`. Distance,
geometry and uncertainty hard gates remain unchanged. Actor history and Kalman
extrapolation remain bounded, and every event retains a finite NULL route.

## Configuration and replay

- `SMART_BACKTRACK_TIME_COST_MODE=soft` selects production soft mode.
- `SMART_BACKTRACK_TIME_SOFT_KAPPA=4.0` controls excess curvature.
- `SMART_BACKTRACK_TIME_COST_MODE=hard` reproduces the former seconds-and-frame
  AND cutoff.
- `StudyConfig.observation_time_cost_mode` and
  `StudyConfig.observation_time_soft_kappa` provide immutable replay settings.
- Resolver sidecars record the selected mode and kappa.

## Validation

- Targeted backtrack/release-policy tests cover the formula, both FPS-limiting
  cases, legacy hard replay and configuration validation: 78 passed.
- Production modules passed Python compilation and `git diff --check`.
- The full pipeline suite reported 244 passed, 12 skipped and 2 failed. Both
  failures are detector-render characterization SHA mismatches outside this
  change's backtrack scope; all non-image-hash assertions in those scenarios
  matched.
- Frozen smoke replay compared hard versus soft on 27 confirmed-event resolver
  inputs from `output/groundtruth_batch_d04_20260907`: 0/27 final route
  selections changed.
- A second replay on 26 older confirmed-event inputs changed one route (case
  135, vehicle 18 to 15); both numeric IDs were outside that replay's recorded
  GT vehicle ID, so it is not evidence of an accuracy gain.

## Limitations

- `kappa=4.0` is not statistically proven optimal.
- The 27-event replay is a no-regression smoke comparison, not a held-out
  attribution-accuracy evaluation.
- Soft mode may admit additional distractors. Promotion claims require
  event-grouped validation with candidate coverage, vehicle top-1 accuracy,
  conditional NLL, NULL rate and paired correctness changes.

## Rollback

Set `SMART_BACKTRACK_TIME_COST_MODE=hard`, or construct a replay config with
`observation_time_cost_mode="hard"`, to restore the former temporal gate
without reverting unrelated backtrack changes.
