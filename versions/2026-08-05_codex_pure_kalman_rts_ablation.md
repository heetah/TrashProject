# D+T + pure Kalman/RTS ablation

- Date: 2026-08-05
- Author: Codex
- Branch: current working branch
- Commit: uncommitted working tree

## Problem

The existing `uncertainty` study stage enabled Kalman/RTS together with
confidence, covariance gates, covariance costs, and Mahalanobis distance. It
could not isolate whether actor position smoothing alone improved attribution.

## Change

- Added `kalman_rts` study/runtime stage.
- Reused the gate-normalized D+T cost configuration without extra components.
- Enabled actor Kalman/RTS interpolation only.
- Used uniform measurement confidence so detector confidence cannot affect the
  smoothed position in this ablation.
- Disabled covariance hard gates and reverse litter trajectory.
- Kept Min-Cost Flow, all route types, and the complete NULL route unchanged.
- Added resolver diagnostics and targeted tests for the isolation boundary.
- Exposed process-noise scale, measurement-noise scale, and maximum actor
  extrapolation seconds as immutable replay parameters; runtime converts
  seconds using each video's FPS.
- Kept D+T time tied to the nearest real detector frame; a predicted state at
  release time cannot turn stale evidence into a zero time gap.

## Interface

```text
SMART_BACKTRACK_STUDY_STAGE=kalman_rts
SMART_BACKTRACK_DT_DISTANCE_WEIGHT=0.5
SMART_BACKTRACK_DT_TIME_WEIGHT=0.5
```

Replay JSON uses `"stage": "kalman_rts"`.

Optional replay tuning keys:

```json
{
  "kalman_process_noise_scale": 1.0,
  "kalman_measurement_noise_scale": 1.0,
  "kalman_max_extrapolation_seconds": 0.3
}
```

## Validation

Targeted tests cover exact D+T-only weights and Kalman/RTS interpolation at a
missing release frame. Full validation evidence is recorded after execution.

## Limits

This change enables a clean ablation; it does not itself establish accuracy.
The fixed multi-actor set still needs replayable sidecars and human-reviewed
admissible routes before weight or accuracy claims.

## Rollback

Remove the `kalman_rts` stage and use `distance_time` for the prior baseline.
