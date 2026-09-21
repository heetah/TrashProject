# Fresh actor gate and source FPS

- Date: 2026-09-19
- Author: Codex
- Branch: `heetah-dev`
- Base: `528ad79` plus working-tree changes
- Status: implementation in working tree; not committed

## Problem

YOLO-Seg skip-frame cache already carries `observed=false`, but vehicle gate
checked only whether the vehicle list was non-empty. Cached actors could refresh
`last_vehicle_frame_index`, extending the gate beyond the latest detector
observation. `main.py` also rounded source FPS before downstream time conversion.

## Change

- Added `scripts/pipeline/actor_gate.py` as a pure freshness/TTL module.
- Integrated single-frame and batch vehicle gates with that module.
- Explicit `cache`, `predicted`, and `prediction` sources never refresh TTL.
- Legacy precomputed actors without provenance remain fresh for compatibility.
- Added `scripts/pipeline/timebase.py`; source FPS now remains fractional and
  invalid FPS uses the existing positive fallback.
- Added focused tests for stale cache, TTL expiry, batch simulation, and FPS
  conversion.

## Validation

- Focused suite: `52 passed`.
- Full `tests/pipeline`: `462 passed, 12 skipped`.
- Sampled gate replay at 10 FPS, 3-second TTL:
  `(fresh@10, cache@11, cache@40, cache@41)` →
  `(active,last=10)`, `(active,last=10)`, `(active,last=10)`,
  `(inactive,last=10)`.
- Compile check passed for changed production modules.

## Limits

- Source PTS/VFR support is not implemented; `SourceClock` currently models
  constant-FPS video.
- No full 406-video rerun was performed; existing output remains baseline
  evidence only.
- This change does not alter attribution costs, route selection, or event
  confirmation thresholds.

## Rollback

Revert the `actor_gate.py`/`timebase.py` integration and retain existing tests;
no model weights or output artifacts were modified.
