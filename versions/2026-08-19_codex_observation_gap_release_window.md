# Observation-gap release window

- Date: 2026-08-19
- Author: Codex
- Branch: current checkout
- Commit: working tree, not committed

## Problem

The previous two-point release model imposed a fixed 0.4-second reverse limit.
That constant could discard a valid release near `T_B0 - 1 second` without
physical evidence. The confirmation frame was also too easy to confuse with a
motion observation even though it only marks accumulated event evidence.

## Change

- Define `B0`, `B1`, and `B2` as the first three distinct accepted RT-DETR
  observations of the same confirmed litter track.
- Define the high-suspicion interval with the first observation gap:
  `H0 = T_B1 - T_B0`, `I0 = [T_B0 - H0, T_B0]`.
- Assign zero window-time cost inside `I0`. Earlier reverse candidates add
  `lambda_w * (T_B0 - H0 - T_r) / H0`; later candidates on the observed
  airborne path retain `lambda_f * (T_r - T_B0) / FPS`. Neither is rejected.
- Derive source direction from the reverse of `B0 -> B1`; use `B2` only to
  report adjacent-velocity cosine consistency.
- Keep `max_back_frames` only as a computational enumeration guard and expose
  `search_truncated` plus `truncation_reason` in candidate sidecars.
- Keep the one-point dustbin prior, two-point low-evidence base prior, physical
  distance gates, NULL route, Hungarian identity scope, and Min-Cost Flow route
  selection unchanged.

`confirm_frame` does not affect the observation-gap window or early direction.
`SMART_BACKTRACK_TWO_POINT_MAX_BACK_SEC` remains accepted for old trial/replay
configuration compatibility but no longer truncates two-point hypotheses.

## Interface and diagnostics

New production/research control:

```text
SMART_BACKTRACK_RELEASE_WINDOW_WEIGHT=0.35
```

Each release hypothesis now records:

- `observation_gap_frames`
- `zero_cost_window_start_frame`
- `zero_cost_window_end_frame`
- `window_prior_cost`
- `source_direction_uv`
- `direction_consistency`
- `search_truncated`
- `truncation_reason`

## Validation

```text
conda run -n rtdetr python -m pytest -q \
  tests/pipeline/test_backtrack_costs.py \
  tests/pipeline/test_smart_backtrack_integration.py \
  tests/pipeline/test_backtrack_sidecar.py \
  tests/pipeline/test_backtrack_study.py
```

Targeted result: `58 passed`. Full pipeline result: `175 passed, 12 skipped`.

Coverage includes gap-window boundaries, retention of a one-second-old
hypothesis, FPS invariance, `B0/B1/B2` direction, confirmation-frame
independence, full resolver integration, and sidecar serialization.

The seven frozen resolver inputs were replayed with `lambda_w=0.35`. After
preserving the seconds-based post-birth prior, six routes were identical to the
previous candidate and Case 42 also retained its reviewed
`person:2 -> vehicle:1` route at frame 32. Case 25 remained incorrectly assigned
to vehicle 1, showing that the remaining failure is actor evidence/occlusion,
not solved by a release-time prior alone. The checked-in annotation queue is
still unreviewed, so this replay is a route comparison, not an accuracy metric.

## Limits and rollback

- The default `lambda_w=0.35` is an explicit starting point, not a claimed
  accuracy optimum. It must be selected on reviewed validation data and tested
  once on the locked test set.
- A finite computational guard can still truncate an earlier true release;
  sidecar diagnostics make this visible rather than calling it invalid.
- Direction consistency is diagnostic in this version. It does not yet reward
  or reject an actor without a reviewed ablation.
- Revert the changes in `trajectory.py`, `resolver.py`, and `study.py`, then
  remove the new environment field to restore the previous fixed-horizon
  behavior.
