# Case 168 Smart Backtrack margin optimization

- Date: 2026-08-20
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `f183e47`
- Commit: working tree, not committed

## Problem

Reviewed truth for `litter_case_168` is `direct:vehicle:1` (BSA-2371), not
the foreground cement mixer `vehicle:3`. The previous resolver selected
`vehicle:3`; the signed truth margin
`min(C_competing_route) - C_truth_route` was `-0.397778`.

Only tracker-accepted litter observations at frames 147 and 148 were used.
That two-point reverse fit pointed toward the foreground vehicle, even though
the RT-DETR detector had already produced a coherent raw trajectory at frames
144, 145, and 146 before geometry/motion/holding postprocessing.

## Changes

1. Preserve RT-DETR class/confidence litter boxes before postprocessing.
2. Only after the litter event is confirmed, recover a contiguous,
   velocity-consistent raw prefix for release fitting. Raw boxes never enter
   pending/confirmed state and cannot create or promote an event.
3. Define `B0` as the earliest actual recovered/accepted observation, so the
   B0/B1 observation-gap release window uses the evidence rather than the old
   tracker birth timestamp.
4. Add a soft `C_BC` `boundary_depth` feature: normalized distance from an
   in-box release point to the nearest original vehicle bbox boundary. It
   penalizes a release point swallowed deep inside a large occluding bbox; it
   does not relax physical gates.
5. Add immutable study weights for existing BC direction diagnostics and the
   new boundary-depth component. Add runtime controls:
   - `SMART_BACKTRACK_RAW_PREFIX=1`
   - `SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=0.0` by safe default
6. Save accepted/recovered frame provenance in replayable sidecars.

## Sequential evidence

### Case 168 real-video rerun

- Previous: wrong `direct:vehicle:3`, truth margin `-0.397778`.
- Raw prefix only, boundary weight 0: correct `direct:vehicle:1`, margin
  `+0.001827`.
- Recovered raw frames: 144, 145, 146.
- Original tracker-accepted frames: 147, 148.
- Existing `exit_deficit` and `relative_motion_deficit` weights had no effect.
- Positive `reverse_direction` weight reduced the truth margin and was rejected.

### Boundary-depth sweep

The truth margin increased monotonically:

| Weight | Case 168 truth margin |
|---:|---:|
| 0.00 | +0.001827 |
| 0.25 | +0.094303 |
| 0.50 | +0.186779 |
| 1.00 | +0.371732 |
| 1.09 | +0.405024 |
| 1.10 | +0.405228 |
| 1.12 to 4.00 | +0.405228 |

`1.10` is therefore the smallest tested weight that reaches the maximum
margin plateau. The current local `.env` selects `1.10`; the shared example
keeps `0.0` until a larger reviewed set is approved.

### Reviewed frozen-route safety replay

Seven reviewed litter events were replayed: cases 25, 42, 50 (two events),
51, 165, and 168. At weights 0 through 4, all six already-correct route
selections remained correct. Case 25 remained its known wrong vehicle and was
not counted as fixed. This is conditional attribution replay, not detector or
end-to-end accuracy. Cases other than 168 were not rerun through detector
inference with raw-prefix recovery.

## Final runtime validation

Command used (GPU 1):

```bash
CUDA_VISIBLE_DEVICES=1 \
OUTPUT_ROOT=output/case168_final_w110_20260820 \
SMART_BACKTRACK=1 SMART_BACKTRACK_SIDECAR=1 LITTER_DEBUG=1 \
SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=1.10 \
conda run -n rtdetr python scripts/main.py \
  /mnt/8tb_hdd/under115a/litter_vidshort/litter/litter_case_168.mp4
```

Result:

- selected: `direct:vehicle:1`
- truth cost: `0.905675`
- second route (`direct:vehicle:3`) cost: `1.310903`
- actor/route margin: `0.405228`
- release: frame 144, point `(903.15, 477.32)`
- output video: H.264, 2592x1944, 240/240 frames, ffprobe decodable
- pipeline tests: `182 passed, 12 skipped` with `LITTER_DEBUG=0`

## Limits and rollback

- This change maximizes one reviewed case under a small seven-event safety
  replay. It is not evidence of general attribution accuracy.
- Case 25 remains wrong and needs a separate occlusion/identity analysis.
- Set `SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=0` to disable the new BC cost.
- Set `SMART_BACKTRACK_RAW_PREFIX=0` to disable confirmed-only raw recovery.
- NULL route, Min-Cost Flow, actor capacities, Hungarian identity scope, and
  all physical hard gates are unchanged.
