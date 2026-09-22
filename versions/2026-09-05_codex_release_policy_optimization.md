# Explicit release time policy and vehicle cost replay

- Date: 2026-09-05
- Author: Codex
- Branch: `heetah-dev`
- Commit: the commit containing this note; baseline parent `066d895`
- Type: feat / research validation

## Problem and behavior

The previous discussion proposed a normalized BC distance cost with D_max=0.4
and a seconds-based release prior, but full-stage code still used the original
BC scales and B0/B1 observation-gap prior. This change implements both as
explicit constructor/StudyConfig options and evaluates them against identical
frozen confirmed inputs. Production defaults are retained pending independent
validation; this is not a deployed operating-point change.

`normalize_bc_distance_time_by_gate=true` uses `w_D*D/D_gate + w_T*T_E/T_gate`
for BC, preserving its other components and BA/AC weights. The study default
weights are 1/1. D/T gates remain independent of weights, including zero weights.

`max_release_back_seconds=1`, `release_soft_seconds=0.25`, and
`release_time_weight=1` replace the backward window term by
`max(0,(T_RB-0.25)/0.75)^2`, with rejection beyond one second.
`T_RB=max(0,(resolver_birth_frame-release_frame)/FPS)`. The model's two-point
and fallback costs are preserved. Forward hypotheses retain their old penalty.
The physical bound is applied before fitting-output enumeration, with floor
conversion to frames, a video-start bound, and separate computational truncation.
Sidecar metadata identifies the policy, elapsed seconds and physical bound.
Every event still has a full NULL route. No detector or confirmation code changes.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 conda run -n rtdetr python scripts/replay_release_policy.py \
  --candidates output/rtdetr_recovery_horiz1_full_20260826 \
  --output artifacts/release_policy_new_run
```

Use a fresh output directory. The script loads the root project environment
and records explicit trial and resolved per-FPS settings, input hashes, source
hashes, event selections, Wilson intervals and paired changes. The measured
run below is in `artifacts/release_policy_20260905_v2`; its baseline is 34/58,
reproducing the last production-gate report. `bc_boundary_depth_weight=1.1`
is retained consistently from the loaded environment. A later audit-only
addition records source hashes for subsequent runs.

## Results

41/58 usable clips contain 61 frozen confirmed events. The 17 clips without
confirmed events remain in the denominator. Person identity is not scored.
Clip-level success means at least one selected vehicle ID matches the reviewed
set; wrong extra events are reported separately. Cases 74/174 have no numeric
GT ID, so their historical manual +2 is explicitly separated.

| Setting | Numeric vehicle IDs | Historical +2 / 58 | Wrong vehicle events | Full NULL events |
|---|---:|---:|---:|---:|
| Current D=0.3 + old release prior | 32/56 | 34/58 | 14 | 1 |
| D=0.4 + normalized BC D/T | 32/56 | 34/58 | 14 | 0 |
| D=0.3 + new 1s release policy | 34/56 | 36/58 | 11 | 1 |
| D=0.4 + normalized BC + new release policy | 34/56 | 36/58 | 11 | 0 |

Both release-policy trials gain cases 67 and 159 and lose no correct clips.
Case 67 changes vehicle 3 to reviewed vehicle 2 and release frame 102 to 100
(birth 105). Case 159 changes both event selections from vehicle 2 to reviewed
vehicle 4, at release frames 238 and 316 (births 243 and 320).
Case 41 changes between two explicitly admissible vehicle IDs. Normalized D=0.4
also resolves an extra case-135 event from NULL to its reviewed vehicle; another
event already made that clip correct, so the clip count does not increase.

Numeric Wilson 95% intervals: baseline 44.14–69.23%; release policy 47.63–72.42%.
Historical-adjusted intervals: baseline 45.80–70.37%; release policy 49.20–73.44%.
The paired exact test for the 2 gained / 0 lost clips gives p=0.5. This is an
observed development-set gain, not statistical proof or held-out accuracy.

The minimal next candidate is `release_1s.config.json`: it obtains the same
clip gain with fewer changes. The normalized D=0.4 option is implemented and
replayable but has not shown additional clip-level benefit. No production
default is promoted from this small, repeatedly examined dataset.

## Validation

- New behavioral and reporting tests: 23 passed.
- Previous related backtrack tests: 76 passed before the final full-suite run.
- Full suite: 234 passed, 12 skipped, 2 existing image-hash failures.
- The two detect characterization golden-image SHA mismatches remain the
  existing failures documented in the 2026-08-27 version note; detector/render
  code is not modified. Optional missing-tool/media tests remain skipped.
- All four trials completed against the same 61 inputs. No detector rerun,
  independent-camera evaluation, OCR evaluation or new manual review occurred.

## Rollback and scope

Omit `max_release_back_seconds` (None) and keep BC normalization false to use
the unchanged production behavior. Both are already the default. Revert this
atomic commit to remove the new research capability. User changes in .gitignore,
deleted media/summary files and prior untracked analysis scripts were preserved.
