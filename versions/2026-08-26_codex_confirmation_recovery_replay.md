# Confirmation recovery replay

- Date: 2026-08-26
- Author: Codex
- Branch: `heetah-dev`
- Scope: reviewed 63-clip RT-DETR replay; this is a research calibration result, not a production accuracy claim.

## Problem

The conservative `LITTER_FP_CONTAINMENT_THR=0.999` replay confirmed 20/63 clips. Several reviewed true-litter clips had a vehicle-associated track with only two observations or a nearly vertical drop, so the existing vehicle gates never reached `confirmed`.

## Reproducible replay

The full replay was run with:

```bash
OUTPUT_ROOT=output/rtdetr_recovery_horiz1_full_20260826 \
SOURCE_DIRECTORY=/home/under115a/under115a/under115a/litter_vidshort/litter \
LITTER_FP_CONTAINMENT_THR=0.999 \
LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR=0 \
LITTER_MIN_CONFIRM_AGE_VEHICLE=2 \
LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE=7 \
LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT=1 \
LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE=10 \
LITTER_MIN_VEHICLE_RELATIVE_SEPARATION=0 \
LITTER_FP_STREAK_RATIO=10 \
LITTER_ALLOW_SHAKE_CANDIDATES=1 \
LITTER_CANDIDATE_SIDECAR=1 SMART_BACKTRACK=1 SMART_BACKTRACK_SIDECAR=1 \
scripts/run_backtrack_fixed_cases.sh
```

The tracker’s default horizontal gate remains 5 px. The replay-only overrides are recorded in each candidate sidecar run record.

## Results

The machine-readable report is `artifacts/litter_postprocess_calibration/recovery_horiz1_full_20260826/calibration_report.json` and the sidecars are under `output/rtdetr_recovery_horiz1_full_20260826/`.

- Confirmed clips: 41/63 (usable: 41/58); Wilson 95% CI: 52.75%–75.67%.
- Conservative containment replay: 20/63; paired gain/loss: +21/0; exact McNemar p=0.000061; paired bootstrap 95% CI for clip-rate difference: 15.52–37.93 percentage points.
- Matched tracker IDs: 38/58; reviewed correct-track matches: 28/58, Wilson 95% CI 35.93%–60.84%.
- Unverified confirmed-track proxy: 33 vs 14 in the conservative replay (+19). This is not a false-positive rate because all usable clips contain true litter and no negative clips were supplied.
- Event annotations remain `unreviewed` (58/58); the result is therefore not evidence for automatic fining.

## Interpretation and rollback

The run meets the short-term research target of more than 40 confirmed clips, but it does not validate the relaxed constants as generally safe or optimal. Before production use, review every newly confirmed route, add negative/background clips, and re-estimate thresholds with clip-level grouped validation. Roll back to the defaults by omitting the replay-only environment variables (birth actor required, age 3, downward 12 px, horizontal 5 px, ratio 3.5, relative separation 60 px, shake candidates disabled).
