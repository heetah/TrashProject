# Distance/time gate sweep

- Date: 2026-08-27
- Author: Codex
- Branch: `heetah-dev`
- Scope: research-only replay of the frozen 2026-08-26 candidate sidecars.

## Question

Compare the current vehicle normalized-distance gate (`0.8`) with `0.3` and
`0.4`, while combining a physical time gate of `0.25 s` with a maximum gap of
`3 frames`. Person distance remains `0.85`. The two time limits are an AND
constraint: the effective integer gap is
`min(round(fps * 0.25), 3)`; this is not an additive `0.55 s` window.

## Reproducible replay

The replay used 61 immutable candidate-event records from
`output/rtdetr_recovery_horiz1_full_20260826`, covering 41 confirmed cases and
58 usable clips. The root `.env` was loaded unchanged, including the existing
`SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=1.10`; only the requested distance
and time fields varied. Results are in
`artifacts/gate_experiment_distance_03_04_time_3f_025_20260827_env/`.

| Trial | vehicle D gate | time | adjudicated clip accuracy |
|---|---:|---|---:|
| baseline | 0.80 | 0.25 s | 30/58 = 51.72% (bootstrap 95% CI 39.66–63.79%) |
| hybrid-0.80 | 0.80 | 3 frames AND 0.25 s | 30/58 = 51.72% |
| hybrid-0.30 | 0.30 | 3 frames AND 0.25 s | 30/58 = 51.72% |
| hybrid-0.40 | 0.40 | 3 frames AND 0.25 s | 30/58 = 51.72% |

All four trials selected the same actor/route identities for every event. The
`0.3` and `0.4` trials changed only one release-frame hypothesis (case 135),
not its selected actor. Event-level 38/61 is reported only as a diagnostic;
the 61 rows are not 61 independent clips.

## Gate diagnostics

The reviewed same-frame geometry table contains 38 correct vehicle candidates
and 210 distractor candidates. Every correct candidate passed `0.3`, `0.4`, and
`0.8` (38/38; Wilson lower bound 90.82%). Distractor pass rates were:

| vehicle D gate | distractors passing | rate | Wilson 95% CI |
|---:|---:|---:|---:|
| 0.30 | 68/210 | 32.38% | 26.42–38.98% |
| 0.40 | 77/210 | 36.67% | 30.44–43.37% |
| 0.80 | 98/210 | 46.67% | 40.04–53.41% |

These are candidate-level exploratory counts; candidates from the same clip are
correlated. They support lower gates as a specificity hypothesis, not as a
universal optimum. In the full resolver replay, the tighter gates reduced
valid BC edges from 231 to 181 (`0.3`) or 190 (`0.4`), but the remaining route
alternatives were sufficient, so final route accuracy did not change.

## Validation and limits

Targeted tests passed: 60 tests. The replay does not validate detector recall,
negative/background false positives, or out-of-domain generalization. A future
locked test should select the smallest gate whose grouped/clip-level Wilson
lower bound for correct-actor coverage meets the predeclared target (for
example 0.90), then report distractor pass rate and NULL/forced-match rate.
The exact values `0.3`, `0.4`, and `0.8` must therefore be described as tested
engineering candidates, not constants proven by TrackFlow or by this small
dataset.
