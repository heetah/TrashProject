# 2026-09-15 Codex — target-41 mask temporal rerank research

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working tree
- Scope: research replay only; no production resolver, cost, route schema or
  default configuration change

## Question

Can the existing YOLO-Seg vehicle masks add an explainable, low-compute
front/behind cue to Smart Backtrack without camera calibration or a new CNN?

## Candidate rule

The replay compares only valid `direct_vehicle` routes within the same release
model. It uses the existing diagnostic
`s = tanh(d_polygon / bbox_diagonal)`, where OpenCV's signed polygon distance
means `s > 0` when the litter center is inside the visible mask and `s < 0`
when it is outside. The release evidence is the median of the release frame and
first post-release frame (`offsets = {0,+1}`); the original route cost remains
the tie-breaker. The sign convention is covered by
`tests/pipeline/test_mask_litter_diagnostics.py` and must not be inverted by a
future replay.

For ballistic routes, an alternative is admissible only when:

```text
s_alt({0,+1}) > s_selected({0,+1}),
s_alt(0) > 0,
s_alt(0) - s_alt(-1) > 0,
the retained history is outside (all s < 0), or at least two available
pre-release samples are outside,
the candidate history q25 is below the selected history q25,
direct_distance_alt = 0.
```

Among admissible alternatives, the smallest existing route cost is retained.
For `constant_velocity_2point`, the research guard additionally requires the
aggregate transition `s_alt({0,+1}) < s_selected({0,+1})`, with
`s_selected(0) > 0` and `s_alt(0) < 0`, higher existing spatial Mahalanobis
diagnostic, and no greater existing uncertainty. Existing appearance quality is
not used as a second gate: it is not a causal release/vehicle measurement and
would reject the same mask-supported correction in `litter_case_143`. No new
weight is added and NULL/non-direct/person routes are left unchanged.

The ballistic condition is intentionally stated as an auditable sign-pattern
over the backward release hypotheses and the observed post-release history; it
does not assert that a 2D mask proves a physical 3D throw direction. This is a
geometric transition/tie-break diagnostic, not a metric depth estimate,
calibrated probability, or learned accuracy score.

## Evidence

The fixed 58-clip reviewed positive set was evaluated with the official paired
readiness evaluator:

| run | exact route correctness | accepted event match |
|---|---:|---:|
| frozen baseline | 38/58 (65.52%) | 50/58 (86.21%) |
| mask temporal candidate | 43/58 (74.14%) | 50/58 (86.21%) |

The five route gains were `litter_case_143`, `litter_case_168`,
`litter_case_193`, `litter_case_25`, and `litter_case_41`; there were no
correctness or event-match losses. The exact paired two-sided test is
`p = 0.0625`; therefore the promotion gate remains **FAIL** despite exceeding
the 41/58 point target.

The candidate was unchanged across YOLO-Seg confidence floors 0.05, 0.10 and
0.20 (43/58, same five gains, no losses). A 47-case route-coverage audit (the
truth route exists in the frozen candidate set) also reports 43/47 with the
same five gains and no losses; this is not an independent camera holdout.

An additional inference-scale perturbation used the same frozen candidate
records and release-mask rule while rerunning YOLO-Seg at `imgsz=512`, `640`,
and `768` (confidence `0.10`). The measured mask records were 431, 437, and
439 respectively; all three scales selected the same five gains
(`litter_case_143`, `168`, `193`, `25`, `41`), with no correctness or
event-match losses:

| YOLO-Seg image size | route correctness | changes vs baseline | gains | losses |
|---:|---:|---:|---:|---:|
| 512 | 43/58 | 5 | 5 | 0 |
| 640 | 43/58 | 5 | 5 | 0 |
| 768 | 43/58 | 5 | 5 | 0 |

This is a within-set inference perturbation, not evidence of camera-
independent generalization or calibrated accuracy. Outputs are retained under
`/tmp/target41_mask_imgsz_stability/` with per-scale audits and paired reports.
Of the 431 observations available at all three scales, 24 changed the sign of
the raw signed-mask diagnostic across scales; the final route decision still
remained unchanged. Thus the evidence supports decision-level stability of
this guarded replay, not pixel-level or physical-depth reliability.

The same replay was tested after in-memory JPEG re-encoding of every sampled
frame at quality 90 and 50 (then YOLO-Seg `imgsz=640`, confidence `0.10`). Both
compression levels retained 43/58, the same five gains, and zero correctness or
event-match losses. Measurements were 440 at each level. This supports
stability to ordinary codec noise, while leaving camera/site generalization
unverified.

Across the frozen replay, confidence floors `0.05/0.10/0.20`, image sizes
`512/640/768`, and JPEG qualities `90/50`, all 58 per-case route tuples were
identical to the retained `{0,+1}` candidate. The consensus audit therefore
has 58/58 stable decisions across nine replay conditions, but this is a
decision-invariance result on the same development clips, not a new accuracy
denominator or a camera holdout.
Two independent default-scale (`imgsz=640`, confidence `0.10`) mask passes
also produced bit-identical signed values for all 437 common measurements,
confirming deterministic replay under the current runtime.

The all-condition rule is now captured by the research-only
`pipeline.backtrack.replay_consensus` primitive. It compares canonical route
tuples rather than local route IDs, returns a route only on unanimous
agreement, and fails closed to NULL/manual review on disagreement. Its focused
tests do not alter the resolver; the 58/58 consensus remains an in-sample
invariance result.

As an acquisition-format sensitivity check (not a camera split), the reviewed
clips break down as follows:

| observed format | n | baseline | candidate | accepted event match (both) |
|---|---:|---:|---:|---:|
| 1440×1080 @ 30 FPS | 19 | 14/19 (73.7%) | 16/19 (84.2%) | 18/19 (94.7%) |
| 2592×1944 @ 10 FPS | 36 | 21/36 (58.3%) | 24/36 (66.7%) | 29/36 (80.6%) |
| 2592×1944 @ 12 FPS | 3 | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |

The gains occur in the first two strata and there are no format-stratum
correctness losses, but the strata are only resolution/FPS metadata; they do
not establish camera or site independence.

The release-frame sensitivity check is deliberately recorded: backward-only
windows can reduce the gains, while the causal `{0}` and `{0,+1}` windows
retained 43/58. Windows containing `-1` retained 42/58 and the five-frame
window retained 41/58, all without losses in this positive set. This supports
using the window only as a research hypothesis, not as a production guarantee.

To test a single-observation dropout/latency condition without selecting a new
weight, the same guarded rule was replayed with each one-frame window
`{-2,-1,0,+1,+2}`. Relative to the fixed 38/58 baseline, the candidate reached
40/58, 41/58, 43/58, 42/58 and 40/58 respectively (ordered by offset
`-2,-1,0,+1,+2`). Every window had zero route-correctness losses and zero
accepted-event losses; only `-1`, `0` and `+1` retained at least the 41/58 point
target. The machine-readable result is
`/tmp/target41_mask_single_offsets__it1z8re/summary.json`. This is a bounded
missing-observation sensitivity check over the same development clips, not an
independent camera or occlusion test.

Replay isolation was then checked by comparing the 93 baseline and candidate
sidecar records by `(input_video, litter_id)`. The key sets and record counts
were identical; all route payloads (including costs and metadata) and all
non-decision evidence were bit-for-bit unchanged after removing only the
documented selection mirrors. Exactly five assignment mirrors changed, the
same five gain cases, with zero residual evidence differences. This supports
the causal explanation that the candidate is a route-selection replay over
existing evidence rather than a hidden detector/tracker change. It does not
prove the selected route is correct or camera-independent. Audit:
`/tmp/target41_mask_candidate_lineage_audit.json`.

For handoff and review, `target41_evidence.py` now builds a strict ledger from
the official baseline/candidate case tables, readiness reports and the
nine-condition consensus report. The frozen `{0,+1}` ledger has 58 identical
case IDs, 53 unchanged cases, exactly five `wrong_route -> correct_route`
transitions, zero losses, and a 43/58 candidate point estimate. It assigns
explicit per-case explanation codes without exposing model scores or treating
consensus as truth. Output: `/tmp/target41_evidence_ledger.json`; this is an
auditable development artifact, not a cross-camera accuracy claim.

An updated route-candidate oracle on the 43/58 replay finds the reviewed truth
tuple among valid routes for four remaining wrong assignments
(`litter_case_141`, `16`, `78`, `9`). The in-sample candidate-space ceiling is
therefore 47/58, but the four gaps are person identity/occlusion or competing
vehicle geometry cases; no common, label-free rule has yet separated them
without risking losses. The oracle report is
`/tmp/target41_mask_route_oracle_gap_v2/REPORT.md` and is diagnostic only.

As a negative-control research screen, a linear pairwise ridge ranker was fit
only on existing AC/BA/BC route cost components and evaluated with
source-case leave-one-group-out folds (44 trainable mapped cases, 41 groups).
Across regularization values `lambda ∈ {0.1, 1, 10, 100}`, it fell to 37/58;
the 7–8 losses outweighed at most two gains. This rejects learned reweighting
of the current cost components as a safe next change. The frozen candidate is
retained and no ranker is connected to production. Results:
`/tmp/target41_pairwise_source_cv_v1/summary.json`.

## Reproducibility inputs

- Candidate source: `/tmp/target41_loocv_output/loocv_backtrack_candidates.jsonl`
- Release mask metrics: `/tmp/target41_mask_route_metrics.json`
- History mask metrics: `/tmp/target41_mask_history_metrics.json`
- Temporal audit: `/tmp/target41_mask_temporal_stability/audit.json`
- Official candidate report: `/tmp/target41_mask_temporal_aggregate/+0_+1/candidate_eval/readiness.json`
- Paired report: `/tmp/target41_mask_temporal_aggregate/summary.json` (the `+0_+1` entry contains the paired result)
- Ground-truth hashes: event `c59e4d0f7e99c478d8ef6bbece7047aec4564716d7559391a9a855422731f7bd`, actor `d7d0de3d2962b10cfdf3d351da55e7a9d520ed835f50b6845db0f4e3aa56e187`, clip `3c5c2b03b225481098cc72eeb0aabb0fc0e7cb68ba7ec5031abec5656b1f5861`.

## Limits and next gate

All 58 clips are one positive development set. No reviewed negative set,
independent camera-group split, reviewed plate ground truth, or statistical
significance is available. The Wilson 95% interval for 43/58 is approximately
`[0.616, 0.837]`, below the product's 85% lower-bound requirement. The mask
candidate therefore stays research-only; the current production default and
resolver behavior remain unchanged.

Before integration, collect a new camera-disjoint reviewed set and annotate
the release/occlusion transition independently. Re-run the same frozen paired
comparison and require no regressions under temporal and confidence
perturbations.

## Rollback

No runtime rollback is needed because this note records an external replay and
the candidate has no production caller. Discard the `/tmp/target41_mask_*`
replay outputs to remove the experiment artifacts.
