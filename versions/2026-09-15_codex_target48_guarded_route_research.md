# 2026-09-15 — Target-48 guarded Smart Backtrack research

- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree (no commit/push/deploy performed)
- Scope: frozen-sidecar research and validation only
- Production caller: none

## Question

Can existing, already logged Smart Backtrack evidence be combined into an
interpretable route guard that reaches the interim target of 48 correct clips
out of the fixed 58 reviewed positive development clips, without changing
event confirmation or introducing a new uncalibrated production weight?

## Frozen input and evaluator

The baseline is the current temporal-mask candidate sidecar:

```text
/tmp/target41_mask_temporal_aggregate/+0_+1/aggregate_backtrack_candidates.jsonl
SHA-256: 72cb1bdad7e768d1e29277a49c94345e0db513170280afc6326bc20515532882
```

It contains 93 confirmed-record rows for 58 clips. The official evaluator is
`scripts/evaluate_reviewed_readiness.py`; it keeps all 58 usable positive clips
in the denominator, maps manual actors to tracklets from the same run, and
accepts only strict/moderate assignment-conditioned release matches. Ground
truth event labels are reviewed (58/58), but this remains a positive-only
development characterization.

## Research composition

For a selected valid route `S` and an alternative valid route `R`, `C(R)-C(S)`
is the existing sidecar route-cost difference. The rules below are sequential,
deterministic comparisons; they do not multiply history points as independent
measurements and do not alter the resolver's production cost function.

1. **Ballistic boundary ambiguity.** For a direct-vehicle ballistic route,
   if `boundary_depth(S) >= 0.99`, compare same-model direct routes with
   `boundary_depth(R) <= 0.20` and `C(R)-C(S) <= 0.50`; choose the smallest
   cost difference. `boundary_depth` is the existing clipped distance-to-box
   edge diagnostic in image coordinates, not metric depth or a recovered 3-D
   coordinate.
2. **Same-vehicle person disambiguation.** For a person-vehicle route, compare
   a different person on the same vehicle when
   `endpoint_AC(S)-endpoint_AC(R) >= 0.02` and
   `C(R)-C(S) <= 0.15`; choose the lowest route cost.
3. **BC quality consistency.** For a person-vehicle route, compare a different
   person on the same vehicle when `quality_BC(S) >= 0.80`,
   `quality_BC(R) <= 0.05`, and `C(R)-C(S) <= 0.50`; choose the lowest route
   cost. The quality values are existing dimensionless cost features, not
   calibrated probabilities.
4. **Weak person-to-vehicle association.** For a selected person-vehicle
   route with `endpoint_AC(S) <= 0.25` and `overlap_AC(S) <= 0.10`, compare the
   direct route for the same vehicle if `C(R)-C(S) <= 1.00`.
5. **Same-model motion consistency.** For a selected direct route, require
   `reverse(S) >= 0.10` and `exit(S) >= 0.20`; a same-release-model alternative
   may replace it only when `reverse(R) <= 0.02`, `exit(R) <= 0.05`,
   `relative_motion(R) <= 0.02`, and `C(R)-C(S) <= 2.00`.

All thresholds above are caller-supplied research values selected for this
development screen. They are not derived camera parameters, confidence
calibration, or production defaults. The composition preserves the full NULL
route and never turns a candidate into a confirmed litter event.

## Results

The generated candidate is:

```text
/tmp/target48_combined_research/combined_backtrack_candidates.jsonl
SHA-256: 87d0246a3fe8ba9d99d56314072528f112dac37c03d0f802cffc8221eace0d2f
```

The official paired comparison is:

| Metric | Baseline | Research candidate |
|---|---:|---:|
| End-to-end route-correct clips | 43/58 (74.14%) | **48/58 (82.76%)** |
| Accepted event matches | 50/58 (86.21%) | **51/58 (87.93%)** |
| Route correctness conditional on accepted event | 43/50 (86.00%) | **48/51 (94.12%)** |
| Correctness losses | — | **0** |

The five correctness gains are `litter_case_12`, `litter_case_141`,
`litter_case_16`, `litter_case_78` and `litter_case_9`; the event-match gain is
`litter_case_12`. The exact paired two-sided sign-test diagnostic is `p=0.0625`
for five gains and zero losses, so the promotion gate correctly remains false.

The point target of 48/58 is reached, but the product 85% point gate requires
`ceil(0.85*58) = 50` clips. The candidate Wilson 95% interval is
`[0.7109, 0.9036]`; this is not a calibrated accuracy guarantee.

## Robustness checks

- Applying the same composition to nine existing confidence/scale/compression
  perturbation sidecars produced 48/58 and 51/58 event matches in every
  condition, with zero paired correctness or event-match losses. These sidecar
  tables are decision-stability checks on the same development cohort, not new
  camera data.
- Applying it to the five existing temporal offsets (`-2,-1,0,+1,+2`) produced
  `45,46,48,47,45` correct clips respectively, with zero losses relative to
  each offset baseline. The target is therefore sensitive to the release-time
  offset even though route changes remain safe in this screen.
- A local caller-parameter sensitivity grid around each rule's thresholds kept
  the same five gains and zero losses throughout the tested ranges. The smallest
  observed margins are approximately 0.011–0.012 in normalized feature units,
  so this does not justify claiming invariance to arbitrary upstream noise.
- All route changes were made in copied research sidecars. `trajectory.py`,
  `costs.py`, `resolver.py`, flow routes, confirmation logic, schemas and
  production configuration were not changed by this experiment.

## Interpretation and limitations

The case-level changes are mechanically auditable: each one names the route
family, the existing feature inequalities, the competing route, and the cost
delta. They are promising as a fine-tuning hypothesis, but the thresholds were
screened on the same 58-case development cohort. The 9/9 perturbation result
does not establish cross-camera generalization, false-positive precision, plate
OCR correctness, or enforcement readiness. There is no reviewed negative set,
camera/site-disjoint reviewed holdout, or reviewed plate ground truth; the
official readiness report remains `ready_for_enforcement: false`.

## Decision and next gate

Retain this candidate as a research artifact only. Before any production
consideration, freeze the rule text and thresholds, obtain one reviewer’s
camera/site/session/source-recording provenance for an independent holdout,
complete reviewed negative and plate-OCR labels, and rerun the same paired
evaluator without using those labels to select rules. A loss, unstable route,
or failure of the independent gates requires discarding this composition and
returning to the current production behavior.

## Rollback

No production rollback is required because no production file or default was
changed. Remove the temporary `/tmp/target48_*` research outputs if storage is
needed; retain this version note and the baseline candidate sidecar hashes for
provenance.
