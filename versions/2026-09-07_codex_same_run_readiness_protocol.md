# Same-run fail-closed readiness protocol

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da6e1ebc19c2ca5c9ed66d99ecdc35d8607`
- Commit: working tree, not committed by Codex

## Problem

The historical actor report compared numeric tracker IDs with a table copied
from an earlier run. Tracker IDs are run-local, so applying that table to a new
inference run can mark correct routes wrong or wrong routes correct. It also
counted any confirmed event in a positive clip as detection coverage even when
the predicted release was too far from the annotated event. These properties
made the previous 33/58 end-to-end interpretation ineligible for an 85%
product-readiness claim.

## Changes

- Added `scripts/evaluate_reviewed_readiness.py`.
  - Recursively reads production batch sidecars.
  - Uses the same run's actor frames and same-frame/same-class IoU mapping.
  - Accepts only strict or moderate event matches.
  - Keeps miss, exploratory, unmapped, NULL, wrong, and failed evidence in the
    fixed positive denominator.
  - Reports Wilson 95% intervals, annotation states, hashes, and explicit
    negative/cross-camera release blockers.
  - Supports a frozen-input research replay for a specified release-back limit;
    this never promotes the setting to production.
- Added `scripts/compare_readiness_runs.py` for paired gain/loss analysis,
  exact two-sided sign tests, and a no-regression promotion gate.
- Made recursive candidate discovery part of
  `scripts/validate_attribution_formula.py` so production `case_<id>/` layouts
  require no temporary flattening.
- Marked `summarize_actor_ground_truth_metrics.py` and
  `replay_release_policy.py` as legacy run-local-ID diagnostics. The former now
  requires `--acknowledge-run-local-ids` at its CLI boundary.
- Updated root and Smart Backtrack documentation and added unit tests.
- Tightened vehicle-track confirmation: the general physics trajectory path
  now also requires either the existing size-aware absolute displacement or
  explicit same-actor release evidence (start near actor and end clear of it).
  The dedicated two-point fast-drop route remains unchanged. This resolves the
  vehicle-edge old-litter and three-frame tiny-noise safety regressions without
  globally raising thresholds or weakening real person-throw logic. A targeted
  replay retained the prior event counts for cases 9, 17, 41, 116, 159 and 164.

## Interface and configuration

Baseline evaluation:

```bash
conda run -n rtdetr env PYTHONPATH=scripts python \
  scripts/evaluate_reviewed_readiness.py \
  --ground-truth runs/grounding_truth \
  --candidates output/groundtruth_batch_dynamic_h_20260907 \
  --output artifacts/product_readiness_same_run_20260907 \
  --target 0.85
```

Research-only replay adds:

```text
--replay-max-release-back-seconds 1.0
```

No production environment default was changed.

## Evidence

Baseline, fixed denominator of 58 usable positive clips:

- Strict/moderate event sensitivity: 32/58 = 55.17%; Wilson 95% CI
  42.45%--67.25%.
- Provisional exact end-to-end route: 23/58 = 39.66%; Wilson 95% CI
  28.09%--52.51%.
- Conditional exact route among accepted events: 23/32 = 71.88%; Wilson 95% CI
  54.63%--84.44%.
- Outcome split: 23 correct, 8 wrong, 1 actor mapping unavailable, 9
  exploratory event matches, 17 misses.

Frozen-input release-window ablations:

- 1.0 seconds: 25/58; paired correctness gain/loss 3/1, exact two-sided
  `p=0.625`; event-match gain/loss 2/1. Promotion gate failed.
- 0.5 seconds: 23/58; paired correctness gain/loss 2/2, exact two-sided
  `p=1.0`; event-match gain/loss 1/1. Promotion gate failed.

Target requirements at `n=58`:

- 85% point estimate: at least 50/58.
- two-sided Wilson 95% lower bound at least 85%: at least 55/58.

Unit and compile evidence is recorded in the task handoff. Machine-readable
reports are under:

- `artifacts/product_readiness_same_run_20260907/`
- `artifacts/product_readiness_release1s_same_run_20260907/`
- `artifacts/product_readiness_release05s_same_run_20260907/`
- `artifacts/readiness_comparison_release1s_20260907/`
- `artifacts/readiness_comparison_release05s_20260907/`

The focused litter regression set passes 55/55 after the vehicle-motion guard;
the full project-owned `tests/` suite passes 351 with 12 skips. The final
isolated runtime batch completed 58/58 with no process failure; all 58 MP4
streams passed `ffprobe`.

The production pipeline subset passes 324 with 12 skips. Production/evaluation
files pass `py_compile`, and the complete working-tree diff passes
`git diff --check`.

Final safety-guard result:

- Strict/moderate event sensitivity: unchanged at 32/58.
- Provisional exact route: unchanged at 23/58.
- Paired correctness and accepted-event gains/losses: 0/0 and 0/0.
- One exploratory confirmation (`litter_case_42`) was removed and none added.
- Development safety-change gate passed; accuracy-improvement gate did not pass.
- Confirmed candidate records changed 62→61.

The exploratory removal is a safety proxy, not an adjudicated false-positive
reduction. Final reports are under:

- `artifacts/groundtruth_batch_vehicle_evidence_final_20260908/`
- `artifacts/product_readiness_vehicle_evidence_final_20260908/`
- `artifacts/readiness_comparison_vehicle_evidence_final_20260908/`

## Limitations and release decision

- All 58 event annotations are `unreviewed`; results are provisional.
- There is no reviewed negative/background set, so precision and false-positive
  rate cannot be estimated.
- There is no independent camera-group holdout, so cross-site generalization is
  unknown.
- There is no reviewed plate/OCR ground truth. Route correctness therefore does
  not establish finable-case correctness.
- Neither the baseline nor either release replay satisfies the 85% criterion.
  The system is not ready for unattended enforcement or automatic penalties.

## Rollback

Remove the two new evaluation scripts and their tests, revert the documentation
and legacy warning additions, and revert recursive `rglob` discovery to `glob`.
No production inference setting or persisted user data was changed.
