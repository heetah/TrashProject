# 2026-09-15 Codex — target-41 evidence ledger

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working-tree change
- Scope: research/reporting only; no production caller

## Purpose

The 43/58 mask-temporal result has to remain explainable and reproducible
without confusing route changes, event matches, replay consistency or model
confidence with ground truth. Existing readiness CSV/JSON outputs were valid
but scattered across several artifacts.

## Change

Added `scripts/pipeline/backtrack/target41_evidence.py` and the CLI
`scripts/build_target41_evidence_ledger.py`. They:

- require the official `reviewed-product-readiness/v1` reports and identical
  case tables;
- enforce reviewed event labels, a caller-supplied denominator and target count,
  strict boolean fields, supported outcome semantics, and report/table count
  agreement;
- preserve baseline/candidate route tuples, per-case outcome transitions and
  explicit explanation codes (`correct_route`, `wrong_route`, `null_route`,
  `actor_mapping_unavailable`, `exploratory_event_match`, `missed_event`);
- verify the caller-supplied `target41-mask-decision-consensus/v1` report covers
  every denominator case and has no unstable cases;
- record SHA-256 digests of every input and state that the target is a point
  estimate only.

The ledger has no detector, tracker, resolver, cost, route-schema or UI caller.
It does not rerun event matching, infer labels, select a route, or claim that
replay consensus proves correctness.

## Frozen result

Using the `{0,+1}` mask replay:

- baseline: 38/58;
- candidate: 43/58;
- paired changes: five `wrong_route -> correct_route`, zero losses;
- unchanged cases: 53/58;
- replay consensus: 9 conditions, 58/58 cases, zero unstable cases.

The generated artifact is `/tmp/target41_evidence_ledger.json`. Its per-case
rows retain route/mapping fields and explain why each row is correct, wrong,
exploratory, missed, or unmapped; they do not include raw model scores.

The existing route-ablation Markdown helper was also corrected to derive the
accepted-event ceiling, best-trial gain/loss count and paired p-value from its
live inputs, and to remove a stale claim that the current event labels remain
unreviewed. This is a reporting correction only.

## Test evidence

The new ledger/report tests pass `5`. The focused camera/replay/release set
passes `61`; the full repository suite passes `490` with `12` skips; `py_compile` and
`git diff --check` pass.

## Limitations and next gate

This is still one reviewed positive development cohort. It cannot estimate
precision, false positives, OCR/finable correctness, or camera-independent
generalization. The 18-case camera-safe worksheet remains unreviewed; after a
human supplies camera/site/source truth, run the completed-review gate and a
frozen camera-disjoint paired evaluation.

The 18-case runtime sidecars contain 29 dynamic-H snapshots, but zero were
applied and all 29 fell back with `feature_disabled`; the ledger therefore
contains no implicit pseudo-homography/depth result.

## Rollback

Remove the standalone ledger module, CLI, focused tests, README section and
this note. No production rollback is required because no runtime caller exists.
