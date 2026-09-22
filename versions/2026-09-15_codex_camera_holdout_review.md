# 2026-09-15 Codex — single-reviewer camera holdout worksheet

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working-tree change
- Scope: research/data-audit tooling; no production caller

## Problem

The 18-case source-disjoint queue is ready for human review, but its existing
index also contains model predictions. A reproducible handoff must prevent
those predictions from being mistaken for truth and must verify source
provenance before a camera-disjoint paired evaluation.

## Change

Added `scripts/pipeline/backtrack/camera_holdout_review.py`:

- `build_from_review_index` and `build_single_reviewer_queue` create a
  deterministic `camera-holdout-review/v1` worksheet for exactly one reviewer.
  Only case/source filename/path/SHA-256 identity is copied; model summaries,
  assignments, route scores, release hypotheses and output videos are omitted.
- `validate_unreviewed_queue` enforces a blank review object and the
  model-blind boundary.
- `verify_source_files` recomputes each source SHA-256 and fails closed on a
  missing or changed file.
- `validate_completed_review` requires reviewed status, clip truth state,
  source-hash confirmation, camera/site/session/source-recording IDs and
  explicit timestamps. It reports (rather than invents) whether the caller's
  reviewed groups are disjoint and whether positive/negative coverage is ready
  for evaluation.

Added `scripts/build_camera_holdout_review.py` as a no-overwrite CLI wrapper.
It can rehash all sources before writing the model-blind worksheet, so the
reviewer receives a concrete, reproducible file rather than a hand-copied
template.

Added `scripts/validate_camera_holdout_review.py` as a fail-closed gate. A
blank worksheet exits with a machine-readable
`ready_for_camera_disjoint_eval: false` report; a partially completed or
malformed worksheet is rejected. A completed worksheet exits successfully
only when the caller supplies the independently reviewed camera/source-
recording groups and all provenance and positive/negative coverage checks
pass. This gate does not infer camera IDs, run evaluation, or alter production
behavior.

## Production boundary

The module has no production import or caller. It does not infer camera IDs,
fill event/route labels, change the resolver, alter costs or NULL behavior, or
calculate accuracy. The existing 43/58 result and nine-condition consensus
remain development-set evidence only.

## Test evidence

Focused tests cover deterministic/model-blind queue creation, index schema
checking, source rehash mismatch, blank-queue leakage rejection, completed
review readiness, missing provenance, overlap/ambiguous blocking, duplicate
case/SHA rejection, the fail-closed CLI blank-queue gate, and the existing
replay-consensus primitives. The CLI was run with source rehash against the
18-case index and completed without error (`18/18` sources verified); the
validation command then correctly reported the still-blank worksheet as not
ready. The focused camera/replay/release set passes `56`; the full suite
passes `485` with `12` skips.

## Limitations and next gate

The worksheet cannot decide camera/site identity or event truth. A reviewer
must fill the 18 cases from the original source videos, then supply the
reviewed camera/source group sets and corresponding event/actor annotations.
Only a completed, disjoint, reviewed set can support the frozen paired
evaluation; until then the candidate remains research-only.

## Rollback

Remove the standalone module, focused tests and this note. No runtime migration
or production rollback is required because no production caller exists.
