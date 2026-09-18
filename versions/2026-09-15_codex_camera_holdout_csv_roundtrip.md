# 2026-09-15 — Codex camera-holdout CSV round trip

- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree (no commit/push/deploy performed)
- Scope: research-only reviewer data handoff for the 18-case camera-safe queue

## Problem

The single reviewer needs to record camera/site/session provenance and event
truth without opening model route scores or other prediction outputs. Editing a
large JSON worksheet directly is error-prone, while the later camera-disjoint
gate must reject changed source identity, duplicate cases, and incomplete
review fields.

## Change

- Added deterministic `export_review_table` and strict `import_review_table`
  helpers to `camera_holdout_review.py`.
- Added `scripts/export_camera_holdout_review.py` and
  `scripts/import_camera_holdout_review.py`.
- The CSV header is exact and contains only immutable identity plus explicit
  review/provenance fields and `events_json`/`notes`; model summaries, route
  IDs/scores, release hypotheses, and annotated output paths are not exported.
- Import merges into the JSON template only after exact case membership and
  path/SHA-256 identity checks. Existing output files are never overwritten.
- Partial import is permitted for saving reviewer progress, but it remains
  `ready_for_camera_disjoint_eval=false` until the existing completion gate
  receives all required human fields and independently supplied groups.

No production resolver, cost, route, schema, configuration, or model behavior
changed. The CSV helpers have no production caller and do not create labels or
infer camera identity.

## Validation

- `tests/pipeline/test_camera_holdout_review.py`: **27 passed**.
- `tests/pipeline`: **475 passed, 12 skipped**.
- `tests`: **502 passed, 12 skipped**.
- `python -m py_compile` for the modified module and four holdout CLIs: passed.
- The 18-case worksheet was exported and imported with source rehashing:
  **18/18 source files verified**, JSON semantic equality `true`, and all
  cases remain `unreviewed`.
- `git diff --check`: passed.

The repository-wide default `pytest -q` still attempts the vendored
`mmaction2/tests` suite and stops during collection because this environment
lacks optional `parameterized`, `mmaction`, and `decord` dependencies. This is
an environment limitation, not a failure in the project pipeline tests.

## Limitations and next step

No human camera/site/source-recording labels have been supplied. Therefore the
camera-disjoint evaluation remains blocked and the 41/58 target is not yet an
independent generalization result. Hand the CSV to the one reviewer, import
the completed copy, run `validate_camera_holdout_review.py`, and only then
freeze a camera-disjoint paired evaluation.

## Rollback

Because production code is unchanged, rollback is to stop using the two CSV
commands and retain the original JSON worksheet; remove/exclude these
research-only helpers and this version note if the handoff format is rejected.
