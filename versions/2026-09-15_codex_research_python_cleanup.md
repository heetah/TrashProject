# 2026-09-15 — research Python cleanup

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree (no commit/push/deploy)
- Scope: repository hygiene; no production behavior change

## Problem

One-off development studies had accumulated in active `scripts/`, `tests/`
and historical artifact directories. They increased discovery noise and made
the maintained pipeline/test surface harder to distinguish from superseded
research. The target-48 result, Phase 0/1A primitives and the next holdout
gate must remain reproducible.

## Cleanup boundary

Before removal, 33 untracked/ignored Python source files were packaged into:

```text
artifacts/research_python_archive_20260915.tar.gz
SHA-256: f64ed8f4fe153b3157b2c06aaa809c81a1c7b93bc8ede80dc60d83d84e3b3a70
archive entries: 33
```

The archive contains the superseded target-41 evidence/certificate stack,
route-oracle and ablation helpers, RT-DETR intermediate probe, ordinal
occlusion builder, litter-cascade view builder, confirmation-prior probe, ten
associated focused tests and eight historical artifact-local helper scripts.
The original relative paths are preserved inside the archive. Restore only
into an isolated directory when historical reproduction is required:

```bash
mkdir -p /tmp/trashproject-research-restore
tar -xzf artifacts/research_python_archive_20260915.tar.gz \
  -C /tmp/trashproject-research-restore
```

All 344 generated `.pyc` files and ten empty `__pycache__` directories were
also removed. They are not archived because Python recreates them.

## Explicitly retained

- all Git-tracked production and project tests;
- target-48 metrics, version notes and presentation;
- Phase 0 observation provenance/lineage behavior and tests;
- Phase 1A `release_marginalization.py` and tests;
- production-used `mask_diagnostics.py` and tests;
- camera-holdout build/export/import/validation workflow and tests;
- release-validation package workflow and tests;
- replay-consensus primitive and tests;
- official readiness evaluator, comparator and live Smart Backtrack modules.

Active Python source count under `scripts/`, `tests/` and `artifacts/` changed
from 155 to 122. Active untracked Python files changed to 15, all belonging to
the retained research/holdout boundary. Ten superseded test modules containing
85 collected tests were removed; no tracked test was deleted.

## Documentation

Root and backtrack READMEs now label the removed entrypoints as historical and
point to the archive. Existing version documents were not rewritten; they
remain the immutable research record.

## Validation

- active Python import-reference scan: no remaining code imports an archived module;
- retained module `py_compile`: passed;
- `tests/pipeline`: **436 passed, 12 skipped**;
- `tests`: **463 passed, 12 skipped**;
- archive listing: 33/33 entries; SHA-256 verified;
- `git diff --check`: passed.

## Limitations and rollback

The archive is under `artifacts/` and remains a local research artifact rather
than a production dependency. Restoring its relative paths will bring back the
historical tools and tests but is not required to run production or the current
maintained test suite. No runtime/config/schema rollback is needed.
