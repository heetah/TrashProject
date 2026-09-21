# JEV API Smart Backtrack reranking evaluation

- Date: 2026-09-21
- Author: Codex
- Branch: `heetah-dev`
- Commit: working-tree research evaluation; no production commit

## Problem

Determine whether a TypeSafe/JEV API call can choose the violator from frozen
Smart Backtrack release-hypothesis evidence, improve route precision/recall/F1,
and preserve frame throughput.

## Change

Added the research-only evaluator
`scripts/evaluate_jev_backtrack.py`. It summarizes local route candidates into
structured text and sends at most the top eight routes plus `NULL` to
`POST https://api.typesafe.ai/v1/systemone`. It does not upload frames, masks,
raw trajectories, or credentials. Invalid or failed responses fall back to the
local assignment. No production pipeline default or route solver was changed.

## Frozen evaluation

- Input: `artifacts/backtrack_testcase_production_20260919/` and its 58 reviewed
  route-readiness rows; the production sidecars contain 268 candidate records.
- JEV calls: 44/44 confirmed-event matches; median latency 0.625 s, p95 0.697 s.
- Baseline route correctness: 35/58 (60.34%).
- Raw and guarded JEV route correctness: 28/58 (48.28%). JEV improved 0 cases
  and regressed 7; the two baseline wrong-route cases were not repaired.
- On the mapped accepted-event subset, the exploratory precision/recall/F1
  proxy changed from 94.59%/94.59%/94.59% to
  82.35%/75.68%/78.87%.
- Frozen production video-level event metrics remain upstream and unchanged:
  precision 93.18%, recall 67.77%, F1 78.47% on 406 cases.
- Aggregate production loop throughput was 16.71 FPS. A synchronous API-call
  extrapolation is 16.06 FPS; asynchronous queuing preserves the steady-state
  loop estimate but adds result-drain latency.

Full machine-readable output is in
`artifacts/jev_backtrack_evaluation_20260921/`.

## Validation

```text
conda run --no-capture-output -n rtdetr python -m py_compile scripts/evaluate_jev_backtrack.py
```

The evaluator completed 44 successful API calls. Repeated calls on the nine
changed cases returned the same route choice twice per case, so the observed
regressions were not a single-response parsing accident.

## Limitations and rollback

The reviewed set is positive-only and from one camera; route precision/recall/F1
are explicitly exploratory and do not certify enforcement accuracy. JEV is
text-only in this integration and cannot recover an event missed by the local
tracker. The API key is read only from an environment variable and is not
stored in code or artifacts. Rollback is deleting the research evaluator and
artifact directory; production behavior is already unchanged.
