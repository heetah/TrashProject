# Release hypothesis identifiability audit

- Date: 2026-09-13
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working tree based on `a149c5975898643e5131caf0579bd0aefe228d75`

## Objective

Continue release-hypothesis research after the normalized soft-min and robust
zero-cost-window rules were rejected. The next question is whether each event
contains enough observation evidence to identify a pre-birth release state.

## Change

- Added `scripts/analyze_release_identifiability.py`.
- Added focused tests in `tests/pipeline/test_release_identifiability.py`.
- Audited the frozen baseline sidecar from
  `artifacts/backtrack_multiview_hypotheses_20260913`.

The tool is diagnostic-only. It never consumes reviewed route labels, never
changes resolver costs, and never promotes an event. It reports unique observed
frames, hypothesis frame span, covariance-derived pixel sigma, search truncation,
route margin, and a conservative identifiability class.

## Result

- Resolver records read: 64.
- Non-identifiable two-point histories: 36.
- Longer histories with a computationally truncated reverse search: 28.
- No record was classified as safely identifiable for production promotion.

The two-point result follows directly from model identifiability: infinitely many
smooth pre-birth trajectories can pass through the same two observations. A
computationally truncated search is also not proof that the physical hypothesis
space has been exhausted.

## Decision

Do not add another release aggregation weight. The next candidate must be a
pre-registered uncertainty/abstention policy, evaluated with independent camera
groups and the complete 58-clip denominator. Abstention can improve safety and
precision, but it does not count as a correct route and cannot be used to inflate
the 85% accuracy metric.

## Validation

- Focused tests: `2 passed`.
- Full pipeline suite after this addition: `356 passed, 12 skipped`.
- Python compilation and `git diff --check`: passed.

## Artifact

The reproducible output is
`artifacts/release_identifiability_20260913/` with `summary.json`, `events.csv`,
and `REPORT.md`.
