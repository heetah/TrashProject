# Smart Backtrack route-error ablation

- Date: 2026-09-12
- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree based on `a149c5975898643e5131caf0579bd0aefe228d75`
- Type: research tooling and frozen development validation

## Problem

The current 58-case run produced 27 correct end-to-end routes, 8 wrong routes,
one unavailable actor mapping, five exploratory event matches, and 17 missed
events. The task was to determine whether existing `C_AC` overlap and `C_BC`
motion diagnostics can improve Smart Backtrack ranking without introducing
new wrong attributions.

## Changes

- `scripts/pipeline/backtrack/study.py` can now load either one sidecar or a
  recursively isolated production batch directory.
- `scripts/backtrack_study.py` uses that loader, avoiding temporary flattening
  of per-case evidence.
- `scripts/report_backtrack_route_ablation.py` produces route-level cost/error
  decomposition and a consolidated frozen-ablation report.
- A regression test covers recursive sidecar discovery.

No production resolver weight or event-confirmation rule was changed.

## Evidence

The protocol, input hashes, 14 predeclared configurations, 65-event replay
outputs, per-config 58-case readiness reports, paired comparisons, and final
decomposition are under `artifacts/backtrack_route_ablation_20260912/`.

- Baseline: 27/58 end-to-end; 27/36 given an accepted event.
- Best exploratory settings: 28/58; 28/36.
- `AC overlap=0.25` and `0.50` each fix case 16 with no observed loss.
- `BC reverse=0.50` fixes case 194 with no observed loss.
- Every exact two-sided paired p-value is 1.0; no setting passes promotion.
- `BC exit` and relative-motion terms yield no gain at 0.05, 0.20, or 0.50.
- Targeted Smart Backtrack tests: 26 passed.

## Limitations

Event labels remain marked `unreviewed`; the set is positive-only and lacks an
independent camera-group split. The maximum end-to-end result attainable by
reranking only the 36 accepted events is 36/58 (62.07%), so backtracking alone
cannot reach the 85% product target.

## Rollback

Remove the recursive study loader, report script, its test, and this note.
Research artifacts are ignored and can be discarded independently. Production
behavior requires no rollback because no production defaults changed.
