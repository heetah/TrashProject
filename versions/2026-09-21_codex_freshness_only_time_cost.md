# 2026-09-21 Codex — freshness-only Smart Backtrack time cost

- Date: 2026-09-21
- Author: Codex
- Branch: current working branch
- Commit: uncommitted working tree
- Scope: production Smart Backtrack route scoring

## Request and decision

Production route ranking now keeps only the BA/BC actor-observation freshness
time cost. The backward release-time quadratic prior and `C_AC`
person–vehicle synchronization-time cost have zero production weight.

Release hypotheses are not removed: the one-second physical search bound,
trajectory position/velocity/covariance, model diagnostics, spatial and
uncertainty gates, Min-Cost Flow and complete NULL route remain. `C_AC` also
retains its spatial, quality, uncertainty, dwell and continuity evidence.
Two-point/fallback model-quality priors remain active because they are not time
costs. Existing raw time/prior sidecar fields remain for schema compatibility
and historical replay.

## Evidence boundary

Two paired 58-clip resolver-input replays were performed before promotion:

- 2026-09-13 frozen inputs: removing the release-time quadratic changed
  `33/58` to `34/58`, with three gains and two losses; removing only `C_AC`
  synchronization time changed no decision.
- 2026-09-19 production snapshots first showed that removing only the backward
  quadratic kept `31/58`, with two gains and two losses; removing only `C_AC`
  synchronization time again changed no decision. The final exact policy also
  removes the former post-birth forward-time penalty while preserving
  two-point/fallback model-quality priors. Against the same 127 frozen records,
  legacy time costs scored `31/58` and freshness-only scored `32/58`: gains in
  cases 30, 41 and 74; losses in cases 7 and 60. Accepted-event matches changed
  `38/58 -> 39/58`, with gains in cases 135 and 74 and a loss in case 60.

These are development-set ablations, not the archived 43/58 or 48/58 research
composition and not a camera-disjoint accuracy estimate. The release-prior
change is therefore a user-directed simplification, not a demonstrated
accuracy improvement. The recurrent losses in cases 7 and 60 remain a known
regression risk.

## Interface and rollback

No sidecar field is removed. Historical replay can restore a positive
`release_time_weight` or explicit research-stage time weights. Roll back by
restoring production `release_time_weight=1.0` and `C_AC time=0.2`.

## Validation

- Focused release/cost/study/integration/reselection tests: `120 passed`.
- Complete repository pipeline suite: `477 passed, 12 skipped`.
- Production backtrack modules passed Python compilation.
- `git diff --check`: passed.
