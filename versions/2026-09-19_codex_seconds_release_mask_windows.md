# 2026-09-19 Codex — seconds-first release and mask windows

- Date: 2026-09-19
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `528ad79`
- Commit: uncommitted working tree
- Scope: production Smart Backtrack temporal policy

## Problem

The production release enumerator inherited a 24-frame-at-10-FPS research
span, later scaled to 2.4 seconds. That value was a computational convention,
not a reviewed physical bound. The promoted YOLO-Seg mask guard likewise used
fixed offsets `{-2,-1,0,+1}`, so its elapsed-time meaning changed with FPS and
detector cadence.

## Evidence and decision

Of the 58 reviewed positive events, 51 have usable release-to-first-visible
timing; seven legacy `0/1` sentinel rows are excluded. The usable distribution
has median 0.0 s, P90 0.5 s, P95 0.7 s and maximum 0.9 s. An earlier frozen
replay of a 1.0-second hard bound plus 0.25-second soft region gained two clips
and lost none. This supports a production candidate, not a universal optimum:
the cohort is positive-only, repeatedly examined and not camera-disjoint.

## Production behavior

Release hypotheses now use

```text
delta_t = max(0, (birth_frame - release_frame) / FPS)
C_time = 0                                      if delta_t <= 0.25
C_time = ((delta_t - 0.25) / (1.0 - 0.25))^2   if 0.25 < delta_t <= 1.0
reject                                             if delta_t > 1.0
```

`SMART_BACKTRACK_MAX_RELEASE_BACK_SEC=1.0` is authoritative. A blank
`SMART_BACKTRACK_MAX_BACK_FRAMES` derives `floor(FPS * 1.0)`; a smaller
explicit value is only a computational guard and records `search_truncated`.

Mask evidence uses fresh observed `seg_track`/`seg_predict` contours only:

```text
pre window:      [-0.20, 0) seconds
release window:  [0, +0.10] seconds
minimum fresh observations: 1 per window
maximum retained observations: 2 nearest release per window
```

For constant-FPS inputs the frame spans are `floor(FPS*T)`. The route sidecar
records both seconds and derived frame spans. Missing pre/release evidence
fails closed to the Min-Cost Flow assignment. NULL remains immutable; event
confirmation, route generation and hard spatial gates are unchanged. Existing
v1 fixed-offset evidence remains readable for old replays.

## Interface

```text
SMART_BACKTRACK_MAX_RELEASE_BACK_SEC=1.0
SMART_BACKTRACK_RELEASE_SOFT_SEC=0.25
SMART_BACKTRACK_MAX_BACK_FRAMES=
SMART_BACKTRACK_MASK_PRE_SEC=0.20
SMART_BACKTRACK_MASK_POST_SEC=0.10
SMART_BACKTRACK_MASK_MIN_PRE_OBS=1
SMART_BACKTRACK_MASK_MIN_RELEASE_OBS=1
SMART_BACKTRACK_MASK_MAX_PRE_OBS=2
SMART_BACKTRACK_MASK_MAX_RELEASE_OBS=2
```

The current implementation uses frame index divided by constant FPS. True VFR
robustness requires propagating frame PTS into resolver tasks and remains
unimplemented.

## Validation

- `py_compile`: production backtrack resolver, trajectory, route re-selection
  and litter tracker passed.
- Focused release/mask/integration/sidecar suite: `73 passed`.
- Complete repository pipeline suite: `454 passed, 12 skipped`.
- UI plus litter regression suite: `27 passed`.
- `git diff --check`: passed.

A repository-root pytest invocation also collected vendored `mmaction2/tests`
and stopped on 92 missing optional-vendor dependencies (`mmaction`,
`parameterized`, `decord`). The repository-owned suites above completed; no
dependency was downloaded or runtime silently changed.

The retained 58-case replay artifacts do not contain the fresh per-frame mask
contours needed to reconstruct the new seconds windows. The earlier 48/58
result belongs to the fixed-offset policy and must not be attributed to the
seconds-based mask policy without a new paired end-to-end run.

## Rollback

Revert the changes in `resolver.py` and `route_reselection.py`, restore the
2.4-second computational default and v1 fixed offsets, and restore the related
environment documentation. Do not remove reviewed ground truth or historical
research artifacts.
