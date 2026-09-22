# 2026-09-15 Codex — target-41 replay consensus gate

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working-tree change
- Scope: research-only replay audit; no production caller

## Problem

The mask-temporal replay was stable across nine within-development-set
conditions, but a route that happens to win one run must not be treated as
robust merely because it is a top-1 result. The comparison must be explicit
about perturbation agreement and must fail closed when conditions disagree.

## Change

Added `scripts/pipeline/backtrack/replay_consensus.py`:

- Canonicalizes a route by `(route_type, person_key, vehicle_key)` and does not
  use local `route_id` values as identity across reruns.
- Requires unique condition IDs and strict route/actor-key shapes.
- Returns `consensus_route` only for unanimous agreement. A disagreement
  returns the explicit `NULL_ROUTE` as `safe_route`, records every observed
  route and route ID, and requires manual review; it never chooses a modal or
  top-1 route.
- Preserves unanimous NULL as a real, auditable consensus.
- Sorts condition rows deterministically and performs no score, likelihood,
  lineage, cost, release or resolver operation.

## Production boundary

This is a standalone mathematical/replay diagnostic with no import from the
production resolver and no production caller. The production route, cost,
release, confirmation, NULL and schema behavior is bit-for-bit unchanged.
The 58/58 route-tuple agreement observed in the nine replay conditions remains
an in-sample decision-invariance result, not calibrated accuracy or
camera-independent robustness.

## Test evidence

Focused tests cover unanimous routes with changing local IDs, disagreement
fail-closed behavior (including NULL), unanimous NULL, deterministic ordering,
missing/duplicate condition IDs, malformed routes and flat JSON/CSV adapter
rows. The full suite and `git diff --check` are run after this change.

Observed validation: focused consensus tests `14 passed`; full repository
suite `470 passed, 12 skipped`; `git diff --check` passed.

## Limitations and next gate

The primitive cannot establish that an agreed route is correct, independent,
or physically depth-aware. The next release gate remains a single-reviewer
confirmation of camera/site/source groups and event truth, followed by a
frozen camera-disjoint paired evaluation. Until that evidence exists, the
target-41 candidate stays research-only and the production resolver is not
changed.

## Rollback

Remove the standalone module, its focused tests and this version note. No
runtime migration or production rollback is required because there is no
production caller.
