# Event confirmation module

- Date: 2026-09-19
- Author: Codex
- Branch: `heetah-dev`
- Base: working tree after `fresh_actor_gate`
- Status: implementation in working tree; not committed

## Problem

`GlobalLitterTracker` had four confirmation paths and a vehicle-quarantine veto
embedded in a long update function. A pending track exposed no stable reason code,
making sampled false-negative triage difficult.

## Change

- Added `scripts/pipeline/event_confirmation.py`.
- Moved final boolean combination into deterministic
  `evaluate_litter_confirmation()`.
- Preserved existing rule order and thresholds.
- Preserved quarantine as hard veto.
- Added per-track `confirmation_evidence` with rule, reason, and four evidence flags.
- New tracks start with `insufficient_history`; matched tracks refresh evidence each
  pending update.

## Validation

- Focused confirmation/tracker/characterization suite: `99 passed`.
- Sampled synthetic replay:
  - moving litter + person → `confirmed`, `by_trajectory`;
  - static jitter → `pending`, `insufficient_confirmation_evidence`;
  - actorless gravity arc using existing sparse-frame fixture → `confirmed`,
    `by_trajectory`.
- A first compressed-frame actorless sample failed existing physics gates; frame-gap
  restoration passed without threshold changes.

## Limits

- This module is an audit/refactor layer; no threshold tuning or recall claim.
- Real input videos are absent from `resources/`; no new clip replay was run.
- Quarantine sub-reasons remain inside tracker and are next triage target.

## Rollback

Remove `event_confirmation.py` import and restore the original final OR condition;
no model weights or output artifacts changed.
