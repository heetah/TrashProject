# Quarantine evidence reason module

- Date: 2026-09-19
- Author: Codex
- Branch: `heetah-dev`
- Base: working tree after `event_confirmation_module`
- Status: implementation in working tree; not committed

## Problem

Vehicle-contained litter tracks already stored relative displacement and downward
evidence, but pending output did not identify which release gate failed. This made
the tracker/quarantine false-negative bucket hard to sample and compare.

## Change

- Added `scripts/pipeline/quarantine_evidence.py`.
- Added deterministic, unit-aware reason classification.
- Integrated reason/status into existing `vehicle_quarantine_evidence`.
- Preserved all state transitions and thresholds.
- Added explicit reasons for non-contained, missing carrier, not-pending, and
  already-released states.

## Validation

- Focused quarantine/tracker/event suite: `80 passed`.
- Sampled scenarios:
  - same-carrier co-motion → active, `relative_displacement_insufficient`;
  - two observations → active, `insufficient_observations`;
  - three-point relative descent → inactive, `released`;
  - ordinary non-contained candidate → inactive, `not_contained`.
- No threshold or model-weight change.

## Limits

- Reason precedence is first-failing gate order, not a calibrated probability.
- Carrier-ID mismatch with a recoverable saved track is not yet a separate reason.
- Real video sample replay remains unavailable because `resources/` has no clips.

## Rollback

Remove the quarantine-evidence import/call and retain original inline release
boolean; no model weights or output artifacts changed.
