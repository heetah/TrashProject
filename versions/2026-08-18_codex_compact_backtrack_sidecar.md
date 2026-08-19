# Compact Smart Backtrack sidecar

- Date: 2026-08-18
- Author: Codex
- Branch: current working branch
- Commit: uncommitted

## Problem

Research sidecars serialized `plate_actor_frames` and NumPy `plate_roi` crops
inside `resolver_input`. Smart Backtrack does not consume those OCR-only
buffers; JSON expansion could grow one clip to several GB.

## Change

- Exclude top-level `plate_actor_frames` and nested `plate_roi` values from
  replay input and candidate actor observations.
- Preserve litter history and actor geometry, IDs, confidence, observed/source
  fields needed to reproduce Smart resolver output.
- Keep runtime OCR and legacy fallback tasks unchanged.

## Validation

- Targeted sidecar unit test verifies no ROI pixels are emitted, geometry stays
  replayable, strict JSON remains below 10 KB for a synthetic large ROI.
- Existing replay test remains responsible for deterministic resolver output.

## Limits and rollback

This changes research sidecar serialization only. It does not alter detector,
confirmation, attribution costs, Min-Cost Flow, OCR, or frontend analysis JSON.
Rollback by reverting the sidecar serializer, its tests, and documentation.
