# 2026-09-13 Codex — diagnostic-only mask/litter evidence

- Date: 2026-09-13
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `a149c5975898643e5131caf0579bd0aefe228d75`
- Commit: uncommitted working-tree change

## Problem

Bounding-box proximity can confuse overlapping foreground and background
vehicles. Direct temporal occlusion scoring was not justified because visible
instance masks are not amodal shapes and the reviewed dataset has no independent
front/behind labels.

## Change

Added a default-off research recorder under
`scripts/pipeline/backtrack/mask_diagnostics.py`. It summarizes same-frame raw
litter boxes against actual observed vehicle/scooter segmentation polygons and
stores scalar evidence only. Cached/Kalman observations, invalid masks and
unsupported classes/sources are rejected. The resolver does not read the new
frame-level field and its weight is fixed at zero.

The diagnostic formulas and experiment gates were locked before execution in
`artifacts/mask_litter_diagnostics_20260913/protocol.json`.

## Evidence

- Unit/integration checks initially passed: 26 tests.
- Diagnostic and validator targeted checks passed: 14 tests before the
  validator's full-cohort-only Markdown renderer rejected the two-event sample;
  generated mapping CSV/JSON remained usable.
- Case 9 skip-2 smoke completed; frame 40 wrong vehicle bbox containment was
  about 0.90 while visible-mask overlap was 0.
- Cases 25/67/168 skip-1 runs completed with valid MP4 output. Case 25 produced
  no confirmed event, so it cannot act as an attribution control in this run.
- Same-run IoU mapping identified case 67 GT as vehicle 2 and case 168 GT as
  vehicle 1.
- Case 67 selected vehicle 2 at frame 105 after cadence changed, but all
  vehicle mask-overlap values were 0; no mask-caused gain is claimed.
- Case 168 selected wrong vehicle 2; its release-frame litter box had
  mask-overlap 1.0 with vehicle 2 and 0 with reviewed vehicle 1. A single-frame
  visible-mask rule would therefore reinforce the wrong attribution.

## Limitations and next gate

This work does not change the 58-case accuracy and does not estimate metric or
ordinal depth. Before temporal occlusion may affect a cost, a single human
reviewer must add model-blind `ordinal_front_key`, `ordinal_behind_key`, reviewed
time window and unknown/occlusion-onset status. Release reliability and ordinal
ordering must be evaluated independently and fused only after both pass.

## Rollback

Set `SMART_BACKTRACK_MASK_DIAGNOSTICS=0` (the default) for the exact prior
serialized schema. Removing the module import, flag, scalar construction and
conditional frame-row copy fully removes the research path. No route setting or
production default requires rollback.
