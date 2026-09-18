# 2026-09-14 Codex — Smart Backtrack observation lineage Phase 0

- Date: 2026-09-14
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `a149c5975898643e5131caf0579bd0aefe228d75`
- Commit: uncommitted working-tree change

## Problem

`GlobalLitterTracker` already recorded accepted history sources as `detector`
or `visual_bridge`. `_submit_backward_resolution` rebuilt the public
`history_sources` array only from recovered frame membership, mapping every
non-recovered point to `accepted_tracker`. That compatibility mapping erased
the distinction between real RT-DETR measurements and derived visual bridges,
so later research could incorrectly count correlated observations as
independent evidence.

## Change

- Preserve the existing `history_sources` compatibility values unchanged.
- Add aligned `history_provenance` with exact sources: `detector`,
  `visual_bridge`, `raw_rtdetr_recovered`, and `legacy_unknown` when an old
  caller supplied no source information.
- Add aligned `history_lineage` with deterministic observation identity,
  parent identity, independence group, derived flag and independent-measurement
  flag. Visual bridges inherit their parent detector family; recovered raw
  prefixes are conservatively tied to the first accepted anchor. Duplicate
  observation identities are never labeled as additional independent evidence.
- Sort all parallel history arrays together. Raw-prefix insertion reconstructs
  provenance by frame without overwriting accepted detector/visual-bridge
  sources.
- Fail closed on partially populated/misaligned source arrays. Fully old tasks
  remain replayable with `legacy_unknown`; missing optional boxes are aligned
  with `null` without enabling raw recovery, and missing confidence uses the
  resolver's pre-existing `1.0` fallback.
- Add provenance and lineage to each `litter_history` sidecar sample while the
  complete additive fields remain in `resolver_input`.

## Decision-equivalence argument

Let `T` be the original resolver task and `T' = T union {history_provenance,
history_lineage}`. Let projection `P(T') = T` remove only the two new fields.
The live resolver reads `history`, `history_frames`, `history_confidences`, actor
frames and existing configuration, but never either new field. Therefore:

```text
release(T') = release(P(T'))
costs(T')   = costs(P(T'))
route(T')   = route(P(T'))
NULL(T')    = NULL(P(T'))
```

An exact regression test compares selected route/person/vehicle and every route
cost for the same task with and without Phase 0 fields. No calibrated parameter,
gate, event confirmation rule, release model or fallback changed.

## Interface and configuration

The immutable task and optional research sidecar gain additive diagnostic
fields only. `schema_version` and `smart-backtrack-candidates/v1` remain stable
for backward compatibility. No environment variable or CLI option was added.

## Test evidence

- Targeted smart-backtrack integration and sidecar tests cover detector-only,
  visual bridge preservation, raw-prefix alignment, old-source compatibility,
  partial-source fail-closed behavior, duplicate lineage identity, strict JSON
  sidecar output and resolver decision/cost equivalence.
- Targeted tests: 29 passed.
- Full `tests/pipeline`: 402 passed, 12 skipped.

## Limitations

- Lineage expresses measurement dependence conservatively; it is not a learned
  probability or proof that observations are the same physical object.
- Old tasks without source metadata cannot recover lost provenance and remain
  `legacy_unknown`.
- The resolver intentionally ignores lineage in Phase 0. Release-state and
  multi-hypothesis smoothing remain unimplemented research work.
- Unit equality proves the new fields are ignored by the resolver; full video
  accuracy and production readiness are outside this serialization-only phase.

## Rollback

Remove the two additive task fields, their `litter_history` serialization, the
focused tests and documentation. No cost/config rollback or artifact migration
is required because existing fields and schemas were not changed.
