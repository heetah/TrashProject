# 2026-09-15 — Target-48 blind source-recording preflight

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree (no commit/push/deploy)
- Scope: research-only Smart Backtrack replay and provenance/output audit
- Production caller: none

## Purpose

Before asking the single reviewer to supply camera/site/session provenance,
run the already-frozen Target-48 guarded route composition on the 18-case
blind runtime package. This is a behaviour and provenance preflight only. It
must not use the existing development labels to tune or score the holdout.

## Inputs and provenance

- Frozen research rule source: `/tmp/build_combined_candidate.py`
  (SHA-256 `cde63b3d22fa9a7252186fb159b4f05eb75af271b90cccdb1be22e32e953e652`).
- Blind run manifest: `/tmp/target41_camera_safe_run_manifest.json`
  (SHA-256 `d4487cfd0ebfc7bd35113c8f39af1d78ff91614689d77a01d6a766ea529e8626`).
- Model-blind review index: `/tmp/target41_camera_safe_review_index.json`
  (SHA-256 `5fe0fab1d542ec0d65b9587d1c9d29218a63999166a307132ed6c2c022ff8be9`).
- Review worksheet: `/tmp/target41_camera_safe_blinded_review.json`
  (SHA-256 `c9b18498a4ef1dd24fa45c84336a7256eeeed2a1d027fcd5230867fe76942121`).
- Source-case audit: `/tmp/target41_camera_safe_source_disjoint_audit.json`
  (SHA-256 `ba874204694e616241a7c54046c1c75e08cb533bea3afd0adb30d9d639c93a65`).

The worksheet contains 18 unique source SHA-256 values and all 18 review
states are `unreviewed`. The source-case audit reports an empty overlap with
the reviewed source cases. This establishes a source-recording non-reuse
screen, not camera/site independence; camera identity is not present in the
file metadata and remains a human-review field.

## Frozen replay result

The composition was applied in memory to the 18 sidecars without reading event,
actor, route, or OCR truth labels and without writing a candidate sidecar.

| Check | Result |
|---|---:|
| sidecars / cases complete | 18 / 18 |
| candidate records / clips with records | 29 / 13 |
| baseline route types | 8 direct-vehicle, 21 person-vehicle |
| frozen-composition route types | 9 direct-vehicle, 20 person-vehicle |
| changed assignments | 1 |
| preserved NULL alternatives | 29 / 29 |
| selected-route alignment | 29 / 29 |
| original input path unchanged | 29 / 29 |

The only changed row is `litter_case_202`, from
`person:1:vehicle:vehicle:3` (`person_vehicle`) to
`direct:vehicle:3` (`direct_vehicle`), fired by the preregistered
`direct_person` guard. No unlisted rule fired. All 18 MP4 outputs passed the
existing decode smoke check.

## Interpretation and gate

No accuracy, precision, event recall, camera generalization, or enforcement
claim is made. The 18 clips are still `unreviewed`; applying the candidate to
their predictions cannot determine whether the one changed assignment is
correct. The next authorized gate is one reviewer completing the worksheet
with verified event status, actor boxes, camera/site/session/source-recording
IDs, source-hash confirmation, and (where applicable) plate transcription.
Only after the fail-closed provenance gate passes may the frozen paired
evaluator run on a camera/site-disjoint set. A loss or unstable route there
discards the research composition and leaves production unchanged.

## Rollback

No production file, resolver, cost, route schema, configuration, or label was
changed. Stop using the in-memory preflight and retain the current production
resolver; no runtime rollback is required.
