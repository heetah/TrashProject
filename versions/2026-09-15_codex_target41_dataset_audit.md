# 2026-09-15 Codex — target-41 dataset and split audit

- Date: 2026-09-15
- Author: Codex
- Branch: `heetah-dev`
- Scope: read-only annotation/split audit; no production resolver, cost,
  schema, model, or configuration change

## Purpose

The mask-temporal research replay reaches 43/58 on the reviewed positive
denominator. This audit checks whether that result can support the requested
robustness claim and identifies the data required for an independent
camera-level holdout. Machine-generated sidecars remain predictions; no
prediction was written back as ground truth. The replay's signed-mask
convention was rechecked against `cv2.pointPolygonTest` and the focused unit
test: `s > 0` is inside the visible polygon and `s < 0` is outside. The
research note now records that convention explicitly; no metric result or
production behavior changed.

## Current annotation inventory

The authoritative files are `runs/grounding_truth/project.json`,
`clip_annotations.jsonl`, `event_annotations.jsonl`, and
`actor_annotations.jsonl`.

| object | result |
|---|---:|
| clip manifest rows | 63, all `reviewed` |
| usable clips | 58 |
| usable clips with one non-ignored event | 58 |
| event rows | 58, all `reviewed` |
| actor rows | 74, unique by `(video_id, actor_id)` |
| vehicle actor rows | 58 |
| person actor rows | 16 |
| derived route labels | 42 `direct_vehicle`, 16 `person_vehicle` |

The remaining five manifest clips (`litter_case_111`, `138`, `24`, `68`,
`92`) are `video_usable=false` with zero event rows. Four say
`contains_litter=true` and one says `contains_litter=uncertain`; none is a
verified negative.

## Integrity checks

- Duplicate clip IDs, event IDs, and `(video_id, actor_id)` pairs: none.
- Event and actor references to a missing clip: none.
- Event release intervals, release coordinates, actor frames, and actor boxes:
  all within the project frame/image bounds.
- Project metadata rows with `decodable=false`, `completed=false`, or a
  non-reviewed state: none.
- The evaluator's ground-truth hashes remain the current provenance anchors:
  event `c59e4d0f7e99c478d8ef6bbece7047aec4564716d7559391a9a855422731f7bd`,
  actor `d7d0de3d2962b10cfdf3d351da55e7a9d520ed835f50b6845db0f4e3aa56e187`,
  clip `3c5c2b03b225481098cc72eeb0aabb0fc0e7cb68ba7ec5031abec5656b1f5861`.

## Split and leakage findings

`project.json` contains video IDs, filenames, frame metadata, and decodability,
but no camera/site/session ID, source-recording ID, or source-content hash. Its
`video_folder` is a historical Windows path. Therefore an independent
camera-group split cannot be proven from the current manifest.

The available source directory contains 242 standardized clips. Sixty-three
stems occur in the reviewed clip manifest and 179 do not. A second
`litter_order` directory contains 233 original-looking files, but the mapping
between those files and standardized clips is not recorded. Filename, frame
rate, resolution, or visual similarity must not be promoted to a camera ID;
full SHA-256 of the 63 reviewed standardized source clips found 63 unique
hashes (no exact duplicate within the evaluated set). A second complete
SHA-256 pass over all 242 standardized clips found 242 unique hashes and no
reviewed/unreviewed exact duplicate. Near-duplicate leakage and the four
reviewed clips without an exact `litter_order` hash match remain unknown.

As a separate source-lineage check, full-file SHA-256 matching against the 233
files in `litter_order` linked 226 standardized clips exactly: 59 of the 63
reviewed-manifest clips and 167 of the 179 unreviewed clips. The 59 reviewed
matches collapse to 54 original source-case identifiers; three source cases
have multiple reviewed segments (`Case081`, `Case151`, `Case110`). Only three
source cases are shared across reviewed and unreviewed exact matches
(`Case081`, `Case139`, `Case184`), corresponding to standardized clips
`litter_case_70`, `litter_case_6`, and `litter_case_140`; those clips are not in
the 18-case camera-safe queue. The remaining four reviewed clips have only
heuristic visual matches and remain unassigned. This excludes exact
source-file reuse from the queue, but original filenames still do not encode a
verified camera/site ID, so human confirmation remains mandatory. The complete
machine-readable audit is `/tmp/target41_crossdir_exact_matches.json` and the
reviewed summary is `/tmp/target41_source_crossmatch_summary.json`.
All 18 clips in the camera-safe queue have exact SHA matches to `litter_order`
and map to 18 distinct original source-case identifiers. This establishes a
source-recording-disjoint queue, but not yet a camera-disjoint queue.
The explicit overlap audit is `/tmp/target41_camera_safe_source_disjoint_audit.json`
(`source_case_overlap_with_reviewed=[]`).

Using those exact source-case links, a dependence-sensitive bootstrap over the
50 source-case groups containing the 54 mapped usable reviewed clips gives a
candidate micro-rate 95% interval of approximately `[0.596, 0.836]` and an
equal-group interval of `[0.580, 0.830]` (baseline intervals `[0.500, 0.755]`
and `[0.490, 0.750]`). The four usable reviewed clips without exact source
hash matches are reported separately, not silently imputed into a group. This
is stronger source-recording dependence evidence than the visual-fingerprint
screen, but it is still not a camera-generalization estimate; the report is
`/tmp/target41_source_case_bootstrap.json`.

A paired version of that bootstrap preserves each source-case group in both
arms and resamples 50 groups with replacement (100,000 replicates, seed
`20260915`). On the 54 mapped clips, candidate minus baseline was `+5/54`
(`+9.26` percentage points); the percentile 95% interval for the paired rate
difference was `[+1.92, +17.54]` percentage points, and 99.466% of replicates
favoured the candidate. This is a descriptive dependence-sensitivity result:
the groups are exact source-lineage units rather than verified camera IDs, the
rule was selected on this same development cohort, and the four unmapped clips
remain in the official 58-case denominator. The reproducible output is
`/tmp/target41_source_case_paired_bootstrap.json`; it must not be reported as
confirmatory significance or cross-camera accuracy.

A descriptive leave-one-source-group-out jackknife over those same 50 groups
keeps the candidate above the baseline in every omission: candidate mapped
rates range from 36/51 (70.6%) to 39/53 (73.6%), while baseline ranges from
31/51 (60.8%) to 34/53 (64.2%). All 50 omitted-group deltas are positive; the
five gain groups are `Case006`, `Case108`, `Case117`, `Case151`, and `Case185`,
and no group contains a candidate loss. This is a concentration/sensitivity
check—not independent cross-validation—because the fixed replay rule was
selected with knowledge of the same development set and the source groups are
not verified camera IDs.

The 179 unlabeled standardized clips are technically readable and split across
five observed format groups (resolution/frame rate): 83 at 2592×1944/10 FPS,
23 at 2592×1944/12 FPS, 68 at 1440×1080/30 FPS, 3 at 1440×1080/10 FPS, and 2
at 1920×1080/30 FPS. These are acquisition-format strata only, not camera
identities. A deterministic five-per-stratum review sample is recorded
temporarily at `/tmp/target41_holdout_sample.json`.

To look for split leakage without inventing camera IDs, a read-only visual
fingerprint audit sampled five frames per each of the 242 source clips, formed
a temporal-median low-resolution background, and ranked cosine-similar pairs.
SIFT/RANSAC on the raw mid-frames confirmed strong fixed-scene matches for the
highest cross-status pairs (for example, `litter_case_6`↔`7`: 615 inliers;
`69`↔`70`: 328; `140`↔`141`: 417; `59`↔`87`: 257; `135`↔`57`: 140;
`166`↔`167`: 350). A conservative cosine ≥0.80 graph yields mixed
reviewed/unreviewed candidate groups such as `{6,7,48,85}`, `{66,67,68,69,70}`,
`{140,141,84}`, `{87,112,136,171,177,179,187,189,43,58,59,63}`,
`{135,57}`, and `{166,167}`. These are strong same-scene leads, not verified
camera identities; the complete audit is `/tmp/target41_source_fingerprint_audit.json`.
Human confirmation is required before using these groups for a camera-disjoint
split.

The sample was then run with the frozen pipeline and recorded in
`/tmp/target41_holdout_run_manifest.json`. The 20-case batch completed 20/20
with return code 0, producing 20 analysis JSON files, 20 backtrack-candidate
JSONL files (36 candidate rows in total), and 20 annotated MP4 files. `ffprobe`
reported no video decode failures. The model summaries contain 16 confirmed
event outputs across 11 cases, but every row remains
`accuracy_status=not_evaluated`; these are review materials, not
positive/negative labels or accuracy evidence. The completed batch manifest
SHA-256 is `2906b75124f439c44c306122aa5c22b7251ff8ea8fc24a21496eaacec37e2b70`.
An output invariant check over all 36 candidate records found no history-array
or lineage-length mismatch; all seven derived observations were explicitly
non-independent, so they were not counted as extra detector measurements.

Because the first 20-case sample was not source-independent, a second queue was
constructed by excluding unreviewed clips whose fingerprint similarity to any
reviewed clip was at least 0.75 and taking one representative per remaining
unreviewed component. This conservative queue has 18 clips (format strata with
too few eligible clips are not padded). Its frozen run manifest is
`/tmp/target41_camera_safe_run_manifest.json`; 18/18 cases completed, producing
18 analysis JSON files, 18 candidate JSONL files, and 18 decodable MP4 files.
The run produced 29 model-confirmed event outputs, all still
`accuracy_status=not_evaluated`. Fingerprint separation is only a screening
heuristic and must be confirmed by the human reviewer before this queue can be
called camera-disjoint. A consolidated single-reviewer index (including the
computed SHA-256 for each of the 18 source files) is available at
`/tmp/target41_camera_safe_review_index.md` (machine-readable form:
`/tmp/target41_camera_safe_review_index.json`).

The same 18-case run was checked with
`summarize_dynamic_homography_validation.py`: it contains 29 event snapshots,
zero applied dynamic-H events, and 29 `feature_disabled` fallbacks (no
`LOCKED` snapshot). This is a runtime-configuration/provenance result, not an
accuracy measurement; the mask-temporal 43/58 candidate therefore has no
hidden pseudo-homography or 3D-depth contribution.

An `ffprobe` metadata pass over the same 18 source files found no embedded
camera/site identifiers. Some files do contain generic provenance tags: 13
have an encoder tag and 5 have a `creation_time` tag, but neither field
identifies a camera or site (and the tags are not treated as such). The files
span objective acquisition strata—5 clips at 1440×1080/30 FPS, 1 at
1440×1080/10 FPS, 2 at 1920×1080/30 FPS, 5 at 2592×1944/10 FPS, and 5 at
2592×1944/12 FPS—and include H.264 Constrained Baseline/Main/High plus one
MPEG-4 Simple Profile file. These fields can help the reviewer detect an
obvious format/session split, but they are not camera identities; no group is
promoted to camera-disjoint without human confirmation.

The maintained annotation exports were checked as well:
`runs/grounding_truth/export.csv` and `export.xlsx` contain reviewed
clip/event/actor fields and route-distance annotations, but no camera, site,
session, or source-recording columns or hidden workbook metadata. They
therefore cannot supply the missing independent camera grouping; the reviewer
table keeps those fields blank until a human supplies them.

The same 18 sidecars were also converted into an attribution-blinded ordinal
queue at `/tmp/target41_camera_safe_ordinal_review`. It contains 78 candidate
actor-overlap windows from 29 event-conditioned records; public rows hide route,
release, litter identity, and model scores. No ordinal window or camera group
has been independently verified yet, so this queue cannot justify a depth
weight or a 3D/ordinal accuracy claim.

As a reproducible dependence sensitivity check, the 242 fingerprint vectors were
connected at cosine thresholds 0.80 and 0.75, then only the 58 usable reviewed
cases were scored with the frozen paired evaluator. The machine-readable report
is `/tmp/target41_source_cluster_report.json`. At 0.80 there are 54 components
containing reviewed cases: baseline/candidate remain 38/58 and 43/58, while
all-reviewed-correct components increase from 34 to 39; at 0.75 there are 45
components and the corresponding counts are 28 to 32. The five gains occur in
five separate components and no component contains a correctness loss. These
are screening clusters over visual fingerprints, not camera IDs or an
independent estimate; human source/session confirmation is still required.

The mask replay was also rerun at YOLO-Seg image sizes 512, 640, and 768 with
the same confidence floor and frozen selection rule. It produced 43/58 at each
scale, with the same five gains and zero losses; this tests inference-scale
stability only and does not replace a camera-disjoint review. Among 431 records
available at all three scales, 24 raw signed-mask values changed sign, while
the guarded route decisions did not; the mask is therefore not treated as a
physical depth measurement.
In-memory JPEG re-encoding at qualities 90 and 50 likewise retained 43/58 with
the same five gains and zero losses (440 measurements per quality), providing a
codec-noise check but not camera-level validation.
The per-case route tuple was identical across all nine retained replay
conditions (three confidence floors, three image sizes, and two JPEG
qualities), yielding 58/58 decision consensus on this development set.
Two independent default-scale passes were bit-identical on all 437 mask
measurements, so the observed stability is reproducible rather than a random
inference artifact.

There is no row with `contains_litter=false`, and no separately reviewed
background/negative set. The positive route metric cannot estimate precision,
false-positive rate, or enforcement readiness. OCR text/legibility ground
truth is also absent.

## Metric consequence

The current research candidate is a development-set route result only:

- 43/58 provisional route-correct cases (point estimate 74.14%).
- 50/58 assignment-conditioned accepted event matches (86.21%).
- The five unusable clips are correctly excluded by the evaluator's usable
  positive denominator, not silently treated as negatives.

As a sensitivity description only, weighting each visual-fingerprint component
equally gives 39/54 (72.2%) at cosine ≥0.80 and 32/45 (71.1%) at cosine ≥0.75
for the candidate (baseline: 34/54 and 28/45) when a component is counted only
if every reviewed case in it is correct. These component rates are not
camera-disjoint estimates: the grouping is an unsupervised screening heuristic
and the denominator is still the same reviewed positive set.
Cluster bootstrap (100,000 resamples, cosine ≥0.80) gives a micro-rate 95%
interval of approximately [0.625, 0.850] and an equal-component interval of
[0.611, 0.843]; the five nonzero component gains retain exact sign-flip
`p=0.0625`. These intervals describe dependence sensitivity only, not a
camera-generalization guarantee.

This evidence supports the explainability and perturbation checks in the mask
research note, but it does not establish cross-camera robustness or the 85%
lower-bound requirement.

## Required holdout gate

Keep the existing 58-case labels immutable. Before promotion, create a new
review queue from the 179 unlabeled standardized clips (or a documented,
source-linked subset) and have a human reviewer record, at minimum:

1. verified positive, verified negative, ambiguous, or unusable status;
2. camera/site/session and source-recording identifiers;
3. source SHA-256 and exact frame/time metadata;
4. independently reviewed event interval/location and actor boxes;
5. route/NULL decision and OCR transcription/legibility where applicable.

Assign complete source recordings to one split only. Re-run the frozen paired
comparison on the camera-disjoint set, including confidence/time perturbations,
and require no correctness or event-match regression. Until that gate passes,
retain the 43/58 mask candidate as research-only and leave production on the
current resolver.

## Reproducibility and rollback

The audit is read-only and has no runtime rollback. Inputs are the four files
listed above plus the source directories under
`/mnt/nas/under115/under115a/under115a/litter_vidshort/`. Removing this note
does not alter labels, model outputs, or production behavior.

The per-file reviewed-source digest from this run is retained temporarily at
`/tmp/reviewed_source_sha256.json`; it contains 63 entries and 63 unique
SHA-256 values.
