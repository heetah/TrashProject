# 2026-09-13 Codex — RT-DETR discarded-intermediate validation

- Date: 2026-09-13
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `a149c5975898643e5131caf0579bd0aefe228d75`
- Commit: uncommitted working-tree change

## Problem

Normal RT-DETR postprocessing keeps only thresholded final boxes, classes and
scores. The encoder proposal, final decoder query state and query-aligned
embedding are discarded even though they may provide reliability or temporal
identity evidence for litter trajectory reconstruction.

## Change

Added a research-only PyTorch extractor in
`scripts/pipeline/litter/rtdetr_intermediates.py` and a reproducible evaluator
in `scripts/analyze_rtdetr_intermediates.py`. The extractor rejects unsupported
heads/exported backends and does not alter Ultralytics Results. The evaluator
writes a manifest before inference, hashes consumed videos, sidecars and model
weights, records per-case completion, and keeps the production weight at zero.

The cohort is the frozen current baseline's 41 clips with confirmed events:
62 event trajectories and 139 `accepted_tracker` observations. A strict source
audit excluded 29 `raw_rtdetr_recovered` observations that the original
`litter_history` convenience field had mixed into the histories. These are
still silver tracker labels, not independent human-reviewed object identities.

## Evidence

- Strict target-blind execution completed for 41/41 event-bearing clips with 0
  failures. The full reviewed benchmark still contains 58 clips; the other 17
  have no confirmed candidate and cannot contribute temporal query histories.
- Only 5/62 trajectories (5 clips) contain at least four accepted observations;
  primary and rolling denominators were 5 and 7 respectively, with no attrition.
- The first three accepted query centers fit an OLS constant-velocity model and
  predict the fourth. Query candidates and geometry/embedding scores are frozen
  before applying a threshold-aware, one-to-one target IoU label. Missing
  positive queries count as failures.
- At production score floor 0.40, positive-query coverage was 5/5 and median
  candidate count was 1. Geometry and embedding both achieved 5/5 top-1:
  paired delta 0, rescue/harm 0/0. Embedding provides no incremental choice at
  the production candidate threshold.
- At research floor 0.01, median candidate count was 7. Geometry achieved 3/5
  and embedding 5/5, for 2 rescue / 0 harm and paired delta +0.40. The
  clip-cluster bootstrap 95% interval was [0.00, 0.80] and exact cluster
  sign-flip p=0.25. This is underpowered and does not pass the retained-signal
  screen.
- The former target-aware hard-negative AUC 0.681 is invalid for promotion:
  negative selection observed positive target geometry/score and histories
  contained recovered prefixes. It is retained only as an audit trail.
- Research extractor/evaluator tests: 16 passed.
- Full production pipeline regression: 377 passed, 12 skipped.

Primary reproducible evidence is stored in
`artifacts/rtdetr_query_retrieval_target_blind_v3_20260913/`; its manifest
hashes the model, videos, sidecars, analyzer and extractor. Earlier exploratory
runs remain preserved only as non-promotable audit history.

## Interface and configuration

The evaluator accepts the sidecar root, `.pt` model, output directory, image
size, research/production query score floors, positive/negative IoU bounds,
hard-negative score and geometry calipers, cluster-bootstrap replicate count,
and minimum independent-cluster evidence gates. The 18-discordant-cluster gate
corresponds to 0.867 power for a positive-sign probability of 0.8 at one-sided
alpha 0.05. No production environment variable or pipeline output schema
changed.

## Limitations

- Confirmed tracker histories contain detector/tracker selection bias and are
  not human-reviewed same-object labels.
- Only 5/62 trajectories survived the four-observation requirement, and three
  of the five include irregular 9, 15 or 21-frame gaps. This selected subset
  cannot establish cross-camera generalization.
- The dataset has no reviewed negative clips/NULL routes, so event precision,
  abstention safety and end-to-end attribution impact remain unmeasured.
- These signals target litter continuity/release reconstruction, not direct
  person/vehicle causality.
- Geometry-only versus embedding-only establishes signal feasibility, not the
  incremental value of a fused cost. Any fusion weight requires a separate
  camera-grouped development split and a held-out paired 58-case evaluation.
- Independent protocol, code and data-audit agents challenged target leakage,
  recovered-prefix contamination, assignment cardinality, clustered inference,
  fixed denominators and insufficient sample size; their blocking findings are
  reflected in the v3 protocol and no-promotion verdict.

## Rollback

Remove the standalone evaluator, extractor, tests and this research note. No
production runtime path, cost, threshold, event confirmation or NULL route
requires rollback because none was changed.
