# Attribution formula validation

- Date: 2026-08-26
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `e8a5fef`
- Commit: not committed

## Problem

The project needs reproducible statistical evidence for Smart Backtrack's
scale-normalized distance gates and distance/time weights.  Existing runtime
sidecars are model outputs rather than ground truth, manual actor IDs do not
match model track IDs, and the exported event rows remain `unreviewed` even
though clip review is complete.

## Changes

- Added `scripts/validate_attribution_formula.py`.
- Kept human event/release/actor labels separate from runtime observations.
- Interpreted `first_visible_litter_frame` values 0/1 as baseline RT-DETR miss
  sentinels and excluded them from temporal-delay statistics.
- Derived route type from the user-confirmed actor-presence rule without
  rewriting source annotations.
- Proposed manual-to-model actor mappings by same-frame, same-class bbox IoU.
- Added strict/moderate/exploratory event-match tiers.
- Added positive-distance bootstrap intervals, event-cluster bootstrap AUC,
  Wilson intervals, a D/T replay sweep, paired bootstrap, and exact McNemar
  comparison.
- Added direct GT release-time comparisons for B0, B0-H0/2, B0-H0, and the
  selected resolver result; also measured B0/B1 speed and B1/B2 speed-change
  association with the required backtrack fraction.
- Added same-frame, same-video correct-versus-distractor comparisons for actor
  release regions, full bboxes, upper-body/box centers, and vehicle centers.
- Added a TrackFlow-inspired candidate likelihood analysis using normalized
  distance, B0/B1-normalized release time, and a von-Mises-motivated forward
  direction penalty, with leave-one-event-out evaluation, event bootstrap
  coefficient intervals, exact paired sign-flip testing, monotone physical
  constraints, regularization sensitivity, and selective coverage diagnostics.
- Added CSV, JSON, Traditional Chinese/English Markdown, and PNG outputs.
- Added targeted tests and documented the research command in `README.md`.

## Interface and configuration

```bash
conda run -n rtdetr python scripts/validate_attribution_formula.py \
  --ground-truth runs/grounding_truth \
  --candidates output/backtrack_multi_actor_v1_20260825_rerun \
  --output artifacts/attribution_formula_validation_20260826 \
  --bootstrap 5000
```

No production defaults, detector thresholds, event-confirmation behavior, or
Smart Backtrack resolver defaults changed.

## Test evidence

```text
PYTHONPATH=scripts /home/under115a/miniconda3/envs/rtdetr/bin/python \
  -m pytest -q \
  tests/pipeline/test_validate_attribution_formula.py \
  tests/pipeline/test_backtrack_study.py \
  tests/pipeline/test_backtrack_annotations.py

28 passed, 1 skipped
```

The validation CLI completed on 58 GT events and 26 confirmed-event candidate
records, matching 19 events and accepting 26/26 actor mappings at IoU >= 0.5.

The likelihood analysis used 15 strict+moderate matched events for its primary
result and kept the 19-event result as exploratory sensitivity only.

## Limitations

- Event rows are still exported as `unreviewed`; derived route labels and
  automatic actor mappings remain provisional evidence.
- Distractor analysis exists only for confirmed-event sidecars.
- The dataset contains no true negative clips and no reviewed NULL routes.
- Person and cross-camera support are insufficient for a universal parameter
  claim.
- The analysis does not validate event-confirmation precision.

## Rollback

Remove the validation script and its targeted test, remove this README section,
and delete the generated ignored artifact directory.  Production behavior is
unchanged, so no runtime rollback is required.
