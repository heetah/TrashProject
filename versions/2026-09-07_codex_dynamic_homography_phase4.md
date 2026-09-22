# Dynamic pseudo-homography Phase 4 robust motion losses

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added quality-weighted Huber losses for relative velocity change, log speed
  ratio, curvature discontinuity and local traffic-flow direction consistency.
- Made the objective resistant to trivial coordinate shrink through per-track
  median-speed normalization, log ratios and angular residuals.
- Added candidate-H safety validation before loss evaluation and explicit
  insufficient-evidence reporting instead of synthetic zero loss.
- Added per-component sample counts, losses, weighted losses and normalized
  total loss to the opt-in calibration analysis summary.
- Reused one local-observation snapshot per summary so field, clusters and
  loss diagnostics are evaluated from identical evidence.

## Mathematical boundary

The motion residual is `||v_i-v_(i-1)||/median(||v||)` and the speed residual
is `|log((||v_i||+eps)/(||v_(i-1)||+eps))|`. Curvature compares consecutive
signed turning angles, not each angle against zero. Direction consistency uses
a short finite-difference probe under H inside existing traffic-flow clusters.
All residuals receive Huber penalties and observation-quality weights.

These quantities are objective diagnostics. They are not probabilities,
accuracy estimates, metric distances or proof that the default weights/deltas
are optimal.

## Configuration

`.env.example` now exposes four component weights, four Huber deltas and the
local direction probe fraction. Equal weights are a neutral starting point;
the robustness transitions remain research priors pending grouped replay.

## Safety

`DYNAMIC_HOMOGRAPHY=0` remains the default. Phase 4 does not optimize H, update
the active transform, affect event confirmation, alter attribution costs or
remove the complete NULL route. Reports retain `affects_attribution=false`.

## Validation

- Phase 1-4 calibration tests: 32 passed.
- Affected pipeline selection: 155 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 276 passed, 12 skipped, 2 failed. The two known
  suite-order detector-render SHA mismatches remain in scenarios A/B; their
  characterization module passes when run in isolation, and no golden output
  was changed.

## Rollback

Removing `losses.py`, its exports and the `calibration_losses` summary field
restores Phase-3 behavior. Disabling `DYNAMIC_HOMOGRAPHY` prevents collection.
