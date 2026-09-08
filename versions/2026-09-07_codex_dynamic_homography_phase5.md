# Dynamic pseudo-homography Phase 5 lane/flow consistency

- Date: 2026-09-07
- Author: Codex
- Branch: `heetah-dev`
- Base commit: `b7f73da`
- Commit: uncommitted working-tree change
- Type: feat / test / docs

## Change

- Added a cross-vehicle local lane/flow consistency component to the candidate-H
  diagnostic objective.
- Built each local centerline leave-one-track-out, so a vehicle cannot validate
  its own trajectory and at least one independent track must support the loss.
- Used a quality-weighted moving median and robust local projected tangent
  rather than a global line, preserving curved roads and longitudinal offsets.
- Normalized perpendicular residual by projected image-footprint scale to
  prevent uniform output scaling from reducing the loss.
- Added image-space grid neighbor lookup to avoid a full all-pairs scan.
- Added explicit sample counts and configurable support, radius, Huber and
  component-weight parameters.

## Mathematical definition

For observation `i`, peer set `N_i` contains only other track IDs in the same
flow cluster and fixed image-space neighborhood. Let `c_i` be their weighted
coordinate median and `t_i` their robust projected unit tangent. With projected
point `q_i` and projected image footprint area `A_H`, the residual is:

`r_lane,i = |cross(q_i-c_i, t_i)| / sqrt(|A_H|)`.

The component is the observation-quality and cluster-confidence weighted mean
of `Huber(r_lane,i)`. It measures local curve-family dispersion, not metric lane
width.

## Evidence boundary

No lane ground truth is available, and one flow cluster may contain adjacent
physical lanes. Therefore a lower lane loss alone does not prove a better H.
It must be combined with independent motion/perspective constraints and tested
by camera/event-grouped replay before any update is enabled.

## Safety

`DYNAMIC_HOMOGRAPHY=0` remains the default. Phase 5 only adds diagnostics;
production H, litter confirmation, attribution costs and NULL routes are
unchanged. Reports continue to state `affects_attribution=false`.

## Validation

- Phase 1-5 calibration tests: 35 passed.
- Affected pipeline selection: 158 passed, 9 skipped.
- Production module compilation and `git diff --check`: passed.
- Full `tests/pipeline`: 279 passed, 12 skipped, 2 failed. The two known
  suite-order detector-render SHA mismatches remain in scenarios A/B; their
  characterization module passes when run alone, and no golden was changed.

## Rollback

Set `DYNAMIC_HOMOGRAPHY_LANE_WEIGHT=0` to exclude the component from the total,
or remove the lane helper and related fields to restore the Phase-4 objective.
