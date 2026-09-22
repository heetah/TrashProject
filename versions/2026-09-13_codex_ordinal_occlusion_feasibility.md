# Ordinal occlusion feasibility package

- Date: 2026-09-13
- Author: Codex
- Branch: `heetah-dev`
- Commit: uncommitted working tree based on `a149c5975898643e5131caf0579bd0aefe228d75`

## Problem

Single-frame visible-mask containment was proposed as a lightweight proxy for
front/behind ordering. Existing reviewed route labels do not contain ordinal
depth truth, and using the correct route actor as a direction label would leak
the desired answer.

## Change

- Added `scripts/pipeline/backtrack/ordinal_validation.py`.
- Added `scripts/build_ordinal_occlusion_package.py`.
- Added focused tests in `tests/pipeline/test_ordinal_occlusion_package.py`.
- Built an attribution-blinded feasibility package at
  `artifacts/ordinal_occlusion_review_20260913_v2`.
- Kept all production ordinal/mask weights at zero; the resolver remains
  unchanged.

The public queue contains only opaque A/B identities, actor-overlap time windows,
and empty review fields. Raw paths, model tracklet identities and proposed
display boxes are isolated in the private mapping. One source-video/tracklet
pair contributes at most one candidate window, preventing fragmented windows
from inflating the count.

## Evidence and limits

- Source event rows read: 3.
- Candidate pair windows: 14.
- Verified independent pair windows: 0.
- Verified camera groups: 0.
- Current status: not ready for ordinal model validation.

The source sidecars are event-conditioned and the A/B boxes are model proposals.
The package therefore demonstrates review plumbing only; it is not an unbiased
dataset or metric-depth ground truth.

The pre-registered promotion boundary requires at least 50 independently
verified windows, at least 5 verified camera groups for annotation feasibility,
balanced decisive examples, tie/unknown coverage, a delayed alias-reversed
repeat round, nominal Cohen kappa lower 95% bound at least 0.60, and orientation
violation upper 95% bound at most 0.05. Production evaluation additionally
requires at least 6 cameras with at least 2 locked holdout cameras. If these
conditions fail, the ordinal feature remains rejected.

## Validation

- Ordinal and mask focused tests: `7 passed`.
- Python compilation: passed.
- `git diff --check`: passed.

## Rollback

Delete the two ordinal package source files and their focused test. No resolver
or production cost rollback is needed because this change is research-only and
no production weight was enabled.
