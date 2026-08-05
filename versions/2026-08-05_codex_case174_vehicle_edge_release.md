# case 174 vehicle-edge release filter

- Date: 2026-08-05
- Author: Codex
- Scope: `scripts/pipeline/geometry.py`, `scripts/pipeline/litter_tracker.py`

## Problem

`litter_case_174` contains a visible light-blue bottle thrown from the right edge of vehicle track 2 during frames 96--102 at 10 FPS. Its early flight is nearly horizontal. The pre-tracker `horizontal_streak` filter rejected frames 97--98, splitting the bottle track and losing the vehicle-origin evidence before confirmation.

## Change

Keep the horizontal-streak rejection by default. A candidate bypasses only that rejection when its trajectory has no more than three observations, its first center is inside a currently detected vehicle/scooter box, and its current center is at least 40 pixels outside that same box. It still passes the existing holding, vehicle relative-separation, trajectory, vehicle thrower, and temporal-confirmation gates.

The thrower resolver now also has an exact-box-origin fallback for that same condition. It applies only after the pseudo-ground score has produced no thrower: a candidate must start inside one vehicle/scooter bbox, end at least 40 pixels outside it, and satisfy the existing release-direction rule. It does not choose the merely nearest vehicle.

## Validation

- Added pure-function regression tests for the recent vehicle-edge horizontal release and exact-box-origin thrower fallback.
- Re-ran `litter_case_174` on GPU 0 with production detector/tracker and Smart Backtrack enabled (`SMART_BACKTRACK_SIDECAR=0`): one confirmed direct-vehicle event at frame 99, resolved to vehicle 2.

## Limit and rollback

This is a candidate-preservation fix, not an accuracy claim. Review additional labeled fast-throw clips before changing the 40-pixel separation or three-observation bounds. Roll back by reverting this file and the paired test/documentation changes.
