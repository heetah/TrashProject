# Actor ground-truth metric join

- Date: 2026-08-26
- Author: Codex
- Branch: `heetah-dev`

Added `scripts/summarize_actor_ground_truth_metrics.py` to join the user's
vehicle/person ID table with the 63-clip recovery replay. It emits clip-level
metrics (independent unit), event-level release/birth geometry, route costs and
margins, and explicitly marks unusable clips 24, 68, 92, 111 and 138.

Generated report:
`artifacts/actor_ground_truth_metrics/recovery_horiz1_full_20260826/`.

The report excludes UNUSED clips and explicit `?` IDs from accuracy
denominators. Multiple confirmed tracks within one clip remain visible in the
event CSV and are not treated as independent statistical samples.
