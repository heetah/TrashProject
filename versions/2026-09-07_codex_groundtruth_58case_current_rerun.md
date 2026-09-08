# Current-production 58-case rerun

- Date: 2026-09-07
- Author: Codex
- Branch: current working branch
- Commit: uncommitted working-tree changes
- Type: runtime validation

## Scope

Executed the current production pipeline independently on all 58 clips marked
`video_usable=true`. The five `extremely small`/unused clips were excluded by
the canonical clip annotations. No old output was reused.

## Evidence

- 58/58 completed, zero failure or timeout.
- 58 analysis files, 58 Smart Backtrack sidecars, 58 logs and 58 annotated MP4s.
- All annotated MP4 video streams passed `ffprobe` decoding.
- 41/58 clips produced confirmed litter events; 62 confirmed records total.
- Vehicle-only numeric-ID result: 32/56; the separate prior human-adjudicated
  convention is 34/58 after adding unknown-ID cases 74 and 174.

Full metrics, confidence intervals, interpretation limits and artifact paths are
recorded in `artifacts/groundtruth_batch_current_20260907/REPORT.md`.

## Safety boundary

Dynamic pseudo-homography remains disabled and isolated from production
attribution. Since every short clip is a new process, this run validates the
current production baseline, not continuous per-camera Homography convergence.

