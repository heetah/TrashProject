# 58-case production rerun

- Date: 2026-09-07
- Author: Codex
- Branch: current working branch
- Commit: uncommitted working-tree changes

## Run

Reran all 58 clips marked `video_usable=true` from the grounding-truth clip
annotations with the production pipeline. Each clip used an isolated output
directory under `output/groundtruth_batch_d04_20260907`; Smart Backtrack
sidecars were enabled for auditability.

## Results

- 58/58 processes completed; 0 failures.
- 20/58 clips produced at least one confirmed litter event.
- 27 confirmed event records were produced in total.
- Fresh same-frame IoU mapping produced 20 comparable GT event matches.
- Vehicle-only attribution: 18/20 = 90.0%, Wilson 95% CI 69.9–97.2%.
- Full person+vehicle route: 15/20 = 75.0%, Wilson 95% CI 53.1–88.8%.

These attribution denominators condition on a confirmed event and accepted actor
mapping. They do not represent end-to-end 58-case accuracy. The 38 clips with
no confirmed event require a separate confirmation-coverage investigation.

## Artifacts

See `artifacts/groundtruth_batch_d04_20260907/REPORT.md`, its
`batch_manifest.jsonl`, and the fresh validation directory
`artifacts/groundtruth_batch_d04_20260907_validation`.
