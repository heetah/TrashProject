# RT-DETR pre-postprocessing bbox debug overlay

- Date: 2026-08-20
- Author: Codex
- Branch/commit: current working tree; not committed

## Problem

Annotated video only showed confirmed litter. A detector hit rejected by geometry,
motion, holding, or tracker confirmation could not be located visually.

## Change

When `LITTER_DEBUG=1`, annotated video now draws every RT-DETR `litter` result that
already passed the configured model confidence threshold but has not yet entered
project geometry/motion/holding/tracker post-processing. Boxes are magenta and use
the label `RTDETR raw <confidence>`.

Normal output remains unchanged when `LITTER_DEBUG=0`.

## Interface

```bash
LITTER_DEBUG=1 OUTPUT_ROOT=output/debug \
  conda run -n rtdetr python scripts/main.py /path/to/video.mp4
```

## Validation

- Unit coverage verifies that a 2-pixel-wide RT-DETR box is rendered even though
  the later geometry gate rejects it.
- Unit coverage verifies that the overlay is disabled by default.

## Limits

“Raw” means Ultralytics RT-DETR output after class/confidence filtering (and model
framework NMS/decoding), not unfiltered decoder queries/logits.

## Rollback

Revert the debug renderer, its call site, tests, and documentation. Production
behavior can be rolled back immediately by keeping `LITTER_DEBUG=0`.
