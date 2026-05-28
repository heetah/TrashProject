# AGENTS.md

## Project Reality

- Repo root: `/home/se_copilot/trashProject`.
- Default runtime env: `conda run -n rtdetr ...`.
- Primary active video pipeline: `scripts-old-test/`.
- Important reference trees:
  - `scripts-old-stable/`: rollback/reference baseline. Do not edit unless asked.
  - `scripts-old-stgcn/`: STGCN / YOLO-Pose comparison and batch-eval path.
  - `scripts-heetah/`: integrated HEE TAH path when present in a task. Verify folder exists before using older notes.
  - `dataset-pose/`, `dataset-pose-wholebody/`, `dataset-pose-wholebody-under115a/`: legacy or existing STGCN/STGCN++ pose artifacts. Verify actual keypoint source before reusing.
  - `mmaction2/`: vendored STGCN/STGCN++ training/inference dependency. Keep edits narrow.
  - `mmpose-rtmw/`: legacy RTMW dependency. Do not use RTMW for the current production keypoint path unless the user explicitly asks to restore RTMW.

## Working Rules

- Inspect live code before answering. This repo drifts often; memory can be stale.
- Use `rg` / `rg --files` first for search.
- Preserve user changes. Never reset or checkout files unless explicitly asked.
- Use `apply_patch` for manual edits.
- Default language for explanations: Traditional Chinese when user asks codeflow/training/debug details.
- For runtime work, prefer actual evidence: command output, validation clip result, `py_compile`, video decode check, or artifact metric.
- If GPU is busy or user asks to lower compile/init churn, avoid repeated heavyweight model loads. Use static checks, small smoke tests, or existing logs first.
- Do not silently change the pipeline responsibility split. In the current workflow:
  - YOLO-Pose + STGCN handles only `normal` / `urinate`.
  - Littering is handled only by the litter object-event branch.
  - RTMW is not used for keypoint extraction.

---

# Current Workflow Contract

## High-Level Flow

```text
Input Video
  -> Load Models
  -> Background Frame Reader Thread
  -> Frame Queue
  -> Main Inference Thread
  -> YOLO-Seg Actor Detection
      -> Person Branch
          -> YOLO-Pose Keypoint Extraction
          -> Accumulate Keypoints by Person Track ID
          -> STGCN Action Recognition
          -> Output: normal / urinate only
      -> Vehicle / Scooter Branch
          -> Vehicle-Person Association
      -> Litter Branch
          -> RT-DETR / YOLO Litter Detection
          -> Motion Check
          -> Temporal Difference
          -> Holding Algorithm
          -> Confirm Littering Event
          -> Backtrack Thrower and Vehicle
          -> OCR License Plate
          -> Output Annotated Video
```

## Responsibility Split

### Person Action Branch

Responsible for:

- detecting / tracking person actors from YOLO-Seg output,
- extracting person keypoints using YOLO-Pose,
- accumulating keypoint sequences by `track_id`,
- running STGCN only after enough keypoint frames are available,
- classifying person action as only:
  - `normal`
  - `urinate`
- confirming `urinate` only with sustained temporal evidence.

Not responsible for:

- detecting littering,
- detecting thrown objects,
- confirming litter objects,
- vehicle/scooter OCR,
- license plate recognition,
- RTMW inference.

### Litter Object-Event Branch

Responsible for:

- detecting litter candidates with RT-DETR / YOLO litter detector,
- assigning litter IDs,
- recording first appearance frame `F2`,
- validating object motion and temporal difference,
- applying `litter_holding()` behavior evidence,
- confirming littering events,
- backtracking the likely thrower,
- associating thrower with vehicle/scooter when possible,
- triggering OCR and warning visualization for confirmed littering.

Not responsible for:

- human pose estimation,
- STGCN action classification,
- urinate classification.

### Vehicle / Scooter Branch

Responsible for:

- tracking vehicle/scooter actors,
- filtering tiny or invalid vehicle candidates,
- associating nearby person and vehicle/scooter actors,
- supporting later violation attribution and OCR crop selection.

Not responsible for:

- STGCN action recognition,
- keypoint extraction,
- direct litter confirmation.

### OCR Branch

Responsible for:

- cropping confirmed violator vehicle/scooter ROI,
- running PaddleOCR,
- outputting license plate text when confident enough.

Not responsible for:

- detecting actions,
- confirming litter objects,
- pose extraction.

---

# Main Pipeline: `scripts-old-test/`

- Entry: `scripts-old-test/main.py`.
- Typical command:

```bash
conda run -n rtdetr python scripts-old-test/main.py resources/resize.mp4 --batch 8
```

- Output goes under `output/*_annotated.mp4`.
- Actor model: YOLO segment, class-name based `person`, `scooter`, `vehicle`.
- Pose model: YOLO-Pose only for current STGCN keypoint extraction.
- Litter model: RT-DETR or YOLO litter detector, `litter` only.
- STGCN action module: `scripts-old-test/action.py`; enabled by default unless `--disable-action`.
- OCR path: `scripts-old-test/licensePlate.py`; expensive, should be gated with `--disable-plate` for litter-only speed work.
- Timing owner: `scripts-old-test/timeUtils.py`; preserve final grouped timing summary and `tqdm` behavior where present.
- RTMW must not be loaded, initialized, called, or used as fallback in the current production workflow.

## Expected Runtime Sequence

1. Read video frames in a background thread.
2. Push frames into queue.
3. Main thread pops a batch, usually `--batch 8`.
4. Run YOLO-Seg actor detection for `person`, `vehicle`, `scooter`.
5. For each tracked person:
   - use YOLO-Pose to extract keypoints,
   - append keypoints to `person_kps_buffer[track_id]`,
   - run STGCN only when enough frames exist,
   - output only `normal` or `urinate`.
6. For each vehicle/scooter:
   - keep valid tracks,
   - associate person and vehicle/scooter by proximity / IoU / policy logic.
7. Run litter detector to find litter candidates.
8. Send litter candidates into `GlobalLitterTracker`.
9. Confirm littering only after behavior / motion / temporal evidence.
10. For confirmed littering:
    - backtrack likely thrower,
    - associate vehicle/scooter,
    - crop plate ROI,
    - run PaddleOCR,
    - draw final violation annotation.
11. Write annotated output video.

---

# Detection Semantics

## Litter Detection Semantics

- RT-DETR / YOLO litter bbox is only a candidate. It is not a confirmed violation by itself.
- Confirmed litter must pass tracker/behavior evidence in `GlobalLitterTracker`.
- Preserve these intent signals:
  - physical trajectory validation,
  - horizontal movement,
  - downward Y movement,
  - ROI/background motion evidence for 1-2 frame detections,
  - polygon-aware holding rules in `litter_holding()`,
  - same-id litter history when requested.
- Keep `litter_holding()` behavior-based. Inside actor polygon, only clear downward + horizontal release motion should pass as released.
- Noise fixes should strengthen motion/shape/component evidence, not only lower thresholds.
- Draw only confirmed litter as confirmed. Avoid showing pending candidates as if final.
- Never trigger littering violation from STGCN output.

## STGCN Semantics

- STGCN classifier score is not person bbox confidence. UI text should say `STGCN`.
- STGCN is only for `normal` / `urinate` action judgment going forward.
- STGCN classes must be exactly:

```python
ACTION_CLASSES = {0: "normal", 1: "urinate"}
```

- Do not classify, alert, or visualize `littering` action through STGCN.
- Littering violations come from confirmed litter tracker/behavior evidence, not STGCN action output.
- `urinate` requires sustained evidence. Current preferred rule:
  - default 8 second temporal window,
  - at least 6 seconds positive `urinate` evidence.
- If current code still uses older policy values, such as 10 second window and 8 seconds positive, do not change silently. Confirm the intended threshold in the task or preserve existing code behavior.
- Suppress `urinate` for person tracks linked to vehicle/scooter when that policy is active.
- `ACTION_PREDICT_INTERVAL` reduces classifier cadence after sequence window is full; pose extraction can still dominate cost.
- STGCN inference should run only after enough keypoint frames are collected, usually:

```python
MIN_STGCN_FRAMES = 30
```

## YOLO-Pose Keypoint Semantics

- YOLO-Pose is the only current source of person keypoints for STGCN.
- Expected keypoint buffer concept:

```python
person_kps_buffer[track_id].append({
    "frame_index": frame_index,
    "keypoints": keypoints,
    "bbox": bbox,
    "confidence": pose_confidence,
})
```

- Convert YOLO-Pose keypoints into the STGCN input layout before inference.
- If keypoints are missing, low-confidence, or unstable:
  - do not invent a violation,
  - skip the frame or apply existing interpolation / smoothing only if already supported,
  - keep logs clear about skipped pose frames.

## RTMW Legacy Semantics

- RTMW is legacy in the current workflow.
- Do not use RTMW for:
  - keypoint extraction,
  - fallback pose inference,
  - STGCN preprocessing,
  - production visualization,
  - new training data generation.
- Search and remove or disable current-production usages of:

```text
rtmw
RTMW
rtmw_model
rtmw_pose
rtmw_keypoints
extract_keypoints_with_rtmw
mmpose-rtmw
```

- Keep vendored RTMW files untouched unless the task explicitly asks for legacy RTMW cleanup.

---

# Code-Level Requirements

## Action Classes

Old or invalid form:

```python
ACTION_CLASSES = ["normal", "urinate", "littering"]
```

Required form:

```python
ACTION_CLASSES = {0: "normal", 1: "urinate"}
```

Do not add:

```python
"littering"
"throwing"
"litter"
```

to STGCN action labels.

## STGCN Output Contract

Expected output shape for person branch:

```python
{
    "track_id": person_id,
    "action": "normal" | "urinate",
    "confidence": float,
    "frame_index": current_frame_index,
}
```

## STGCN Postprocessing

Allowed postprocessing pattern:

```python
if action == "urinate":
    update_urinate_state(track_id)
elif action == "normal":
    update_normal_state(track_id)
else:
    ignore_unknown_action(track_id, action)
```

Invalid postprocessing pattern:

```python
if action == "littering":
    ...
```

## Warning Visualization

For confirmed `urinate` violation:

```text
Draw warning around the person.
Display text similar to:
"WARNING: URINATE"
```

For confirmed littering violation:

```text
Draw warning only from the litter object-event branch.
Do not use STGCN result for littering warning.
```

For pending candidates:

```text
Do not draw them as final confirmed violations.
Use debug-only overlay if needed and if a debug flag exists.
```

---

# Validation Clips

Use exact clips when relevant instead of substitutes:

- Litter confirmation regression: `resize.mp4`, `manyFast.mp4`, `resize5.mp4`, `1fast.mp4`, `success.mp4`.
- STGCN/action regression: `urinate.mp4`, `normal_case1.mp4`, `normal_case2.mp4`, `best_urinate.mp4`.
- Use litter clips only for tracker-based litter confirmation, not STGCN littering action.
- For `scripts-heetah/` detector-vs-pipeline mismatch, compare root `test.py` raw detector result with integrated pipeline before blaming weights.

Useful lightweight validator:

```bash
conda run -n rtdetr python validate_old_test_videos.py --expect-positive resize.mp4 manyFast.mp4
```

For code-only edits, at minimum run targeted compile:

```bash
conda run -n rtdetr python -m py_compile scripts-old-test/main.py scripts-old-test/detect.py scripts-old-test/litterTracker.py scripts-old-test/action.py
```

If output video integrity is part of task, verify final MP4 is decodable with `ffprobe` or OpenCV read.

## Required Regression Expectations

### Case 1: Normal person

Expected:

```text
YOLO-Pose extracts keypoints.
STGCN predicts normal.
No violation warning is drawn.
```

### Case 2: Urinating person

Expected:

```text
YOLO-Pose extracts keypoints.
STGCN predicts urinate.
Temporal confirmation verifies sustained evidence.
If confirmed, draw urinate warning.
```

### Case 3: Person throws trash

Expected:

```text
YOLO-Pose + STGCN must not classify this as littering.
STGCN may output normal.
Littering must be confirmed by litter object-event branch only.
```

### Case 4: Litter object appears near vehicle

Expected:

```text
Litter detector finds candidate.
Motion / temporal / holding checks confirm event.
System backtracks possible thrower.
System associates thrower with vehicle/scooter.
OCR reads license plate.
Annotated output video shows littering violation.
```

---

# Performance Policy

- Rank bottlenecks before changing knobs.
- For `scripts-old-test/`, first suspects:
  - actor `YOLO.track()` / actor batch path,
  - `RTDETR.predict()` or YOLO litter branch,
  - YOLO-Pose extraction inside STGCN branch,
  - STGCN sequence inference,
  - OCR preload/background work,
  - video encode/decode.
- Use existing profiler labels when possible. Do not replace grouped timing output with noisy ad hoc prints.
- Keep expensive branches optional or gated: STGCN/action, OCR, debug overlays, full per-frame repair.
- TensorRT engines should be preferred only when sibling `.engine` exists and runtime smoke passes. `--no-engine` must remain a valid fallback.
- Do not reintroduce RTMW to solve performance or stability issues unless explicitly requested.

---

# Training / Artifacts

- Do not guess best checkpoints. Rank from `results.csv`, `args.yaml`, checkpoint paths, or evaluation scripts.
- `train-fusion.py` changes must anchor on observed best run artifacts, not generic hyperparameters.
- If user asks YOLO vs RTDETR training control, preserve explicit target selection (`yolo`, `rtdetr`, `both`) if present.
- STGCN/STGCN++ training consumes generated `.pkl` pose annotations. MP4 files are preprocessing input, not direct training input.
- `all_videos.txt` / split text files are traceability and split lists; config `ann_file` points to actual `.pkl` used by MMACTION2.
- For STGCN/STGCN++ pose work, default to YOLO-Pose keypoints.
- Treat RTMW / wholebody notes as legacy unless the user explicitly asks for RTMW again.
- STGCN training should target only `normal` and `urinate` classes.
- Do not include `littering` as an STGCN class.
- Littering training and evaluation should belong to the object-detection / tracker branch, not the STGCN action branch.

---

# `rtmw_test_image.py`

- Treat `rtmw_test_image.py` as legacy / visual test utility unless the task explicitly asks about RTMW.
- Do not connect `rtmw_test_image.py` back into the current production STGCN keypoint path.
- Keep image/video/GIF behavior symmetric where possible.
- `--keypoints-only` means only keypoints/skeleton: remove boxes, labels, scores, and extra text unless explicit opt-in.
- `--show-label` is opt-in for top-left mode/action/score label.
- Do not add labels back by default after user requests minimal visuals.

---

# Root `test.py`

- Treat as module-by-module smoke/eval harness, not production pipeline.
- Preserve independent toggles for YOLO, RTDETR, STGCN, Pose, PaddleOCR.
- Use it to separate raw model capability from tracker/postprocess bugs.
- If a task touches STGCN action, verify whether `test.py` uses YOLO-Pose or legacy RTMW before trusting its result.

---

# Git / Dependencies

- This workspace may contain many generated outputs. Avoid unrelated formatting or metadata churn.
- Do not run dependency installs or network downloads without approval.
- If a required command fails due sandbox/network/package access, request escalation with the exact command reason.
- Do not modify vendored `mmaction2/` or `mmpose-rtmw/` broadly; patch minimal import/config drift only when needed.
- Do not remove legacy dependency folders just because RTMW is no longer used in the current workflow.

---

# Acceptance Criteria

The implementation is considered correct if:

1. RTMW is not loaded or used anywhere in the current keypoint extraction pipeline.
2. YOLO-Pose is the only source of person keypoints for STGCN.
3. STGCN only outputs `normal` or `urinate`.
4. STGCN no longer outputs or handles `littering`.
5. Littering detection is handled only by the litter object-event branch.
6. Urinate warnings are triggered only after temporal confirmation.
7. Littering warnings are triggered only after object-event confirmation.
8. The output video still correctly draws warning boxes for confirmed violations.
9. Terminal logs clearly separate:
   - actor detection,
   - YOLO-Pose keypoint extraction,
   - STGCN normal / urinate classification,
   - litter object detection,
   - litter behavior confirmation,
   - OCR result.
10. Code-only edits pass targeted `py_compile`.
11. Video-output edits produce a decodable annotated MP4.
