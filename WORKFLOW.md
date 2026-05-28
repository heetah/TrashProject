# scripts-old-test Workflow

此文件為 UTF-8 更新版工作流程說明。實作目標：`scripts-old-test/` 只用 YOLO-Pose + STGCN 判斷 `normal` / `urinate`，亂丟垃圾只由 litter object-event branch 確認。

## 核心合約

- Entry：`scripts-old-test/main.py`
- Runtime env：`conda run -n rtdetr ...`
- Actor detection：YOLO-Seg，解析 `person` / `vehicle` / `scooter`
- Pose source：YOLO-Pose only
- STGCN classes：`normal` / `urinate`
- Litter detection：RT-DETR / YOLO litter detector，輸出只當候選
- Litter confirmation：`GlobalLitterTracker` 依 motion / temporal / holding evidence 確認
- OCR：只在 confirmed violator vehicle/scooter 上啟動
- RTMW：legacy，不進 production keypoint path

## 高階流程

```text
Input video
  -> model preload
  -> background frame reader
  -> frame queue
  -> main inference loop
  -> YOLO-Seg actor detection
      -> person branch
          -> YOLO-Pose keypoint extraction
          -> keypoint history by track_id
          -> STGCN after window full
          -> output normal / urinate only
      -> vehicle / scooter branch
          -> track vehicle-like actors
          -> associate person with vehicle/scooter
      -> litter branch
          -> RT-DETR / YOLO litter candidate detection
          -> motion evidence
          -> temporal difference
          -> litter_holding()
          -> confirmed littering event
          -> backtrack thrower / vehicle
          -> OCR plate lookup
  -> draw confirmed warnings only
  -> write annotated MP4
```

## Person Action Branch

`scripts-old-test/action.py` owns STGCN state.

Expected output:

```python
{
    "track_id": person_id,
    "action": "normal" | "urinate",
    "conf": float,
    "stgcn_conf": float,
    "alert": bool,
}
```

Rules:

- `ACTION_CLASSES = {0: "normal", 1: "urinate"}`
- `urination` / `urinating` are accepted only as legacy aliases and normalized to `urinate`.
- STGCN never outputs or confirms `littering`.
- Urinate alert needs sustained evidence, default current code policy:
  - `--urination-window-sec 10.0`
  - `--urination-min-sec 8.0`
- Person linked to vehicle/scooter suppresses and clears urinate state.

## Litter Object-Event Branch

`scripts-old-test/litterTracker.py` owns confirmed littering.

Rules:

- Litter detector bbox is candidate only.
- Confirmed littering needs tracker / behavior evidence.
- Keep these evidence signals:
  - physical trajectory validation
  - horizontal movement
  - downward Y movement
  - ROI/background motion evidence for 1-2 frame detections
  - polygon-aware `litter_holding()`
  - same-id history when requested
- Pending litter candidates must not render as confirmed warnings.

## Runtime Defaults

```bash
conda run -n rtdetr python scripts-old-test/main.py resources/resize.mp4 --batch 8
```

Important switches:

```bash
--disable-action       # skip YOLO-Pose + STGCN branch
--disable-plate        # skip OCR branch for litter-only speed work
--action-window 30     # STGCN sequence length
--action-threshold 0.5 # raw STGCN urinate confidence threshold
--no-engine            # force .pt weights when TensorRT engine is bad
```

Current 2-class STGCN deploy path:

```text
modules_weight/stgcnpp_urinate_2class_best.pth
```

If this file is missing, action module falls back instead of loading old 3-class weights.

## Training Refresh

Use YOLO-Pose26x to rebuild two-class pose data and train STGCN++:

```bash
scripts/train_yolo_pose_urinate_stgcnpp.sh
```

Default source:

```text
/mnt/8tb_hdd/under115a/dataset-pose/real/{normal,urinate}
```

Outputs:

```text
dataset-pose/garbage_raw.pkl
dataset-pose/garbage_final.pkl
dataset-pose/garbage_final_balanced.pkl
dataset-pose/stgcn_balanced_metadata.json
modules_weight/stgcnpp_urinate_2class_best.pth
```

## Validation

Code-only check:

```bash
conda run -n rtdetr python -m py_compile \
  scripts-old-test/main.py \
  scripts-old-test/detect.py \
  scripts-old-test/litterTracker.py \
  scripts-old-test/action.py
```

Litter confirmation regression:

```bash
conda run -n rtdetr python validate_old_test_videos.py --expect-positive resize.mp4 manyFast.mp4
```

STGCN/action regression:

```text
urinate.mp4
normal_case1.mp4
normal_case2.mp4
best_urinate.mp4
```

Expected:

- normal clips：keypoints may exist, no warning.
- urinate clips：STGCN predicts `urinate`, then sustained evidence confirms.
- litter clips：STGCN does not output `littering`; litter warning comes only from confirmed litter event.

