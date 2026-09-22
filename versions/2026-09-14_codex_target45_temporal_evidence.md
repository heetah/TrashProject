# Litter confirmed-event 45/58 temporal evidence

- 日期：2026-09-14
- 作者：Codex
- Branch：`heetah-dev`
- Commit：`a149c5975898643e5131caf0579bd0aefe228d75` + tracked working patch `dc2a63d4ba2274b29db6ac6d13dd0c5b62c15da74420daab90d25b7a73e04be7`
- 類型：feat / fix / test / docs

## 問題背景

既有候選在 58 個 reviewed positive clips 上有 41 個 assignment-conditioned accepted
event matches。主要漏失包含快速小物件 bbox 形變造成 ID 斷裂、RT-DETR 中間漏一幀、
vehicle-contained 軌跡無法使用成熟的 release observation，以及沒有 actor detection、但物體
軌跡本身具有明確重力弧線的事件。目標是在不降低全域 detector/confirmation threshold、
不強制 actor attribution 的前提下達到至少 45/58。

## 實作內容

- Pending bbox 尺寸不一致時，以短時間 constant-velocity residual 和 combined bbox diagonal
  檢查 motion continuity；streak probation 同樣使用 prediction residual，避免把正常形變當成
  identity jump。
- 對兩個真實 detector anchors 後的 quarantine 軌跡，允許一個不可連鎖的 grayscale
  change-component observation；sidecar 明確標成 `litter_visual_bridge`，且 detector observation
  count 不增加。
- 對畫面占比至少 `0.001` 的單一大型 detector seed，允許最多四個 prediction-gated visual
  observations；四點完成前不能 confirmation，最後仍需通過 holding、trajectory、displacement
  與 stationary gates。
- 成熟 quarantine 可使用第一個 detector-only、向下分離的 ordinary observation 解除；包含
  visual bridge 的歷史不能授權後續 stream handoff，完整 missed-window 後的新 observation 必須
  建立獨立 identity。
- 無 actor 的物體只有在至少五個 observation、有效物理軌跡、足夠位移以及明確內部
  apex/descent 時確認 event；attribution 保持獨立，可輸出 `NULL`。

## API／Config／Schema 變更

- 沒有新增 CLI 或環境變數，production entrypoint 不變。
- Litter candidate sidecar 新增 `record_type=litter_visual_bridge`，`evidence_source` 為
  `temporal_component_after_two_detector_anchors` 或 `bounded_large_seed_temporal_chain`。
- Tracker 內部資料新增 `history_sources`、`detector_observation_count`、
  `birth_frame_area_ratio`；不改變對外 event schema。

## 測試證據

- `python -m py_compile`：`litter_tracker.py`、`geometry.py`、`detect.py` 通過。
- Targeted：`75 passed`，涵蓋 visual bridge、large-seed chain、actorless arc、quarantine
  release、stale identity 與既有 litter regression。
- Pipeline suite：`387 passed, 12 skipped`。
- 完整批次：58/58 jobs `completed`；58/58 annotated MP4 可由 OpenCV 開啟並讀取首幀。
- Readiness：45/58 accepted event matches，77.59%，Wilson 95% CI 65.34%--86.41%；
  相對 41/58 基準新增 cases 23、30、69、75，loss=0。
- Per-GT bbox calibrator：confirmed correct gain=6、loss=0；未驗證 confirmed tracks 36→33。
- 證據：`artifacts/target45_full_20260913/experiment_manifest_v2.json`、
  `artifacts/target45_full_20260913/readiness_v2/`、
  `artifacts/target45_full_20260913/calibration_v2/`、
  `output/groundtruth_batch_target45_v2_20260914/batch_manifest.jsonl`。

## 已知限制

- 58 clips 全為正樣本，不能估計 precision 或 false-positive rate；多 confirmed records 只能列為
  safety warning，不能在沒有 reviewed negatives 時裁定為 false positive。
- 規則使用同一批 development cases 建立，沒有獨立 camera-group holdout，generalization 未驗證。
- 45/58 是 selected route 的 release frame/point 參與配對的 event sensitivity，不是純 RT-DETR
  detector recall。
- Route correctness 為 31/58；case 23 actor mapping unavailable、case 30 wrong route。Plate OCR
  沒有 reviewed ground truth，因此不代表可開罰案件正確率，`ready_for_enforcement=false`。

## 回滾方式

回滾本次 atomic change 時，移除 `litter_tracker.py` 的 motion association、visual bridge、
actorless arc 與 quarantine identity guards，移除 `detect.py` 的 bridge sidecar forwarding，並將
`geometry.py` streak probation 恢復為 raw-step 判斷；同時回滾對應測試與本 README 段落。
不要刪除既有輸出或研究 artifacts；它們是 paired regression 的 provenance。
