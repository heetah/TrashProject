# TensorRT 加速環境與 litter case smoke benchmark

- 日期：2026-08-24
- 作者：Codex
- Branch：working tree
- Commit：未提交
- 類型：chore / perf / docs

## 問題背景

`rtdetr` environment 已具備 production 基礎依賴，但缺少 live code 優先使用的
TensorRT runtime 與可重現 engine export toolchain。既有 `.engine` 使用舊 serialization
version 240，無法由 TensorRT 11.2 的 version 243 runtime 載入。

## 實作內容

- 在 root `requirements.txt` 固定 TensorRT 11.2.1.2、NVIDIA ModelOpt 0.46.0、
  ONNX 1.21.0、ONNX Runtime GPU 1.24.4 與 Ultralytics tracking 所需的 lap 0.5.13。
- 在 `rtdetr` conda environment 安裝上述套件；ONNX Runtime 可見 TensorRT、CUDA、
  CPU execution providers。
- 於 RTX 5090 重建 production actor batch-4 與 pose batch-1 FP16 TensorRT engines。
- 舊 actor／pose engines 分別保留為 `best-yolo-seg_v3_b4.trt10_backup.engine` 與
  `yolo26x-pose.trt10_backup.engine`。
- README 同步說明 GPU runtime、export dependencies 與 `.pt` fallback。

## API／Config／Schema 變更

無 CLI、環境變數或輸出 schema 變更。`PREFER_TENSORRT=1` 與 `PIPELINE_BATCH=8` 的
既有行為保持不變。

## 測試證據

- `python -m pip check`：`No broken requirements found.`
- TensorRT 11.2.1.2：RTX 5090 實際 deserialize 與 inference 通過。
- Actor engine：batch 4 實際 frame inference 通過，每張 frame 產生 5 boxes。
- Pose engine：batch 1 實際 frame inference 完成；該測試 frame 沒有 pose boxes。
- `python -m py_compile`：`scripts/main.py`、`scripts/export_tensorrt.py`、
  `scripts/pipeline/config.py`、`scripts/pipeline/infra/models.py` 通過。
- 端到端影片：`litter_case_25.mp4`，1440x1080、30 FPS、60 幀、2.012 秒。
- 乾淨重跑：exit status 0；pipeline internal wall 7.867 秒；影片 loop 2.255 秒，
  26.60 frame/s；外部 wall 12.43 秒（含 conda process 啟動與輸出 flush）。
- 輸出 MP4：H.264、1440x1080、60 幀、2.000 秒，ffprobe decode metadata 通過。
- Analysis JSON：可解析，包含 `events`、`litter_detection`、`summary`、`video`。

## 已知限制

- Production `best-rtdetr-4c-background.pt` 的 batch-8 FP16 export 在 ModelOpt reference
  pass 因系統記憶體不足被 exit 137 終止，因此 litter branch 正確回退 `.pt`；未產生或
  啟用半成品 engine。
- Benchmark 執行時 GPU 另有約 10.4 GB 使用量，數字是共享 GPU 狀態，不是隔離效能結果。
- 測試 clip 產生 16 個 RT-DETR candidates、5 個 motion-filtered candidates，但沒有
  confirmed litter event。此 smoke 只證明 pipeline、engine 與輸出路徑可運作，不代表
  detection／attribution accuracy。
- `mmcv-lite` 的 `MultiScaleDeformableAttention` warning 不影響 production STGCN skeleton
  inference；若未來 live code 使用該 op 才需評估完整 `mmcv` CUDA build。

## 回滾方式

從 `requirements.txt` 移除 acceleration/export group 與 lap；移除新建的 actor／pose
engines，再將兩個 `.trt10_backup.engine` 還原原檔名。由於舊 engines 與 TensorRT 11.2
不相容，回滾後須同時改回相容的 TensorRT runtime，否則 live code 會回退 `.pt`。
