# PreparedFrame 有界前處理 Pipeline

- 日期：2026-08-19
- 作者：Codex
- Branch：heetah-dev
- Commit：基準 `bc631ac`；變更未提交（working tree）
- 類型：perf / refactor / test / docs

## 問題背景

Production 已有 `PIPELINE_BATCH=8`、非同步影片讀取與 FFmpeg writer，但 RT-DETR 的
`[R, G, B, change]` input 仍在主推論執行緒、每次 `predict()` 前建立。CPU 的 BGR→RGB、
相鄰幀差分、normalize 與 channel stack 因此不能和前一批 GPU inference 重疊。

參考 `/mnt/8tb_hdd/under115a/pipeline-version-with-ui` 的 bounded queue 與
`PreparedFrame` 契約，只移植不改變現行 evidence chain 的部分；不搬入該專案的 tracker、
事件判定或最近 actor attribution。

## 實作內容

- 新增 frozen `PreparedFrame`，明確保存 frame index、未標註 BGR、foreground mask 與可選
  4-channel model input。
- `AsyncVideoFrameReader` 在背景 thread 依序完成 decode、motion mask 與 4-channel input，
  並以有界 queue 提供下一批；保留舊 `read_batch()` 兩欄介面。
- 主流程改用 packet index 驗證順序，再將 prepared inputs 送入 `detect_batch()`。
- RT-DETR 正常 batch、batch exception repair、mixed-zero repair 共用同一份 prepared input，
  不重做前處理。
- Profiler 新增 `frame.litter_input_prepare`，其累積時間可能與主流程重疊，不可和 wall time
  直接相加。

## API／Config／Schema 變更

- 新增 `PIPELINE_QUEUE_SIZE`，預設 `8`。
- 新增 `PIPELINE_PREPARE_4C`，預設 `1`；設為 `0` 可回退原本主執行緒即時前處理。
- `detect_batch()` 新增 optional `prepared_litter_inputs`；既有 caller 不傳時行為不變。
- Analysis JSON schema、事件 evidence、Smart Backtrack 與 OCR schema 無變更。

## 測試證據

- `conda run -n rtdetr python -m py_compile ...`：通過。
- `conda run -n rtdetr python -m pytest -q tests/pipeline`：`166 passed, 12 skipped`；
  包含 final 4-channel tensor contract、prepared input 不重算、queue 跨 batch 相鄰幀與
  frame index 順序。
- 實際 A/B：GPU 1 `NVIDIA GeForce RTX 4090`，輸入
  `litter_case_100.mp4`（2592×1944、12 FPS、72 frames），兩邊均使用
  `PIPELINE_BATCH=8`、`PIPELINE_QUEUE_SIZE=8`，只切換 `PIPELINE_PREPARE_4C=0/1`。
- Warm-cache baseline：video loop `9.800s`、`7.35 FPS`；prepared：`6.335s`、
  `11.36 FPS`。此樣本 throughput 為 `1.55x`，video-loop wall time 減少 `35.4%`。
- 同一輪以 `nvidia-smi` 250ms 取樣 video-loop window：平均 GPU utilization 約
  `6.5% -> 9.8%`（約 `+3.3` percentage points），峰值 `26% -> 28%`。這是粗粒度
  device sample，不是 CUDA kernel trace。
- 兩邊輸出 MP4 均為 9,057,649 bytes，SHA-256 同為
  `249c56300ecf29b89597ae187f3a78ad7982eaf4d8e4207a675972c6632c711d`；analysis JSON
  欄位與事件內容相同。輸出位於 `/tmp/trashproject-pipeline-ab/`。

## 已知限制

- Queue overlap 只表示不同批次的 CPU 前處理可與 inference 重疊，不證明 CUDA kernel concurrency。
- 目前只有單支 72-frame 短片 A/B；不能外推成全 corpus、其他解析度或含大量 person/STGCN/OCR
  場景的固定提升。GPU utilization 是 `nvidia-smi` 粗粒度取樣，不等同 Nsight/CUDA trace。
- Vehicle gate 關閉昂貴路徑的批次仍可能已在 reader 做完 4-channel CPU 前處理；若該場景 CPU
  或 RAM 更重要，可設 `PIPELINE_PREPARE_4C=0`。

## 回滾方式

執行時設 `PIPELINE_PREPARE_4C=0` 即回到原本 RT-DETR 呼叫前即時建立 input；完整回滾則移除
`PreparedFrame` reader 路徑與 `detect_batch(prepared_litter_inputs=...)`，保留既有 async
reader/writer 與 batch inference。
