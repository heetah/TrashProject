# Production root `.env` 集中設定

- 日期：2026-08-19
- 作者：Codex
- Branch：`heetah-dev`
- Commit：未提交（base `d195e02`）
- 類型：refactor / docs / test

## 問題背景

Production pipeline 原本只在 `PipelineConfig` 集中部分數值；模型路徑仍硬編碼在
`scripts/main.py` 與 `scripts/pipeline/plate.py`，其他 action、plate、motion、device、
video I/O 與 Smart Backtrack 環境變數分散在使用點，部署者難以確認完整可調項目。

## 實作內容

- 新增 root `.env` 載入器，於 import Torch/Ultralytics 前載入，且不覆蓋 shell 或 UI worker
  已明確 export 的設定。
- 新增可提交的 `.env.example`，集中列出 production 目前全部環境變數；root `.env` 加入
  `.gitignore`，本機可直接保存模型路徑與調參值。
- 將 YOLO-Seg、RT-DETR、YOLO-Pose、STGCN 模型路徑、TensorRT 偏好、RT-DETR enable、
  detector confidence、motion mask、video capture、writer 與 output 設定收進 `PipelineConfig`。
- 車牌 detector／OCR 的模型路徑、model name、detector confidence、接受門檻與掃描間隔改由
  `.env` 控制；低信心仍保留失敗狀態，不產生虛構車牌。
- YOLO-Pose confidence 與 bbox normalization pad/confidence 改由 `.env` 控制。
- TensorRT export 工具與 production 共用相同模型路徑與 smoke confidence，避免匯出另一份
  checkpoint。

## API／Config／Schema 變更

- 新增 `load_env_file()`、`load_project_env()` 與 `PipelineConfig.model_paths_for_batch()`。
- 新增 `PIPELINE_ENV_FILE` 外部 selector；優先序為 exported env > dotenv > code fallback。
- 新增 model、plate、motion、video 與 profiling 環境變數，完整清單見 root `.env.example`。
- Analysis JSON schema 與事件／歸因／OCR 證據責任不變。

## 測試證據

- `conda run -n rtdetr python -m pytest -q tests/pipeline`：`170 passed, 12 skipped`。
- `conda run -n rtdetr python -m pytest -q tests/ui`：`14 passed`。
- `conda run -n rtdetr python -m py_compile ...`：main、export、config、action、detect、
  litter tracker、plate compile 通過。
- `conda run -n rtdetr python scripts/export_tensorrt.py --help`：CLI parse/import smoke 通過。
- Production import smoke 確認載入 root `.env`，batch 8 解析到既有 batch actor checkpoint 與
  `best-rtdetr-4c-background.pt`，`BBOX_CONF=0.45`、`TRASH_CONF=0.4`、`OUTPUT_ROOT=.` 未漂移。

## 已知限制

- 未執行真實影片、GPU inference、TensorRT export 或 OCR 模型 runtime；本次測試不代表
  detector accuracy、事件確認、歸因準確率、OCR 正確率或效能。
- `.env` loader 支援一般 `KEY=VALUE`、引號與 `export KEY=VALUE`；不做 shell expansion。
- 低頻物理 gate、固定 class mapping 與 RT-DETR 4-channel preprocessing contract 保留在
  live code，未全部外露成一般部署設定。

## 回滾方式

回滾本次 atomic change；或在保留程式時移除 root `.env`，pipeline 會使用
`PipelineConfig` 與各模組原 production fallback。不得刪除模型、輸出或人工複核資料。
