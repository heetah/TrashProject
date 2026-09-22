# 統一 Python runtime requirements

- 日期：2026-08-24
- 作者：Codex
- Branch：working tree
- Commit：未提交
- 類型：chore / docs

## 問題背景

Repository 沒有 root production dependency manifest，README 指定的 `rtdetr` conda
environment 無法由 live code 重建。UI backend 及 vendored MMAction2 各自只有局部依賴清單。

## 實作內容

- 依 `scripts/`、`scripts/pipeline/`、`UI/backend/`、`tests/` 與實際 STGCN config
  的 imports 建立 root `requirements.txt`。
- 固定 Python 3.11 的 production、STGCN、PaddleX OCR、UI backend 與測試依賴。
- README 新增 `rtdetr` environment 建立與安裝指令。
- `tensorrt`、`transformers`、legacy `paddleocr`、generic MMAction `decord` 與模型
  export 套件保持 optional；STGCN production 直接接收 skeleton dict。

## API／Config／Schema 變更

無 runtime API、環境變數或 analysis schema 變更。

## 測試證據

- Miniconda 26.5.3 安裝於 `/home/under115a/miniconda3`；`rtdetr` 使用 Python 3.11.16。
- `python -m pip check`：`No broken requirements found.`
- Torch 2.13.0+cu130：識別 RTX 5090 / compute capability 12.0，CUDA matrix smoke 通過。
- 現有權重 dummy inference：YOLO-Seg、YOLO-Pose、RT-DETR 4-channel、plate detector 均通過。
- STGCN checkpoint：CUDA 載入成功，100-frame skeleton inference 輸出 `(2,)` finite scores。
- PaddleX `en_PP-OCRv5_mobile_rec`：CPU preload 與 dummy OCR schema 通過。
- Compile smoke：通過。
- `LITTER_DEBUG=0 python -m pytest -q tests/pipeline`：183 passed、12 skipped。
- `python -m pytest -q tests/ui`：15 passed。
- Node 22.23.2 / npm 10.9.8：`npm ci`、`npm audit`（0 vulnerabilities）與 Vite
  production build 通過。

## 已知限制

- TensorRT engine 與完整影片 inference 需另外以實際 engine/clip 驗證。
- Root requirements 只涵蓋 live production、UI backend 與 repository tests，不涵蓋
  vendored MMAction2 的 docs/full upstream test extras。
- 額外執行整個 `tests/` 為 208 passed、12 skipped、2 failed；兩個失敗是既存
  `tests/test_litter_regression.py` positive cases 保持 pending，非 dependency/import 失敗。
- 本機 `.env` 設為 `LITTER_DEBUG=1`；完整 pipeline suite 驗證以 shell 明確覆寫
  `LITTER_DEBUG=0`，不修改使用者本機設定。

## 回滾方式

移除 root `requirements.txt`，並還原 README 的 Python 依賴段落。
