# Distance-Time 1:1 Backtrack Baseline

- 日期：2026-08-03
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交
- 類型：test

## 問題背景

研究先固定觀察三個成本矩陣的距離與時間，避免 confidence、uncertainty 與
trajectory feature 同時改變而無法解釋結果。

## 實作內容

- `distance_time` stage 的 `C_BA` release distance/time、`C_BC` direct
  distance/time、`C_AC` endpoint distance/time 全部設為 1.0:1.0。
- 新增 `SMART_BACKTRACK_STUDY_STAGE`，未設定或非法值皆保持 `full` production
  行為；設定 `distance_time` 時以 birth anchor 與 observed actor bbox 產生基準。

## API／Config／Schema 變更

```text
SMART_BACKTRACK_STUDY_STAGE=distance_time
```

## 測試證據

- Runtime：2026-08-03，以 GPU TensorRT engines 實跑
  `/mnt/8tb_hdd/under115a/litter_vidshort/litter/` 的 50 支 MP4：
  `litter_case_1..21`、`litter_case_23..51`（case 22 為 `.avi`，未納入）。
- 固定環境：`SMART_BACKTRACK=1`、
  `SMART_BACKTRACK_STUDY_STAGE=distance_time`；case 1--3 保留 sidecar，
  case 4--51 關閉可選 sidecar 以避免多 GB 寫入拖慢人工檢閱輸出。
- 50/50 pipeline run 產出 summary，49 個 batch child process 均為 exit status 0
  （case 1 為 batch 前完成的 smoke/runtime run）。
- `ffprobe` 驗證 50/50 annotated MP4 都有可解碼的 video stream。
- Summary route coverage：45 confirmed events；45 resolved non-NULL、0 dustbin。
  31 支影片至少有一個 resolved route；0 支影片有 confirmed event 但全為
  NULL；19 支影片沒有 confirmed litter event。可檢閱輸出與可重算 report：
  `output/backtrack_dt_1to1_50_20260803_gpu/run_report.json`。
- 本機缺少 OCR detector weight，run log 顯示 plate OCR disabled；不影響
  confirmed-litter 或 Smart Backtrack route 計數。
- Runtime second batch：以兩張 RTX 4090 平行實跑 52--156、158--200；
  `litter_case_157.mp4` 不存在，故共 148 支。148/148 pipeline child process
  exit status 0，且 148/148 annotated MP4 通過 `ffprobe` decode smoke。
- Second batch summary route coverage：202 confirmed events；195 resolved non-NULL、
  7 dustbin。86 支影片純 resolved、1 支全 NULL、3 支 mixed resolved/NULL、
  58 支沒有 confirmed litter event。可檢閱輸出與可重算 report：
  `output/backtrack_dt_1to1_200_20260803_gpu/run_report.json`。

## 已知限制

此實驗產出的 resolved/unattributed 數量是 route coverage，不是人工 reviewed
attribution accuracy。

## 回滾方式

移除 `SMART_BACKTRACK_STUDY_STAGE` 即回到原 full Smart Backtrack。
