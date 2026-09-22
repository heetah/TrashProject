# 每影片精簡 analysis JSON

- 日期：2026-08-05
- 作者：Codex
- Branch：`heetah-dev`
- Commit：未提交；base `3c261bc`
- 類型：feat

## 問題背景

前端不需要完整 pipeline diagnostics、所有 actor observation 或多份資料檔。原設計同時
輸出 `analysis.json`、`summary.json` 與 `events.jsonl`，欄位也超過目前 dashboard
能呈現的範圍。

## 實作內容

- 每支影片預設只產生 annotated MP4 與一份 `_analysis.json`。
- 移除 production 的 `summary.json`、`events.jsonl` 寫檔。
- Analysis schema 縮成 `video`、`summary`、`events` 三區。
- Dashboard 只載入 analysis JSON，不再要求 legacy summary/events。
- `SMART_BACKTRACK_SIDECAR` 預設改為 `0`；研究者可明確設 `1` 產生 JSONL。
- 新增 `scripts/pipeline/ANALYSIS_JSON.md`，記錄完整範例、欄位與證據限制。

## API／Config／Schema 變更

- Schema version：`2.0.0`，為 breaking change。
- `build_analysis_report()` 不再接受 width、height、artifacts 或 generated timestamp。
- Summary 只保留 confirmed litter/urinate 數、通行車輛估計、平均 litter confidence、
  accuracy 狀態、違規車牌與人工複核狀態。
- Event 只保留網頁會顯示的時間、confidence、vehicle、plate 與 attribution status。
- `detection_accuracy` 無 human-reviewed ground truth 時固定為 `null`；confidence 不是
  accuracy。

## 測試證據

- `conda run -n rtdetr python -m pytest -q tests/pipeline/test_events.py`：`11 passed`。
- `conda run -n rtdetr python -m pytest -q tests/pipeline`：`137 passed, 12 skipped`。
- `conda run -n rtdetr python -m py_compile scripts/main.py scripts/pipeline/events.py scripts/pipeline/litter_tracker.py`：通過。
- `scripts/frontend/sample_analysis.json` 經 `python -m json.tool`：通過。
- Dashboard JavaScript syntax check：通過。
- `git diff --check`：通過。

## 已知限制

- Unique tracker ID 可能因 fragmentation 高估通行車輛數。
- Event start/end 可能來自 estimated release 或 candidate birth，不是人工 reviewed interval。
- `resolved` 仍是模型歸因；所有 confirmed AI event 仍需人工複核。
- 舊 run 已存在的 summary/events 檔不會自動刪除。

## 回滾方式

恢復 `scripts/main.py` 的 summary/events writer 與 sidecar 預設值，還原 schema `1.0.0`
report builder、dashboard legacy loaders、README 與測試。
