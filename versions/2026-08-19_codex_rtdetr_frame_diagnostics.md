# RT-DETR 垃圾辨識逐幀診斷

- 日期：2026-08-19
- 作者：Codex
- Branch：`heetah-dev`
- Commit：未提交；base `f183e47`
- 類型：feat

## 問題背景

Annotated MP4 只顯示已通過 `GlobalLitterTracker` 的 confirmed 垃圾。原本 pipeline 雖在
記憶體內累積 raw／filtered candidate 總數，精簡 analysis JSON 卻只輸出 confirmed
events，因此零事件影片無法判斷 RT-DETR 沒有輸出，或 candidate 被 geometry、motion、
holding、tracker confirmation 淘汰。

## 實作內容

- 在 RT-DETR parse 階段、基本 geometry 通過後、motion/holding 通過後，分別記錄
  candidate observation 總數。
- 每個有 candidate 的幀保存 0-based `frame_index` 與該幀 `candidate_count`。
- Analysis JSON 新增 `litter_detection`，同時輸出 RT-DETR evaluated/skipped 幀數與
  confirmed event 數。
- 保留既有 `raw_litter_candidates` 內部欄位語意，避免舊工具失效；新增真正位於 geometry
  之前的 `rtdetr_litter_candidates`。
- UI backend 同時接受舊 `2.0.0` 與新 `2.1.0`，既有 analysis JSON 仍可匯入。

## API／Config／Schema 變更

- Analysis schema：`2.0.0` → `2.1.0`。
- 新增頂層 `litter_detection`：
  - `rtdetr_4channel`：enable、`TRASH_CONF`、evaluated/skipped 幀與原始模型 candidate。
  - `geometry_passed`：通過 bbox size/aspect ratio 的 candidate。
  - `motion_holding_passed`：通過 camera-shake、motion、core-motion、vehicle FP、holding gate。
  - `confirmed_event_count`：最終 confirmed 事件數。
- 沒有新增或修改環境變數；MP4 仍只渲染 confirmed litter。

## 測試證據

- `tests/pipeline/test_events.py`：`13 passed`。
- `tests/pipeline/test_detect_characterization.py`：`4 passed`；包含 RT-DETR 有 bbox、但
  geometry 淘汰仍保留原始幀診斷的案例。
- `tests/ui/test_backend_core.py`：`11 passed`；包含 schema `2.1.0` 讀取相容性。
- 完整 `tests/pipeline`：`172 passed, 12 skipped`。
- 完整 `tests/ui`：`15 passed`。
- `python -m py_compile`：`scripts/main.py`、`scripts/pipeline/detect.py`、
  `scripts/pipeline/events.py`、`UI/backend/analysis.py` 通過。
- `scripts/frontend/sample_analysis.json` 經 `python -m json.tool` 通過。

## 已知限制

- Candidate 是跨幀 bbox observation，不是去重後的垃圾實體，也不是 accuracy。
- Vehicle gate 略過的幀未交由 active litter branch 評估，不能把零 candidate 解讀為漏檢。
- 本次未執行真實影片／GPU inference；尚未驗證特定影片產生的新 analysis JSON 數值。
- React 目前仍顯示 confirmed events，新的逐幀診斷先提供 JSON 與 API 消費者使用。

## 回滾方式

還原 `scripts/pipeline/detect.py` 的逐幀 stats、`scripts/main.py` 的 run summary 欄位、
`scripts/pipeline/events.py` 的 `litter_detection` builder 與 schema version，並移除 UI
`2.1.0` 支援、對應測試、範例與文件。此回滾不需要改模型權重或環境變數。
