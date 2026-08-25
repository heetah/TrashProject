# Flask + React 持久化人工複核介面

- 日期：2026-08-06
- 作者：Codex
- Branch：`heetah-dev`
- Commit：尚未提交
- 類型：feat

## 問題背景

Production `scripts/main.py` 只能一次處理一支影片並輸出每影片獨立的 annotated MP4
與 schema 2.0.0 analysis JSON。既有靜態 dashboard 無法排程上傳／資料夾、多影片切換，
也沒有可跨重啟保存的人工審核狀態。

## 實作內容

- 新增獨立 `UI/`，不搬動或複製 production pipeline。
- Flask 使用 SQLite 保存 queued/running/completed/failed 工作與逐事件人工審核。
- 單一 background worker 依序以既有 positional CLI 呼叫 `scripts/main.py`；每個 job
  使用獨立輸出資料夾，避免同名影片互相覆蓋。
- 支援 multipart 上傳與 `.env` allowlist server 資料夾的多影片排程。
- 可掃描既有 `*_annotated_analysis.json`，同一路徑以 deterministic id 去重。
- React 提供未審核／已審核兩頁、可捲動影片清單、上一支／下一支、影片 seek、
  confirmed event、模型 confidence、歸因／NULL、車牌／OCR 證據與逐項人工判定。
- 移除頂部 banner 與播放器下方檔名，縮小全頁字級與卡片間距；桌面版改成左側影片／
  摘要、右側全部事件與審核表單的雙欄工作區，右欄可獨立捲動，窄螢幕回到單欄。
- 將每個事件的證據與人工判定合併為同一卡片；車牌旁新增鉛筆按鈕，可保存或移除
  SQLite 人工修正版，AI analysis JSON 與原始 OCR 值保持不變。
- 沒有 AI event 的影片仍建立人工項目，用於檢查可能漏判。

## API／Config／Schema 變更

- 新 API 詳見 `UI/README.md`；production analysis schema 未修改。
- 新增 `UI/.env.example`，包含 paths、upload size、polling、conda env、pipeline batch、
  worker 與 Vite proxy 設定。
- 人工審核只寫入 `UI/data/ui.sqlite3`，不回寫 analysis JSON 或 research sidecar。
- 新增 `plate_corrections` table 與車牌修正 PUT/DELETE API；車牌修正不會改變事件審核狀態。

## 測試證據

- `conda run -n rtdetr python -m pytest -q tests/ui`：10 passed。
- `conda run -n rtdetr python -m pytest -q tests/pipeline`：142 passed、12 skipped。
- `conda run -n rtdetr python -m py_compile UI/backend/*.py`：通過。
- `cd UI/frontend && npm run build`：Vite 8.2.0 production build 通過。
- `npm audit --json`：0 vulnerabilities。
- 暫時以 `UI_WORKER_ENABLED=0` 啟動 Flask：`/api/health`、`/api/config` 與 React
  `index.html` 均回傳 HTTP 200。

未執行真實影片 GPU inference；本變更沒有修改 detector、tracker、backtrack 或 OCR。

## 已知限制

- 目前為單機、單 Flask process、單 GPU worker，沒有多使用者認證或分散式 queue。
- 只有工作狀態，沒有逐 frame 進度百分比。
- Confidence 只是模型分數；人工審核資料也不等於依法完成可開罰案件。

## 回滾方式

移除 `UI/`、`tests/ui/` 與本次 README/AGENTS 章節即可；production `scripts/` 未改動。
