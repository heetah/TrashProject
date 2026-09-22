# Flask + React 人工複核介面

`UI/` 與 production `scripts/` 分開。Flask 只負責排程、持久化、讀取每支影片的
`*_annotated_analysis.json` 與提供 annotated MP4；AI 判斷仍全部由
`scripts/main.py` 與 `scripts/pipeline/` 產生。

## 資料流

```text
上傳影片／輸入允許的 server 資料夾
  -> Flask 建立 SQLite job
  -> 單一 background worker 依序執行
       conda run -n rtdetr python scripts/main.py <video>
  -> output/ui_runs/<job-id>/
       <name>_annotated.mp4
       <name>_annotated_analysis.json
  -> Flask 驗證 schema 2.0.0／2.1.0 並提供 JSON/media API
  -> React 未審核／已審核頁
  -> SQLite 保存逐事件 accepted/rejected、審核者與備註
  -> 已審核 accepted 事件匯出 MP4 片段 + Excel ZIP
```

工作不在 Flask request handler 裡直接跑。內建 worker 一次只取一支影片，避免
多個 HTTP request 同時啟動模型並搶同一張 GPU。伺服器非正常中止時，原本
`running` 的工作會在下次啟動時重新排入佇列。

## 持久化與證據界線

- `UI/data/ui.sqlite3` 保存工作與人工審核，重開網站不會消失。
- 上傳原檔位於 `UI/data/uploads/<job-id>/`；新 pipeline 輸出位於
  `output/ui_runs/<job-id>/`。兩者都不進 Git。
- 啟動與「重新掃描」會從 `UI_DISCOVERY_ROOTS` 匯入既有
  `*_annotated_analysis.json`，相同 JSON 不會重複建立案件。
- AI analysis JSON 保持唯讀；人工判定不會改寫模型輸出。
- 畫面顯示 RT-DETR/STGCN confidence、歸因狀態、可能車輛、OCR 狀態與 OCR
  confidence。Confidence 不是 accuracy；`resolved` 也不是人工 ground truth。
- 垃圾與隨地便溺事件的車牌旁都提供鉛筆按鈕。隨地便溺使用 STGCN confirmed 人物
  回追到的 vehicle/scooter 與其 OCR 結果；找不到可靠車輛時保留 `NULL`。人工修正版獨立寫入 SQLite
  `plate_corrections`，不覆寫 AI OCR 值；單獨修正車牌不代表事件已審核。
- 人工判定只有「AI 辨識正確」與「AI 誤判」。舊資料若仍為 `uncertain`，資料不會被刪除，
  但不再計為已完成審核，必須重新選擇後才能進入已審核頁。
- 桌面版將影片與摘要放左欄，所有事件及逐項審核表單放在可獨立捲動的右欄；窄螢幕
  會自動改為單欄。事件證據與其人工判定合併為同一卡片。頁面不顯示頂部 banner
  或影片下方的檔名。
- 沒有 AI confirmed event 的影片仍建立一個人工審核項目，讓審核者檢查漏判。

## 設定

本機設定放在 `UI/.env`，可提交範本為 `UI/.env.example`。重要欄位：

| 變數 | 預設 | 用途 |
|---|---|---|
| `UI_ALLOWED_INPUT_ROOTS` | repository `resources/` | 可由資料夾輸入功能讀取的根目錄，逗號分隔 |
| `UI_DISCOVERY_ROOTS` | `output` | 掃描既有 analysis JSON 的根目錄，逗號分隔 |
| `UI_OUTPUT_ROOT` | `output/ui_runs` | UI 新工作的 pipeline 輸出根目錄 |
| `UI_DATABASE_PATH` | `UI/data/ui.sqlite3` | 工作與審核 SQLite |
| `UI_UPLOAD_ROOT` | `UI/data/uploads` | 上傳影片保存位置 |
| `UI_EXPORT_ROOT` | `output/ui_exports` | 已審核違規 ZIP 保存位置 |
| `UI_FFMPEG_EXECUTABLE` | `ffmpeg` | 剪輯違規片段的 FFmpeg 執行檔 |
| `UI_EXPORT_PRE_ROLL_SEC` | `3` | 事件起點前保留秒數 |
| `UI_EXPORT_POST_ROLL_SEC` | `3` | 事件終點後保留秒數 |
| `UI_CONDA_ENV` | `rtdetr` | 執行 production pipeline 的 conda env |
| `UI_PIPELINE_BATCH` | `8` | 傳給 `scripts/main.py` 的 `PIPELINE_BATCH` |
| `UI_MAX_UPLOAD_MB` | `2048` | 單次 HTTP request 上限 |
| `UI_POLL_SECONDS` | `3` | React 重新取得工作狀態間隔 |
| `UI_HOST` | `127.0.0.1` | Flask 綁定位址；預設只開放本機 |

瀏覽器不能直接瀏覽 server 的任意磁碟。若要輸入其他資料夾，先把它的絕對根路徑
加入 `UI_ALLOWED_INPUT_ROOTS`，例如：

```dotenv
UI_ALLOWED_INPUT_ROOTS=/home/se_copilot/trashProject/resources,/mnt/video_archive
```

## 安裝與啟動

首次安裝：

```bash
conda run -n rtdetr python -m pip install -r requirements.txt
cd UI/frontend
npm ci
npm run build
cd ../..
```

Frontend 目前的 Vite 需要 Node `^20.19.0 || >=22.12.0`；本專案驗證版本為
Node 22。`npm ci` 依 `package-lock.json` 從乾淨狀態重建依賴與 `.bin` symlink，適合 checkout
與部署驗證；不要直接複製其他 checkout 的 `node_modules`。

匯出片段另需主機可執行 FFmpeg；若執行檔不在 `PATH`，以
`UI_FFMPEG_EXECUTABLE` 指定絕對路徑。

正式本機啟動（Flask 同時提供 build 後的 React）：

```bash
conda run -n rtdetr python -m UI.backend
```

瀏覽 `http://127.0.0.1:5000`。開發 React 時可另開終端：

```bash
cd UI/frontend
npm run dev
```

Vite 會把 `/api` proxy 到 `VITE_API_PROXY_TARGET`。不要同時啟動多個
`UI_WORKER_ENABLED=1` 的 Flask process；目前是單機、單 worker 設計。

## API

| Method | Path | 說明 |
|---|---|---|
| `GET` | `/api/health` | 服務狀態 |
| `GET` | `/api/config` | 前端可見設定與信心分數警語 |
| `GET` | `/api/jobs?review_status=unreviewed` | 工作清單與分頁數量 |
| `GET` | `/api/jobs/<id>` | analysis JSON、影片資訊、審核項目 |
| `POST` | `/api/jobs/upload` | multipart `videos` 上傳一或多支影片 |
| `POST` | `/api/jobs/folder` | 排程 allowlist 資料夾中的影片 |
| `POST` | `/api/jobs/discover` | 重新掃描既有 production JSON |
| `POST` | `/api/exports/reviewed` | 下載已審核 accepted 違規片段與 Excel ZIP |
| `PUT` | `/api/jobs/<id>/reviews/<event-key>` | 保存逐事件人工判定 |
| `PUT` | `/api/jobs/<id>/events/<event-key>/plate` | 保存人工修正版車牌 |
| `DELETE` | `/api/jobs/<id>/events/<event-key>/plate` | 移除人工修正並恢復顯示 AI 值 |
| `POST` | `/api/jobs/<id>/retry` | 重新排程 failed job |
| `GET` | `/api/jobs/<id>/video` | 支援 Range request 的 annotated MP4 |

## 已知限制

- 匯出範圍是「所有審核項目都已完成」的案件，且只包含有實際 event、人工判定為
  accepted 的違規；rejected 與「無 AI event，判定正確」不會列為違規。
- 每個事件以 JSON `start_sec/end_sec`（舊資料退回 `time_sec`）前後加設定秒數後，使用
  FFmpeg 重新編碼成獨立 MP4。舊事件若沒有時間戳或 annotated MP4 已遺失，Excel 仍保留
  該違規並在「匯出狀態」說明未剪輯原因。
- 匯出目前在單一 Flask request 內同步執行；大量長片段會讓下載等待較久。每次 ZIP 保存
  在 `UI_EXPORT_ROOT`，目前沒有自動清理週期。

- 目前沒有帳號、權限與稽核簽章，預設只能綁 `127.0.0.1`；開放 LAN 前必須加上
  reverse proxy、認證與存取控制。
- 內建 worker 只回報 queued/running/completed/failed，沒有逐 frame 百分比。
- 一個 Flask process 只跑一個 pipeline worker；多 GPU／多人 production deployment
  需要把 queue 與 worker 拆成獨立服務。
- UI 可讀既有 production schema `2.0.0` 與含 `litter_detection` 診斷的 `2.1.0`；
  React 目前仍以 confirmed events 為主要畫面，不顯示 research backtrack sidecar，也不把
  candidate 或 route 當 ground truth。
