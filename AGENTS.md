# 環保科技執法系統 AI Agent Instruction

- Version: `v0.2`
- Last verified: `2026-08-01`
- Repository: `/home/se_copilot/trashProject`
- Production pipeline: `scripts/`
- Default runtime: `conda run -n rtdetr ...`

本文件是本 repository 唯一的 AI Coding Agent 規則來源，適用於 Codex、Claude 或其他會讀取專案指令的 Coding Agent。AI 的任務是協助開發固定監視器影片中的亂丟垃圾、隨地便溺、違規者反追蹤、車輛關聯與車牌辨識系統。

本文件不是監視器運行時的事件判斷 prompt，也不能取代人類對罰單與證據的最終審核。

---

## 1. Instruction Authority

發生衝突時依以下順序處理：

1. `AGENTS.md`：產品安全、證據層級、模組責任與開發規則。
2. Live code：當前函式、CLI、環境變數、schema 與實際預設值。
3. `README.md`、subsystem README、`versions/`：人類說明與歷史紀錄。

若 live code 違反本文件的安全或責任合約，不可自行把錯誤行為寫回文件。先指出衝突，再做最小修正。若只是介面或參數已更新，則以 live code 為準並同步更新文件。

回答架構、執行流程或效能問題前，必須先讀 live code；本 repository 漂移頻繁，禁止只憑記憶回答。

---

## 2. Repository Directory Contract

```text
trashProject/
├── scripts/                  # 唯一 production pipeline
│   ├── main.py               # 影片推論入口
│   ├── pipeline/             # 主要 Python 實作
│   │   ├── action.py
│   │   ├── detect.py
│   │   ├── litter_tracker.py
│   │   ├── plate.py
│   │   ├── events.py
│   │   ├── config.py
│   │   ├── infra/
│   │   ├── litter/
│   │   └── backtrack/
│   ├── frontend/             # summary/events 靜態檢視介面
│   └── *.py                  # 舊 import compatibility shims 或工具入口
├── tests/
│   ├── pipeline/             # production pipeline unit/integration tests
│   └── test_litter_regression.py
├── modules_weight/           # 本機模型權重；不進一般 Git
├── resources/                # 本機影片；不進一般 Git，可不存在
├── dataset-pose/             # STGCN pose annotation/training artifacts
├── mmaction2/                # vendored STGCN/STGCN++ dependency
├── artifacts/                # backtrack sidecar、annotation、metrics；預設不進 Git
├── output/                   # annotated video、summary、events；不進 Git
├── versions/                 # 每次整合版本的 Markdown 紀錄
├── scripts-old-stable/       # rollback/reference baseline，非 production
├── README.md                 # 給人類開發者的架構與流程
└── AGENTS.md                 # 唯一 AI Agent Instruction
```

### Folder rules

| Path | Agent 可以做 | Agent 不可以做 |
|---|---|---|
| `scripts/` | 實作 production 功能、修 bug、更新相容 shim | 建立個人版本副本、混入測試影片或權重 |
| `scripts/pipeline/` | 新增可重用模組、維持清楚 data flow | 把核心功能寫回頂層 shim、跨分支偷改責任 |
| `tests/` | 新增 unit/integration/regression test | 把測試重新塞回 `scripts/` |
| `modules_weight/` | 讀取與驗證本機模型 metadata | 把大型 `.pt/.pth/.onnx/.engine` 加進一般 Git |
| `resources/` | 執行明確指定的測試影片 | 把大量影片加入一般 Git |
| `dataset-pose/` | 維護 YOLO-Pose 產生的 STGCN annotation | 重新引入 RTMW keypoints |
| `mmaction2/` | 必要時做最小相容修正 | 大範圍格式化或無關 vendor 改寫 |
| `artifacts/` | 儲存研究 sidecar、metrics、預覽 | 把 coverage 當作 accuracy |
| `output/` | 儲存執行輸出並驗證可解碼性 | 把可解碼影片宣稱為模型正確性證據 |
| `versions/` | 新增版本說明 | 改寫既有版本歷史以掩蓋變更 |
| `scripts-old-stable/` | 唯讀比較與 rollback 參考 | 未經明確要求直接修改 |

不得建立 `heetah/`、`pgdr/` 或其他個人 production 副本。開發隔離完全使用 Git branch/worktree。

`mmpose-rtmw/` 已從 repository 移除。不得重新建立、載入或作為 production fallback。

---

## 3. Product Goal and Evidence Levels

系統目標是從固定監視器畫面建立可稽核的違規證據鏈：

```text
detection candidate
  -> confirmed event
  -> attributed person/vehicle
  -> readable plate
  -> finable case
```

以上每一層都必須分開。下游失敗不能回頭偽造上游證據。

- Detector bbox 只是 candidate，不是 confirmed event。
- Confirmed litter 代表垃圾事件成立，不代表已找到正確違規者。
- 找到附近人車不代表歸因成立。
- 車牌 OCR 失敗不得猜測或補造文字。
- 人、車或車牌證據不足時保留事件並輸出 `NULL`／人工複核，不得強制配對。
- 自動開罰必須同時具有可靠事件、歸因與車牌證據；AI 輸出仍需人類依法規與證據程序複核。

降低誤罰優先於強制產生結果。

---

## 4. Current Production Flow

```text
Input video
  -> model preload/warmup
  -> background frame reader + motion mask
  -> batched main inference
  -> YOLO-Seg vehicle/scooter detection
  -> vehicle gate
      -> YOLO-Pose person detection/tracking/keypoints
      -> STGCN normal/urinate classification
      -> RT-DETR 4-channel litter candidates
      -> motion/shape/core-motion/holding filters
      -> GlobalLitterTracker pending/confirmed
      -> Smart Backtrack person/vehicle/NULL route
      -> plate detection + PaddleOCR
  -> confirmed-event rendering
  -> annotated video + summary/events/sidecar
```

同幀主要偵測流程有明確順序。背景 reader、writer 或 worker overlap 不等於 GPU kernels 已並行；除非有 profiler/CUDA trace，不得宣稱模型同時執行。

---

## 5. Model and Module Responsibility

### Vehicle / scooter branch

- YOLO-Seg 負責 `vehicle`、`scooter` detection/tracking。
- Vehicle/scooter 是 person association、litter attribution 與 OCR 的基礎。
- `VEHICLE_GATE` 預設開啟；`VEHICLE_GATE_TTL_SEC` 預設 `3.0` 秒。
- Gate 關閉時可略過 pose、STGCN、litter 與 OCR 等昂貴路徑。
- 不負責 STGCN action classification 或直接確認 litter。

### Person / action branch

- YOLO-Pose 是 production person detection、tracking、keypoints 的唯一來源。
- Keypoints 依 YOLO-Pose `track_id` 直接對齊，不再做 pose-vs-seg IoU 配對。
- STGCN 只接受 YOLO-Pose keypoint sequence。
- `ACTION_CLASSES` 必須保持：

```python
ACTION_CLASSES = {0: "normal", 1: "urinate"}
```

- 禁止新增 `littering`、`throwing` 或 `litter` STGCN class。
- Sequence 預設為 `100` frames。
- `PipelineConfig` 目前預設 urinate window `8.0` 秒、minimum evidence `5.0` 秒。
- Top-p evidence、high/low confidence 可由現有環境變數調整，但不可靜默修改預設。
- 單次 urinate score 不可直接產生 violation，必須通過 temporal confirmation。

### Litter object-event branch

- RT-DETR 4-channel 只產生 class `litter` candidate。
- 第四通道來自 temporal pixel-change map；修改時必須驗證最終模型輸入，不只檢查來源影像。
- Candidate 必須經過 geometry、motion、core-motion、camera-shake、holding 與 tracker evidence。
- `pending` 不得畫成 confirmed violation。
- 只有 `GlobalLitterTracker` 確認後才能建立 littering event。
- 噪聲修正應強化 motion、shape、component 或 physical evidence，不可只降低 threshold。

### Smart Backtrack

- Smart Backtrack 位於 `scripts/pipeline/backtrack/`，預設由 `SMART_BACKTRACK=1` 啟用。
- Hungarian 只處理同類、同一物件的跨幀 identity，不可拿來做 person↔vehicle 或 litter attribution。
- 流程為 confidence-aware Kalman、RTS smoothing、反向 litter trajectory、`C_BA/C_AC/C_BC`、route scoring、Min-Cost Flow。
- Person/vehicle capacity 必須允許同車多人與同人多事件。
- 每個事件必須保留完整 `NULL` route，禁止 forced match。
- Smart attribution 只能處理已 confirmed litter；不能救回未通過 litter confirmation 的 candidate。
- Sidecar 是研究與標註資料，不是 ground truth。未經人工 reviewed annotation，不得宣稱 attribution accuracy。

### OCR branch

- 只處理已可靠歸因的 vehicle/scooter ROI。
- 使用原始未畫框 frame crop，避免 annotation 污染 OCR。
- OCR 低信心、遮擋或無結果時保存證據與失敗狀態，不得生成虛構車牌。
- OCR 不負責 event confirmation、pose 或 action classification。

---

## 6. Runtime Contract

目前 `scripts/main.py` 只接受一個 positional video path。執行參數主要由 `PipelineConfig` 與環境變數控制。

```bash
OUTPUT_ROOT=output PIPELINE_BATCH=8 \
conda run -n rtdetr python scripts/main.py resources/resize.mp4
```

不得沿用已失效的舊 CLI，例如：

```text
--batch
--disable-action
--disable-plate
--no-engine
--trash-conf
```

在提供命令前，先讀 `scripts/main.py`、`scripts/pipeline/config.py` 與使用點的環境變數。

---

## 7. Coding Agent Workflow

任何修改前必須：

1. 完整讀取根目錄 `AGENTS.md`。
2. 讀取 `README.md` 與目標 subsystem README。
3. 執行 `git status --short`，保留使用者與其他開發者的未提交變更。
4. 以 `rg` 搜尋 live entrypoint、callers、tests、config 與 schema。
5. 確認任務屬於哪個 module responsibility，禁止跨分支偷接捷徑。
6. 先建立或選定可驗收行為，再進行最小修改。

修改時：

- 使用清楚、模組化、可測試的 Python。
- 新 production 實作放在 `scripts/pipeline/`；頂層同名檔只保留 compatibility shim。
- 不做無關格式化、批次改名或 opportunistic cleanup。
- 不覆蓋不相關的 working-tree changes。
- 不使用 `git reset --hard`、`git checkout --` 或其他破壞性復原。
- 不在未獲批准時下載 dependency、模型或影片。
- 模型 device/CUDA 問題先檢查 torch、device count、`/dev/nvidia*` 與 `nvidia-smi`，不要直接改成 CPU 後宣稱已修復 GPU。

完成時：

- 執行與風險相稱的測試。
- 更新 `README.md` 中受影響的架構或操作內容。
- 新增 `versions/YYYY-MM-DD_<author>_<topic>.md`。
- 回報修改檔案、測試證據、未驗證項目與已知限制。

---

## 8. Git and Team Development Rules

### Branches

- `main`：正式穩定版本，只接受通過 review/test 的整合。
- `dev/heetah`：張宇誠個人整合 branch。
- `dev/pgdr`：張哲誠個人整合 branch。
- 較大功能可由個人 branch 再建立 `feat/...`、`fix/...` 短期 branch。
- 禁止以本機資料夾複製個人版本。

### Commit boundary

Commit 以可獨立驗收的行為為單位，不以單一函式行數為單位。程式、必要測試與相依文件應形成 atomic change。

採 Conventional Commits：

```text
feat(backtrack): add NULL route for uncertain attribution
fix(litter): reject stationary vehicle components
refactor(pipeline): extract video I/O workers
test(action): add sustained urination regression
docs(architecture): document vehicle gate behavior
perf(detection): reduce repeated model inference
chore(repo): reorganize project layout
```

完成流程：

```text
sync branch
  -> implement one behavior
  -> unit test
  -> integration/regression where relevant
  -> update README.md
  -> add versions note
  -> commit
  -> push
  -> PR/review
  -> merge to main
  -> release tag when appropriate
```

---

## 9. Validation Rules

### Fast code validation

```bash
conda run -n rtdetr python -m py_compile \
  scripts/main.py \
  scripts/pipeline/action.py \
  scripts/pipeline/detect.py \
  scripts/pipeline/litter_tracker.py
```

```bash
conda run -n rtdetr python -m pytest -q tests/pipeline
```

若完整 suite 因缺少外部影片、模型或工具而無法執行，必須跑可執行的 targeted tests，並清楚列出未執行項目，不得寫成全部通過。

### Runtime validation

- Litter regression 使用明確指定的 litter clips。
- STGCN regression 使用 normal/urinate clips，不得用 litter clip 評估 STGCN littering。
- 修改輸出影片時，用 `ffprobe` 或 OpenCV 驗證 MP4 可解碼。
- 可解碼只代表容器 smoke pass，不代表事件或歸因正確。
- GPU concurrency、效能改善、模型 accuracy 必須提供 profiler、trace、reviewed labels 或正式 metrics。

### Required behavior scenarios

1. 新 litter candidate 證據不足：保持 pending 或丟棄，不 confirmed、不開罰。
2. Confirmed litter 找不到可靠人車：保存事件，選擇 `NULL`／人工複核。
3. Person sustained urinate：只由 YOLO-Pose + STGCN temporal evidence 確認。
4. Person throws litter：STGCN 可以是 normal；littering 只由 object-event branch 確認。
5. 找到違規車但 OCR 失敗：保存 ROI 與失敗狀態，不產生車牌、不自動開罰。

---

## 10. Documentation Rules

`README.md` 給人類開發者，必須保持：

- 目錄與 module responsibility 正確。
- 執行命令與 live CLI/env 一致。
- 輸入、輸出、模型與測試方式可重現。
- 不保留已失效的架構或參數範例。

每次完成整合功能新增：

```text
versions/YYYY-MM-DD_<author>_<topic>.md
```

版本文件至少記錄日期、作者、branch、commit、問題、變更、介面/config、測試證據、限制與回滾方式。
