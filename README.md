# 環保科技執法系統

本專案以 Python 與電腦視覺模型分析固定監視器影片，辨識新拋出的垃圾或持續性隨地便溺行為，反追蹤可能違規者及其車輛，並在證據充分時辨識車牌，供後續人工複核與執法流程使用。

系統設計以降低誤罰為優先。垃圾 detector bbox、事件確認、違規者歸因、車輛關聯與車牌辨識是不同證據層級；任一步驟證據不足都必須保留 `NULL`／人工複核，不強制產生罰單。

## Repository 架構

```text
trashProject/
├── scripts/                     # 唯一 production pipeline
│   ├── main.py                  # 影片推論入口
│   ├── pipeline/                # 主要模組化實作
│   │   ├── action.py            # YOLO-Pose + STGCN
│   │   ├── detect.py            # actor/litter detection orchestration
│   │   ├── litter_tracker.py    # litter confirmation、violator state
│   │   ├── plate.py             # 車牌 detection + PaddleOCR
│   │   ├── events.py            # summary/events schema
│   │   ├── config.py            # 集中執行參數
│   │   ├── infra/               # model、motion、video I/O、worker
│   │   ├── litter/              # litter trajectory helpers
│   │   └── backtrack/           # Kalman/RTS/cost/flow/sidecar
│   ├── frontend/                # summary/events 靜態 dashboard
│   └── *.py                     # compatibility shims 與工具入口
├── tests/
│   ├── pipeline/                # production unit/integration tests
│   └── test_litter_regression.py
├── modules_weight/              # 本機權重，Git ignored
├── resources/                   # 本機測試影片，Git ignored，可不存在
├── dataset-pose/                # STGCN pose annotations/training artifacts
├── mmaction2/                   # STGCN/STGCN++ dependency
├── artifacts/                   # sidecar、annotation、metrics，Git ignored
├── output/                      # annotated videos，Git ignored
├── versions/                    # 每次整合版本說明
├── scripts-old-stable/          # rollback/reference baseline
├── AGENTS.md                    # 唯一 AI Agent Instruction
└── README.md                    # 本文件
```

### Production 與 reference

- `scripts/` 是唯一 production source of truth。
- `scripts/pipeline/` 放真正實作；`scripts/action.py`、`detect.py`、`litterTracker.py` 等頂層檔案是舊 import compatibility shim。
- `scripts-old-stable/` 只用來比較與 rollback，不進行日常開發。
- 不建立 `heetah/`、`pgdr/` 或其他個人程式碼副本，開發隔離完全使用 Git branch/worktree。
- `mmpose-rtmw/` 已移除，production keypoints 只來自 YOLO-Pose。

## 高階系統流程

```text
輸入監視器影片
  -> 模型 preload / warmup
  -> 背景讀取 frame + temporal motion mask
  -> YOLO-Seg vehicle/scooter detection
  -> vehicle gate
      ├── YOLO-Pose person detection/tracking/keypoints
      │     -> STGCN normal/urinate
      │     -> sustained temporal confirmation
      ├── RT-DETR 4-channel litter candidate
      │     -> geometry/motion/core-motion/holding filter
      │     -> GlobalLitterTracker pending/confirmed
      └── confirmed event
            -> Smart Backtrack person/vehicle/NULL
            -> vehicle/scooter ROI
            -> plate detector + PaddleOCR
  -> annotated video
  -> summary.json / events.jsonl / backtrack sidecar
```

## 模組責任

### Vehicle / scooter

YOLO-Seg 負責 vehicle/scooter detection 與 tracking，並提供 vehicle gate、person association、反追蹤及 OCR 的車輛候選。它不負責 STGCN action 或直接確認垃圾事件。

`VEHICLE_GATE` 預設開啟；最近出現車輛的 TTL 預設為 3 秒。沒有近期車輛時，系統可略過昂貴的 pose、STGCN、RT-DETR 與 OCR 路徑。

### Person action

YOLO-Pose 直接負責 person detection、tracking 與 17 點 keypoints，骨架依 `track_id` 累積後交給 STGCN。

STGCN classes 固定為：

```python
ACTION_CLASSES = {0: "normal", 1: "urinate"}
```

STGCN 不判斷 littering。預設 sequence window 為 100 frames；urinate 使用 8 秒視窗與 5 秒 evidence 基準，並受現有 top-p/hysteresis 環境變數控制。只有通過持續性證據才會產生 urinate warning。

### Litter object-event

RT-DETR 使用 BGR 加 temporal change map 的 4-channel 輸入。模型輸出的 `litter` bbox 只是 candidate，必須依序通過：

1. Bbox size/aspect-ratio filter。
2. 全框 motion evidence。
3. 中心區 core-motion evidence。
4. Camera-shake cooldown。
5. Actor polygon/relative-motion holding gate。
6. `GlobalLitterTracker` trajectory、displacement、temporal confirmation。

只有 `state == "confirmed"` 才是垃圾事件。Pending candidate 不會畫成最終違規。

### Smart Backtrack

Smart Backtrack 位於 `scripts/pipeline/backtrack/`：

```text
actor detections
  -> Kalman + Hungarian same-object tracking
  -> confidence-aware filtering
  -> RTS smoothing

litter trajectory
  -> x-linear / y-quadratic reverse hypotheses

release hypotheses + actor tracklets
  -> C_BA(litter, person)
  -> C_AC(person, vehicle)
  -> C_BC(litter, vehicle)
  -> route candidates
  -> Min-Cost Flow
  -> person / vehicle / NULL
```

Hungarian 只維護同一物件跨幀 identity，不做 person↔vehicle 或 litter attribution。Person/vehicle capacity 允許同車多人與同人多事件；每個事件都有 `NULL` route，避免證據不足時強制歸因。

Backtrack sidecar 用於標註、成本校正與 gate 分析。沒有人工 reviewed ground truth 時，只能報告 candidate coverage/resolved/dustbin，不能宣稱歸因準確率。

### Plate OCR

OCR 只處理已可靠歸因的 vehicle/scooter ROI。無法辨識、低信心或遮擋時，保留影像與失敗狀態，不猜測車牌、不自動開罰。

## 執行方式

預設環境：

```bash
conda run -n rtdetr ...
```

目前 `scripts/main.py` 只接受一個 positional video path；batch、threshold 與輸出位置主要使用環境變數。

```bash
OUTPUT_ROOT=output PIPELINE_BATCH=8 \
conda run -n rtdetr python scripts/main.py resources/resize.mp4
```

也可直接提供絕對路徑：

```bash
OUTPUT_ROOT=output \
conda run -n rtdetr python scripts/main.py /path/to/video.mp4
```

請勿沿用舊版本的 `--batch`、`--disable-action`、`--disable-plate`、`--no-engine`、`--trash-conf` 參數；目前 CLI 不接受這些選項。

### 常用環境變數

| 變數 | 預設 | 說明 |
|---|---:|---|
| `PIPELINE_BATCH` | `8` | Pipeline batch size |
| `YOLO_SEG_FRAME_SKIP` | `2` | Vehicle/scooter detector cadence |
| `BBOX_CONF` | `0.45` | Actor confidence |
| `TRASH_CONF` | `0.4` | Litter candidate confidence |
| `ACTION_WINDOW` | `100` | STGCN sequence frames |
| `URINATION_WINDOW_SEC` | `8.0` | Urinate evidence window |
| `URINATION_MIN_SEC` | `5.0` | Binary evidence minimum |
| `VEHICLE_GATE` | `1` | Vehicle gate enable |
| `VEHICLE_GATE_TTL_SEC` | `3.0` | Recent vehicle TTL |
| `RTDETR_ENABLED` | `1` | Litter detector/OCR enable |
| `SMART_BACKTRACK` | `1` | Smart attribution enable |
| `SMART_BACKTRACK_SIDECAR` | `1` | Candidate sidecar enable |
| `OUTPUT_ROOT` | `.` | Output directory；建議明確設為 `output` |

完整預設值以 `scripts/pipeline/config.py` 與各環境變數使用點為準。

## 輸出

假設輸入為 `resize.mp4`，且 `OUTPUT_ROOT=output`：

```text
output/resize_annotated.mp4
output/resize_annotated_summary.json
output/resize_annotated_events.jsonl
output/resize_annotated_backtrack_candidates.jsonl
```

實際 sidecar 是否產生取決於功能開關與事件狀態。

## 測試

### Compile smoke

```bash
conda run -n rtdetr python -m py_compile \
  scripts/main.py \
  scripts/pipeline/action.py \
  scripts/pipeline/detect.py \
  scripts/pipeline/litter_tracker.py
```

### Pipeline tests

```bash
conda run -n rtdetr python -m pytest -q tests/pipeline
```

### Targeted GPU-free tests

```bash
conda run -n rtdetr python -m pytest -q \
  tests/pipeline/test_config.py \
  tests/pipeline/test_import_smoke.py \
  tests/pipeline/test_backtrack_kalman.py \
  tests/pipeline/test_backtrack_costs.py \
  tests/pipeline/test_backtrack_flow.py
```

大型模型與影片測試必須明確列出使用的 weights、clip、環境變數與輸出結果。MP4 可解碼只代表輸出容器正常，不代表事件或歸因正確。

## Git 開發流程

### Branch ownership

```text
main          正式穩定版本
dev/heetah    張宇誠個人整合 branch
dev/pgdr      張哲誠個人整合 branch
```

禁止建立個人 production 資料夾。需要平行開發時使用 branch 或 Git worktree。

### 每次功能開發

1. 從最新穩定基準建立或同步自己的 branch。
2. 先確認目標行為與測試案例。
3. 完成一個可獨立驗收的功能或修正。
4. 執行 unit test；依風險執行 integration/regression。
5. 更新本 `README.md` 中受影響的架構、命令或參數。
6. 新增 `versions/YYYY-MM-DD_<author>_<topic>.md`。
7. 建立 atomic Conventional Commit。
8. Push、建立 PR、review、通過測試後合併。

Commit 範例：

```text
feat(backtrack): add NULL route for uncertain attribution
fix(litter): reject stationary vehicle components
refactor(pipeline): extract video I/O workers
test(action): add sustained urination regression
docs(architecture): document vehicle gate behavior
chore(repo): rename production pipeline directory
```

## Version note 格式

檔名：

```text
versions/2026-08-02_heetah_backtrack.md
```

內容至少包含：

```markdown
# 變更標題

- 日期：
- 作者：
- Branch：
- Commit：
- 類型：feat / fix / refactor / test / docs / chore

## 問題背景
## 實作內容
## API／Config／Schema 變更
## 測試證據
## 已知限制
## 回滾方式
```

## AI Agent 規則

所有 AI Coding Agent 必須完整閱讀 `AGENTS.md`，再讀 live code。`AGENTS.md` 是唯一 AI 規則內容來源；本專案不維護第二份重複的 `CLAUDE.md`。
