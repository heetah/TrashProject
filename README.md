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
│   │   ├── events.py            # 精簡 analysis JSON schema
│   │   ├── config.py            # 集中執行參數
│   │   ├── infra/               # model、motion、video I/O、worker
│   │   ├── litter/              # litter trajectory helpers
│   │   └── backtrack/           # Kalman/RTS/cost/flow/sidecar
│   ├── frontend/                # analysis JSON 靜態 dashboard
│   └── *.py                     # compatibility shims 與工具入口
├── UI/                          # 獨立 Flask API + React 人工複核介面
│   ├── backend/                 # SQLite job/review、單一 pipeline worker、JSON/media API
│   └── frontend/                # 未審核/已審核、可捲動影片列與事件證據
├── tests/
│   ├── pipeline/                # production unit/integration tests
│   ├── ui/                      # UI persistence/API tests
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
  -> 有界背景 reader queue
      -> 讀取 frame + temporal motion mask
      -> 預先建立 RT-DETR 4-channel input
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
  -> annotated video + analysis.json
  -> optional research backtrack sidecar
```

背景 reader 以 `PreparedFrame(index, source_bgr, foreground_mask,
litter_model_input)` 傳遞每幀資料。`source_bgr` 維持未標註，供 motion、OCR 與證據影像使用；
第四通道前處理在主執行緒處理前一批 GPU inference 時準備下一批，RT-DETR batch 不再於
呼叫 `predict()` 前等待 RGB 轉換、相鄰幀差分與 4-channel 拼接。Queue 為有界且逐幀保留
`index`，主流程會驗證 batch index 連續才推論，避免重排或跨影片狀態污染。

這是 CPU preprocessing／GPU inference 的跨批 overlap，不代表 YOLO-Seg、YOLO-Pose、
RT-DETR 或 STGCN 的 CUDA kernels 同時執行。主模型同幀順序及所有事件確認、歸因、OCR
證據責任都保持不變。

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

RT-DETR 4-channel inference 對齊
`/mnt/8tb_hdd/under115a/4c-yolo/run_4ch.py`：OpenCV BGR frame 先轉成 RGB，
第四通道使用相鄰幀 grayscale absolute difference，每幀 min-max normalization 到
`0..255` 後乘 `1.5` 並 clip 為 `uint8`。Ultralytics 對 4-channel NumPy input
不會自動交換 BGR/RGB；送入 predictor 的 channel order 因此必須明確為
`[R, G, B, change]`，再由 predictor 除以 `255` 形成最終 model tensor。單幀、
batch、batch repair 與 TensorRT smoke test 共用同一個 input builder。

模型輸出的 `litter` bbox 只是 candidate，必須依序通過：

1. Bbox size/aspect-ratio filter。
2. 全框 motion evidence。
3. 中心區 core-motion evidence。
4. Camera-shake cooldown。
5. Actor polygon/relative-motion holding gate。
6. `GlobalLitterTracker` trajectory、displacement、temporal confirmation。

近車候選仍會先排除 vehicle-contained、共動與純水平條紋；唯一例外是剛由車框內明顯脫離的前三個 observation。當該軌跡起點確實在同一車框內、終點已脫離，會保留該車作為 thrower fallback；這不是最近車輛配對，仍須通過 vehicle-relative、物理與 temporal confirmation gate。

只有 `state == "confirmed"` 才是垃圾事件。Pending candidate 不會畫成最終違規。

### Smart Backtrack

Smart Backtrack 位於 `scripts/pipeline/backtrack/`：

```text
actor detections
  -> Kalman + Hungarian same-object tracking
  -> confidence-aware filtering
  -> RTS smoothing

litter trajectory
  -> B0/B1 observation-gap release window
  -> B0/B1/B2 early direction
  -> constant-velocity / x-linear-y-quadratic reverse hypotheses

release hypotheses + actor tracklets
  -> C_BA(litter, person)
  -> C_AC(person, vehicle)
  -> C_BC(litter, vehicle)
  -> route candidates
  -> Min-Cost Flow
  -> person / vehicle / NULL
```

Hungarian 只維護同一物件跨幀 identity，不做 person↔vehicle 或 litter attribution。Person/vehicle capacity 允許同車多人與同人多事件；每個事件都有 `NULL` route，避免證據不足時強制歸因。

Production 的 direct litter→vehicle hard gate 使用未擴張的 vehicle bbox。令 release
point 為 `p`、vehicle bbox 為 `B`、bbox 寬高為 `w,h`，距離定義為
`D = dist(p, B) / sqrt(w²+h²)`，要求 `D <= 0.30`。Actor evidence 與 release
的時間差同時要求 `Δframe <= 3` 及 `Δframe/FPS <= 0.25 s`；兩者是 AND，
不是相加成 0.55 秒。固定秒數保留跨 FPS 的物理意義，3-frame cap 則限制逐幀
detector 可漏失的 observation 數。

Backtrack sidecar 用於標註、成本校正與 gate 分析。沒有人工 reviewed ground truth 時，只能報告 candidate coverage/resolved/dustbin，不能宣稱歸因準確率。

### Plate OCR

OCR 只處理已可靠歸因的 vehicle/scooter ROI。無法辨識、低信心或遮擋時，保留影像與失敗狀態，不猜測車牌、不自動開罰。

## 執行方式

預設環境：

```bash
conda run -n rtdetr ...
```

### Python 依賴

Production pipeline、vendored MMAction2/STGCN、Flask UI backend 與 repository
測試所需的 Python 套件統一列在 root [`requirements.txt`](requirements.txt)。支援環境為
Linux x86_64 + Python 3.11；首次建立環境與安裝：

```bash
conda create --override-channels -c conda-forge -n rtdetr python=3.11 pip -y
conda install --override-channels -c conda-forge -n rtdetr "nodejs>=22.12,<23" -y
conda run -n rtdetr python -m pip install --upgrade pip
conda run -n rtdetr python -m pip install -r requirements.txt
conda run -n rtdetr npm --prefix UI/frontend ci
```

GPU 加速與可重現模型匯出所需的 `tensorrt`、`nvidia-modelopt[onnx]`、`onnx`、
`onnxruntime-gpu` 已納入
requirements。Live code 會優先使用與 batch 相符的 TensorRT `.engine`，engine 缺失或
不相容時回退 `.pt` 權重。`transformers`、legacy `paddleocr` 與 generic MMAction `decord`
仍非 production 必要依賴：STGCN 直接接收 skeleton dict，不走 Decord 影片 loader；production
車牌文字辨識使用 PaddleX，不使用 legacy PaddleOCR API。

React/Vite 套件不放入 Python requirements，由 `UI/frontend/package-lock.json` 鎖定；
使用 Node 22 與 `npm ci` 重建，不沿用從其他路徑複製的 `node_modules`。

目前 `scripts/main.py` 只接受一個 positional video path；模型路徑、batch、threshold、
device、motion 與輸出位置集中在 repository root 的 `.env`。首次 checkout 可由範本建立：

```bash
cp .env.example .env
```

載入優先序為「shell／UI worker 已 export 的值 > `.env` > 程式內安全預設」，因此單次命令
仍可覆寫設定，UI job 的獨立輸出路徑也不會被 `.env` 蓋掉。若要切換多組設定，可先 export
`PIPELINE_ENV_FILE=/absolute/path/to/profile.env`；設定檔內的相對模型路徑一律以 repository root
解析。`.env` 是本機檔案且不進 Git，完整可提交範本為 [`.env.example`](.env.example)。

```bash
OUTPUT_ROOT=output PIPELINE_BATCH=8 \
conda run -n rtdetr python scripts/main.py resources/resize.mp4
```

也可直接提供絕對路徑：

```bash
OUTPUT_ROOT=output \
conda run -n rtdetr python scripts/main.py /path/to/video.mp4
```

若要逐支處理 `/mnt/8tb_hdd/under115a/litter_vidshort/litter_order` 內的影片，使用：

```bash
bash scripts/run_litter_order.sh
```

腳本會遞迴尋找 `.mp4`、`.avi`、`.mov`、`.mkv`、`.wmv`、`.m4v`，依檔名字典序逐支
呼叫 `scripts/main.py`，預設輸出到 `output/litter_order/`。可用環境變數覆寫輸入資料夾、
輸出位置或 conda 環境：

```bash
INPUT_DIR=/path/to/videos OUTPUT_ROOT=output/my_run CONDA_ENV=rtdetr \
bash scripts/run_litter_order.sh
```

每支影片仍會產生自己的 `_annotated.mp4` 與 `_annotated_analysis.json`；任一影片失敗時，
腳本會繼續處理其他影片，最後以非零狀態結束並列出失敗數量。

請勿沿用舊版本的 `--batch`、`--disable-action`、`--disable-plate`、`--no-engine`、`--trash-conf` 參數；目前 CLI 不接受這些選項。

### Flask + React 人工複核介面

`UI/` 是獨立 web application，不把 Flask/React 混入 `scripts/`。UI server 設定仍使用
`UI/.env`；production inference 使用 root `.env`。Flask 把上傳影片或 UI allowlist 內的
資料夾影片排入 SQLite queue，再由單一 background worker 逐支
呼叫上述 `scripts/main.py`。React 讀取 Flask 提供的 analysis JSON，顯示 annotated
影片與摘要，並在桌面版右欄列出全部 confirmed litter/STGCN 事件、模型 confidence、
Smart Backtrack 狀態、可能車輛與 OCR 證據；每個事件的證據與人工判定共用一卡片。
Confirmed 隨地便溺也會顯示 STGCN 動作回追所得的關聯車輛、OCR 狀態與車牌；車牌可
保存獨立人工修正版，不覆寫 AI OCR 值。介面分為未審核／已審核兩頁。

人工 accepted/rejected 與備註只寫入 `UI/data/ui.sqlite3`，不修改 production analysis
JSON。已審核頁可將所有完整審核案件中 accepted 的實際事件匯出成 ZIP；每個事件有獨立
MP4 片段，並附一份列出違規、關聯車輛、車牌與審核資料的 Excel。重新啟動後工作與審核
仍保留，也可重新掃描 `output/` 下既有的 `*_annotated_analysis.json`。完整設定、安裝、
啟動與 API 見 [`UI/README.md`](UI/README.md)。

### 常用環境變數

| 變數 | 預設 | 說明 |
|---|---:|---|
| `MODEL_BBOX_PATH` / `MODEL_BBOX_PATH_BATCH` | `modules_weight/...` | YOLO-Seg 單幀／batch 權重；相對 repository root |
| `MODEL_TRASH_PATH` / `MODEL_TRASH_PATH_BATCH` | `modules_weight/best-rtdetr-4c-background.pt` | RT-DETR 單幀／batch 權重 |
| `POSE_MODEL_PATH` | `modules_weight/yolo26x-pose.pt` | YOLO-Pose 權重 |
| `STGCN_WEIGHT_PATH` / `STGCN_CONFIG_PATH` | `modules_weight/...` / `mmaction2/...` | STGCN checkpoint 與 config |
| `PLATE_MODEL_PATH` | `modules_weight/best-licnese-plate.pt` | 車牌 detector 權重 |
| `PREFER_TENSORRT` | `1` | 優先嘗試同模型的 matching `.engine`，失敗仍依既有候選回退 |
| `PIPELINE_BATCH` | `8` | Pipeline batch size |
| `PIPELINE_QUEUE_SIZE` | `8` | Prepared frame 有界 queue；預設保留一個 batch，避免高解析影片無界佔用 RAM |
| `PIPELINE_PREPARE_4C` | `1` | 在背景 reader 預先建立 RT-DETR 4-channel input；設 `0` 回到主推論執行緒即時建立 |
| `YOLO_SEG_FRAME_SKIP` | `2` | Vehicle/scooter detector cadence |
| `BBOX_CONF` | `0.45` | Actor confidence |
| `TRASH_CONF` | `0.4` | Litter candidate confidence |
| `ACTION_POSE_CONF` | `0.3` | YOLO-Pose person confidence |
| `ACTION_WINDOW` | `100` | STGCN sequence frames |
| `URINATION_WINDOW_SEC` | `8.0` | Urinate evidence window |
| `URINATION_MIN_SEC` | `5.0` | Binary evidence minimum |
| `VEHICLE_GATE` | `1` | Vehicle gate enable |
| `VEHICLE_GATE_TTL_SEC` | `3.0` | Recent vehicle TTL |
| `RTDETR_ENABLED` | `1` | Litter detector/OCR enable |
| `PLATE_DETECT_CONF` | `0.6` | 送入 plate detector 的 confidence |
| `PLATE_ACCEPT_DETECT_CONF` | `0.8` | 接受車牌 bbox 的最低 confidence |
| `PLATE_OCR_CONFIDENCE` | `0.85` | 接受 OCR 文字的最低 confidence；不足時維持失敗狀態，不猜牌 |
| `MOTION_DIFF_THRESHOLD` | `10` | Temporal motion mask 灰階差 threshold |
| `MOTION_MIN_COMPONENT_AREA` | `4` | Litter motion evidence 的最小 component 面積 |
| `SMART_BACKTRACK` | `1` | Smart attribution enable |
| `SMART_BACKTRACK_SIDECAR` | `0` | Research candidate sidecar；需明確設 `1` 啟用 |
| `SMART_BACKTRACK_STUDY_STAGE` | `full` | Research ablation stage；production 預設不變 |
| `SMART_BACKTRACK_RAW_PREFIX` | `1` | confirmed event 才能使用 pre-postprocessing RT-DETR bbox 補 release trajectory；不參與 event confirmation |
| `SMART_BACKTRACK_DT_DISTANCE_WEIGHT` | `1.0` | D+T stage 的 gate-normalized distance weight |
| `SMART_BACKTRACK_DT_TIME_WEIGHT` | `1.0` | D+T stage 的 gate-normalized time weight |
| `SMART_BACKTRACK_TWO_POINT_MAX_BACK_SEC` | `0.4` | 舊 replay 相容欄位；新版不再作為兩點軌跡的物理截止 |
| `SMART_BACKTRACK_TWO_POINT_PRIOR_COST` | `1.0` | 兩點常速 release hypothesis 基礎 prior cost |
| `SMART_BACKTRACK_MAX_FORWARD_RELEASE_SEC` | `0.5` | ballistic release window 可晚於 detector birth 的上限；仍受已觀測 airborne 軌跡限制 |
| `SMART_BACKTRACK_RELEASE_WINDOW_WEIGHT` | `0.35` | 超出 B0/B1 零成本窗後，每一個 observation-gap 的軟性 prior 增量 |
| `SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT` | `0.0` | `full/reverse` 中 release 點位於 vehicle bbox 深處的軟成本；`0` 關閉，須經 reviewed replay 後才啟用 |
| `LITTER_DEBUG` | `0` | 設為 `1` 時輸出逐幀診斷；annotated video 顯示 actor track ID，並以洋紅框顯示 RT-DETR 通過 class/confidence、但尚未經 geometry/motion/holding/tracker 後處理的 litter bbox 與 confidence |
| `LITTER_FP_CONTAINMENT_THR` | `0.999` | RT-DETR 候選與車輛 bbox 幾乎完全重疊時才在前處理淘汰；設 `0.85` 可重現舊版 baseline，後續仍須通過 motion/holding/tracker confirmation |
| `LITTER_CANDIDATE_SIDECAR` | `0` | 研究用逐 candidate JSONL，記錄 gate reason、tracker ID 與可重現設定；不供前端或 ground truth 使用 |
| `LITTER_CANDIDATE_DEDUP` | `0` | 實驗性同幀 IoU 去重；目前 replay 未採用（未增加正確 confirmed 且 safety proxy 惡化） |
| `LITTER_CANDIDATE_DEDUP_IOU` | `0.5` | 同幀 candidate 去重 IoU 門檻；僅在 `LITTER_CANDIDATE_DEDUP=1` 時生效 |
| `LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR` | `1` | Confirm 是否要求垃圾出生幀已有 thrower；研究 replay 可設 `0`，但仍必須通過運動、軌跡與 actor 關聯 gates |
| `LITTER_MIN_CONFIRM_AGE_VEHICLE` | `3` | vehicle/scooter thrower 的最少 observation 次數；研究 replay 的 2 需以 reviewed clip-level 結果解讀 |
| `LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE` | `12` | vehicle/scooter thrower 的向下位移門檻（px）；僅供可重現 A/B replay |
| `LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT` | `5` | confirm 所需水平位移門檻（px）；降低會放行近垂直落下案例，必須同步檢查 FP proxy |
| `LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE` | `3.5` | vehicle thrower 水平/向下位移最大比例；過大會放行純水平滑動 |
| `LITTER_MIN_VEHICLE_RELATIVE_SEPARATION` | `60` | 垃圾相對載體車輛的最小分離（px）；0 會關閉此 FP 抑制 gate |
| `LITTER_FP_STREAK_RATIO` | `5` | 前處理水平 streak 與向下位移比例門檻 |
| `LITTER_ALLOW_SHAKE_CANDIDATES` | `0` | 是否在 camera-shake frame 繼續提交 candidate；僅用於研究 replay |
| `OUTPUT_ROOT` | `.` | Output directory；建議明確設為 `output` |

其餘 action smoothing、video I/O、writer、Smart Backtrack 與 legacy research 開關都已列在
`.env.example`。演算法內部固定 class mapping、4-channel preprocessing contract 與低頻物理 gate
仍保留在 live code，避免一般部署調參意外改變證據責任。型別化 production 預設以
`scripts/pipeline/config.py` 為準。

### 反追蹤研究 replay

`scripts/backtrack_study.py` 只重播 confirmed event sidecar 內的 frozen
resolver input；不重跑 detector，也不改變 litter confirmation。先由新 sidecar
建立不可變 group split，再只在 development/validation 調參，最後才讀 test：

```bash
conda run -n rtdetr python scripts/backtrack_study.py manifest \
  --candidates artifacts/backtrack_candidates --output artifacts/study_manifest.json

conda run -n rtdetr python scripts/backtrack_study.py replay \
  --candidates artifacts/backtrack_candidates --manifest artifacts/study_manifest.json \
  --config artifacts/distance_time.json --split validation \
  --output artifacts/distance_time_validation.jsonl
```

固定案例 manifest 可用 shard runner 重跑；每個 worker 寫獨立 TSV，且明確
啟用 compact research sidecar：

```bash
CUDA_VISIBLE_DEVICES=0 WORKER_INDEX=0 WORKER_COUNT=2 \
  MANIFEST=artifacts/backtrack_study/multi_actor_v1.json \
  OUTPUT_ROOT=output/backtrack_multi_actor_v1 \
  scripts/run_backtrack_fixed_cases.sh
```

若原始影片不在 manifest 的舊 `source_directory`，可用
`SOURCE_DIRECTORY=/home/under115a/under115a/under115a/litter_vidshort/litter` 覆寫；
`CASE_IDS=13,17,167` 可只重播指定案例，`SKIP_EXISTING_SUCCESS=1` 會保留已完成的
analysis 與 candidate sidecar。RT-DETR gate 校正使用：

```bash
LITTER_CANDIDATE_SIDECAR=1 \
  LITTER_FP_CONTAINMENT_THR=0.999 \
  SOURCE_DIRECTORY=/home/under115a/under115a/under115a/litter_vidshort/litter \
  OUTPUT_ROOT=output/rtdetr_postprocess_containment0999_20260826 \
  scripts/run_backtrack_fixed_cases.sh

conda run -n rtdetr python scripts/calibrate_litter_postprocess.py \
  --sidecar-dir output/rtdetr_postprocess_containment0999_20260826 \
  --baseline-sidecar-dir output/rtdetr_postprocess_baseline_20260826 \
  --output artifacts/litter_postprocess_calibration/containment0999
```

校正工具以 58 個 usable、各含一個人工事件的 clip 作 event-level 配對，另以 63
個 clip 報告 confirmed coverage。candidate、686 列或跨幀 observation 都不是獨立樣本；
目前結果僅顯示 0.999 replay 為 13/58 vs baseline 12/58 正確 tracker ID、20/63 vs
19/63 confirmed clip，paired gain=1/loss=0，Wilson 95% CI 分別為 13.59%--34.66%
與 12.25%--32.77%。未驗證 confirmed track 14→14，但沒有 negative clip，故這只是
safety proxy，不能宣稱 false-positive rate 或普適最佳門檻；event annotations 的
`review_state` 目前仍應由人工確認（工具會在報告中標出 58 筆未 review）。

`exit_code=0` 只代表 pipeline 完成。無人工 reviewed route 的案例只能比較
candidate coverage、route 變化與 margin，不可宣稱 attribution accuracy。

2026-08-26 的研究 recovery replay（完整 63 部）使用 sidecar 記錄的暫時性放寬門檻，
得到 41/63 confirmed clips（usable 41/58，Wilson 95% CI 52.75%--75.67%），達到
「超過 40 部」的短期 coverage 目標；但未驗證 confirmed-track proxy 同時由 14 增至
33，且 58 筆 event annotation 仍未人工 reviewed，因此不得把此結果解讀為 accuracy
或可直接部署的參數。完整命令、paired CI 與回滾方式記於
`versions/2026-08-26_codex_confirmation_recovery_replay.md`；機器可讀結果位於
`artifacts/litter_postprocess_calibration/recovery_horiz1_full_20260826/`。

將人工 actor ID 表與 recovery sidecar 對齊，可使用：

```bash
conda run -n rtdetr python scripts/summarize_actor_ground_truth_metrics.py \
  --sidecar-dir output/rtdetr_recovery_horiz1_full_20260826 \
  --output artifacts/actor_ground_truth_metrics/recovery_horiz1_full_20260826
```

輸出 `actor_clip_metrics.csv`（以影片為獨立單位）與
`actor_event_metrics.csv`（含 release/birth 座標、距離、時間、margin 與 D/T/A
components）；`UNUSED` 與 `?` 不會被放入 accuracy 分母。

`stage` 可為 `distance_time`、`kalman_rts`、`confidence`、`uncertainty`、
`reverse` 或 `full`。其中 distance/time 以 birth anchor 與原始 actor
observation 為基準；
distance 與 time 先各自除以 hard gate 成為 0--1 無因次比例，再套用 trial
的 `distance_weight`、`time_weight`；runtime 可用上述兩個 DT 環境變數設定。
`kalman_rts` 只把均一 measurement confidence 的 Kalman/RTS 平滑 actor
位置交給 D+T，不使用 covariance gate/cost、Mahalanobis 或反向 trajectory；
`uncertainty` 才加入 confidence 與 covariance evidence，`reverse` 才引入反向
trajectory。此工具沒有
reviewed annotation 時只產生 candidate diagnostics，不能輸出 accuracy 結論。
`evaluate` 另報 selected non-NULL 的錯誤率與 deterministic bootstrap 95% CI；
它是安全風險指標，不能由 resolved/dustbin 比例取代。

Ground truth 已包含 release point／interval 與人工 actor bbox 時，可使用公式驗證工具
比較正確 actor、干擾 actor、尺度正規化和 D/T 權重。工具不會改寫標註或 production
參數；`first_visible_litter_frame=0/1` 依標註合約視為特定 baseline 的 RT-DETR miss
sentinel，不會納入 release 時差：

```bash
conda run -n rtdetr python scripts/validate_attribution_formula.py \
  --ground-truth runs/grounding_truth \
  --candidates output/backtrack_multi_actor_v1_20260825_rerun \
  --output artifacts/attribution_formula_validation_20260826 \
  --bootstrap 5000
```

輸出包括中英文 Markdown 報告、完整 JSON、event/actor mapping CSV、B0/B1 timing、
同幀 distance-definition、D/T sweep CSV 與對應 PNG 圖表。人工 actor ID 只以同幀、
同類 bbox IoU 建議對應 model tracklet；此對應與
由 actor presence 推導的 route 會明確標記 provenance，不能靜默寫回 canonical reviewed
annotation。報告中的 candidate AUC 使用 event-cluster bootstrap；binomial 命中率使用
Wilson 95% CI。沒有 negative clip、reviewed NULL route 或跨攝影機 test 時，不得把同場域
結果宣稱為 precision、NULL safety 或 universal optimum。

若要將距離與時間改寫成 TrackFlow-inspired 的負對數關聯成本，可在同一份
reviewed GT 與 frozen sidecar 上執行候選層級分析：

```bash
conda run -n rtdetr python scripts/analyze_attribution_log_likelihood.py \
  --ground-truth runs/grounding_truth \
  --candidates output/backtrack_multi_actor_v1_20260825_rerun \
  --output artifacts/attribution_log_likelihood_20260826
```

工具以 `D=release point 到 actor release region 的尺度正規化距離`、
`alpha=(B0-release_frame)/(B1-B0)`、training-fold-only 的時間中心，以及
`A=(1-cos(theta))/2` 的 forward-direction penalty 建立
`P(correct association)`，再評估 `-log(P)`。方向項採 von Mises 圓形統計形式，
並提供所有 penalty 係數不得為正的物理單調約束版本。主結果使用 leave-one-event-out，
並分開報 actor top-1、release hit 與兩者同時正確的 exact top-1。它只評估第一條
release→actor edge；沒有 reviewed NULL／negative clip 時，risk-coverage 不可宣稱為
正式 NULL safety 或自動開罰 threshold。

純 Kalman/RTS trial 可在 JSON 另外掃描
`kalman_process_noise_scale`、`kalman_measurement_noise_scale` 與
`kalman_max_extrapolation_seconds`。三者只調整 actor 平滑／補點，不啟用額外
cost component。Kalman 補點只提供位置；D+T 的時間差仍取最近真實 detection
frame，不會因預測點剛好落在 release frame 就被改寫為零。

`reverse/full` 的 release 時間先驗只使用同一條 confirmed litter track 的
accepted RT-DETR observations。令 `B0`、`B1`、`B2` 為前三個不同 detection
frame，`H0 = T_B1 - T_B0`，零成本可疑窗為
`I0 = [T_B0 - H0, T_B0]`。候選時間 `T_r` 超出此窗時才加入
軟性 prior；反向區使用
`lambda_w * (T_B0 - H0 - T_r) / H0`，birth 後的已觀測 airborne 區則保留
`lambda_f * (T_r - T_B0) / FPS`，不作 hard reject。早期來源方向取
`-(p_B1-p_B0)`，`B2` 只用來計算相鄰速度 cosine consistency。
`confirm_frame` 不參與 window 或方向。`SMART_BACKTRACK_MAX_BACK_FRAMES`
仍限制枚舉量，但 sidecar 會標記 `search_truncated`，不可把它解讀成物理上
不可能更早 release。

建立人工盲標 queue 時使用 `scripts/backtrack_annotations.py init`；輸出的
annotation schema 不複製 selected route、cost、rank 或 release prediction。

### Release 時間與距離成本優化研究

可透過 `StudyConfig` 啟用車輛 `C_BC` 的 `D/0.4`、`T_E/0.25` 成本，以及
release 回推的分段平方 prior（0.25 秒內不加罰，最多 1 秒）。新時間 prior
取代原本的 backward window 成本，保留短軌跡 prior、forward penalty 與 NULL。
Production 預設仍維持 D=0.3 與 observation-gap prior；新的組合先作明確的
研究設定。`scripts/replay_release_policy.py` 可重播四組控制實驗，逐影片比較最後
vehicle 是否正確，並分開列出數值 ID 正確率與歷史人工加分。
完整設定與重現命令見 [`backtrack README`](scripts/pipeline/backtrack/README.md)。

### Smart Backtrack 版本比較簡報

簡報由 live backtrack contract 產生，文字、方塊與數學式均為可編輯物件；可重建
16 頁 ODP 與 PPTX：

```bash
python3 scripts/create_backtrack_slides.py
```

輸出至 `artifacts/presentations/smart_backtrack_version_comparison_20260819.pptx`
與同名 `.odp`。內容比較 D+T 舊版與 release-synchronized 新版，並涵蓋
Kalman/RTS、confidence/covariance、C_BA/C_AC/C_BC、route cost、Min-Cost Flow、
NULL、margin 與 replay evidence。

## 輸出

假設輸入為 `resize.mp4`，且 `OUTPUT_ROOT=output`：

```text
output/resize_annotated.mp4
output/resize_annotated_analysis.json
```

`resize_annotated_analysis.json` 是每支影片各自產生、寫在標註影片同資料夾的
前端單檔資料源；不跨影片累積狀態。寫檔使用 `.tmp` 後 atomic replace，避免網頁讀到
半份 JSON。Production 不再另外輸出 `summary.json` 或 `events.jsonl`。

Schema version `2.1.0` 保留 `video`、`summary`、`events`，並加入
`litter_detection`：逐層列出 RT-DETR 4-channel 原始 candidate、geometry 通過、
motion/holding 通過的 bbox observation 總數與 0-based 幀號，最後另列 confirmed event
數。這能區分「模型沒有輸出」、「後處理淘汰」與「tracker 未確認」；candidate 仍不是
confirmed event 或 accuracy。`scripts/frontend/dashboard.html` 可直接載入。完整欄位、
範例與證據限制見
[`scripts/pipeline/ANALYSIS_JSON.md`](scripts/pipeline/ANALYSIS_JSON.md)。

`SMART_BACKTRACK_SIDECAR=0` 為預設。研究時明確設為 `1` 才會額外輸出
`*_backtrack_candidates.jsonl`；該 sidecar 不供網頁使用，也不是 ground truth。
Replay 用的 `resolver_input` 不保存 OCR 專用 `plate_actor_frames`／`plate_roi` 像素；
Smart resolver 不讀取這些影像資料，人物／車輛幾何、ID、confidence 與垃圾軌跡仍完整保留。

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

### UI tests 與 frontend build

```bash
conda run -n rtdetr python -m pytest -q tests/ui
cd UI/frontend && npm run build
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
