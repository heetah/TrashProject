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
│   │   ├── calibration/         # Online pseudo-homography calibration
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
      │     -> vehicle-contained quarantine / same-carrier release evidence
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
4. Camera-shake evidence；8/27 recovery profile 允許候選繼續進入下游 gates。
5. Actor polygon/relative-motion holding gate。
6. 幾乎完全位於 vehicle/scooter 內的候選進入 tracker-only quarantine，不可直接 confirmation。
7. Quarantine 只能由同一載體座標系中連續下落軌跡解除；離開載體的普通候選改建獨立 pending 軌跡，禁止隔離歷史污染正常確認。
8. `GlobalLitterTracker` trajectory、displacement、temporal confirmation。

稀疏 detector observation 可使用兩種有界 temporal evidence，但都不能自行建立軌跡：

- 已有兩個 detector anchors 的 quarantine 軌跡，允許下一幀使用一次 grayscale change-component 補點，以完成至少三個同載體 observation；補點不增加 detector observation count，也不能授權後續 containment→ordinary identity handoff。
- 單一 detector seed 只有在 bbox 至少占畫面 `0.001`、連續取得四個 prediction-gated change-components，且最後仍通過 holding、trajectory、displacement 與 stationary gates 時才可確認。小物件不能使用這條鏈。

Pending bbox 尺寸劇變時，短 gap association 另以 constant-velocity residual／bbox diagonal
檢查 motion continuity；完整 missed-window 後的 quarantine observation 必須建立新 identity，
避免舊車身候選吸收後來真正的 release。沒有可歸因 actor 的候選只有在至少五個 observation、
具有內部 apex/descent 的強重力弧線且通過其餘物理 gate 時，才能確認 object event；下游仍保留
`NULL` route，不因事件確認而強制歸因。

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
`D = dist(p, B) / sqrt(w²+h²)`，要求 `D <= 0.40`。Actor evidence 與 release
的時間差使用 hybrid soft penalty：令
`z = max((Δframe/FPS)/0.25, Δframe/3)`，再計算
`ρ(z) = z + κ max(0,z-1)²`，production 預設 `κ=4`。`0.25 s` 與 `3 frames`
是 soft boundary，不再直接刪除超界候選；固定秒數保留跨 FPS 的物理意義，
frame尺度限制逐幀 detector observation 的陳舊程度。距離／不確定性 hard gate、
有界 Kalman horizon 及完整 `NULL` route 仍保留，避免 forced match。

Backtrack sidecar 用於標註、成本校正與 gate 分析。沒有人工 reviewed ground truth 時，只能報告 candidate coverage/resolved/dustbin，不能宣稱歸因準確率。

2026-09-15 已將被後續研究取代的一次性 Python 工具與測試移出 active tree，封存於
`artifacts/research_python_archive_20260915.tar.gz`。下文若提到這些歷史工具，代表既有
研究方法與結果；如需重跑，先在隔離目錄解開封存檔。Production、目前 48/58 研究成果、
Phase 0/1A、mask diagnostics、camera holdout 與 release-validation 工具不在清理範圍。

Visible segmentation mask 與 bbox overlap 目前只作研究診斷，權重為 0。它們只能描述
影像中的可見包含／鄰近關係，不能等同真實深度，也不能直接判定垃圾來源。獨立的
歷史的 `build_ordinal_occlusion_package.py` 曾建立匿名 A/B 車輛遮擋標註 queue；原始碼
現已封存，既有研究結論保留。該 queue 不讀取
route、assignment、release 或 ground-truth actor，但目前仍由事件 sidecar 取樣且使用模型
bbox proposal，因此只適合可行性研究。未取得單一複核者的盲化重標一致性、可驗證相機群組
與 camera-separated holdout 前，不得把 ordinal 訊號加入 production costs。

Release hypothesis 另有已完成的獨立可識別性稽核；其一次性
`analyze_release_identifiability.py` 已封存。該研究只讀 resolver sidecar，統計每個事件的
觀測點數、hypothesis 範圍、協方差與搜尋截斷。最新 64 筆 records 中，36 筆只有兩個
觀測點、28 筆的反推搜尋仍被計算上限截斷；兩點資料不能唯一識別 birth 前軌跡。因此
這兩類事件不能用未驗證的 release cost 強行配對，未來應先建立獨立 camera holdout 的
不確定性／abstention 驗證。

### Online pseudo-homography calibration

`scripts/pipeline/calibration/` 依序建立可跨攝影機部署的自我校正資料流。Phase 1
目前只收集 YOLO-Seg 真實觀測：從 vehicle/scooter segmentation mask 的底部
98-percentile band 取中位數接地點，經面積、邊界、confidence、track continuity 與
teleport quality filter 後，寫入限時、限量的輕量 trajectory buffer，再保留彎道所需的
local tangent observations。跳幀 cache (`observed=false`) 不會重複成為 calibration evidence，
也不保存 frame 或完整 mask。

Phase 2 將 local observations 放入 configurable image grid，以 circular medoid、
angular-residual trimming、weighted median speed 建立 robust traffic motion field；接著用
正規化位置、局部方向及同一 track 的 segment continuity 執行 grid-indexed lightweight
DBSCAN。它不會只因兩段軌跡方向相同就跨越不同道路區域合併，也不會把反向車流平均掉。
Debug API 可在 frame copy 上畫 local segments、cell dominant arrows 與 flow clusters，
不進入 production renderer。

Phase 3 提供 deterministic initial transform：`H0=diag(1/width,1/height,1)`，將
image rectangle 映射到相對 `[0,1]×[0,1]` pseudo-ground。這只是安全且非隨機的
relative-scale baseline，不是透視校正完成或真實公尺。`image_to_ground()` 對分母接近0、
NaN/Inf 及輸出爆炸的 row 回傳 NaN；matrix validator 另檢查 determinant、condition
number、完整 image grid 的 denominator variation/sign、corner orientation、projected area
與 bounds。Analysis snapshot 保存 H、版本0、confidence 0、validation report，且仍標示
`affects_attribution=false`。

Phase 4 對任一通過 Phase 3 safety validation 的候選 H 計算四個 robust diagnostics：
相鄰 ground-plane velocity 的相對變化、相鄰 speed ratio 的 log 變化、連續 turning
angle 的差（curvature discontinuity），以及同一 traffic-flow cluster 經局部 Jacobian
近似投影後的方向離散。前三者按 observation quality 加權，所有 residual 經 Huber
penalty；速度以每條 track 的 median projected speed 正規化，direction 只比較角度，
因此候選 H 不能只靠縮小輸出座標降低 loss。合法轉彎或平順加減速不被強迫為零。
Analysis snapshot 會輸出各 component loss、sample count 與 active-weight normalized total；
這些是候選矩陣的相對 objective，不是 accuracy、probability 或已完成的 calibration。

Phase 5 再加入 cross-vehicle lane/flow consistency。每個 observation 只使用同一 flow
cluster、局部影像鄰域中「其他 track」的點，以 quality-weighted moving median 建立
leave-one-track-out 局部 centerline，並以其他車輛的 robust local tangent 計算垂直距離。
因此不需要不同車輛在同一時間或同一 longitudinal position 對齊，也不會把整條彎道擬合成
直線。距離除以投影 image footprint 面積平方根，使 uniform coordinate scaling 不會降低
成本；鄰居以固定 image-space grid 尋找，典型複雜度接近 `O(N·k)`。Lane loss 仍只是
flow-cluster 內的幾何診斷；沒有 lane label 時不可把它單獨解讀成真實車道寬或 calibration
accuracy，Phase 6 也必須與 perspective constraint 和其他 loss 共同評估。

Phase 6 提供 opt-in bounded candidate search。它不直接自由調整 H 的 9 個元素，而把
候選寫成 `H_candidate=P(theta)H_previous`；`theta` 只包含 log-anisotropy、兩個 shear
與兩個 perspective coefficient。搜尋採 deterministic coordinate perturbation、逐輪縮小
step，所有候選都必須通過 Phase 3 hard validation。Objective 為 data loss、local-Jacobian
anisotropy/area-variation perspective regularization，以及新舊 H 在固定 image grid 上投影
位移的 temporal prior 加權平均。

Confidence 明確拆成 track count、independent flow count、spatial coverage、trajectory
duration、ground-point quality、motion-field stability、residual、lane evidence、candidate
improvement 與 historical stability 十項 `[0,1]` 分數。即使平均 confidence 足夠，只要
valid tracks、flow、coverage、duration、relative improvement 任一 hard evidence gate 不足，
仍輸出 freeze。通過時 `alpha=max_alpha×confidence×improvement_score`，在 theta 空間做
small-step proposal，然後重新做 geometric validation 與 objective improvement 檢查。
目前 `DYNAMIC_HOMOGRAPHY_OPTIMIZE=0`，安全預設只輸出
`applied_to_production=false` 的建議。若另外明確啟用 optimizer 與
`SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=1`，Smart Backtrack 只會接受事件確認當下
`LOCKED`、達最低信心且再次通過矩陣驗證的 frozen H；其他事件保留 image-space 成本。

Phase 7 由 `DynamicHomographyCalibrator` 持有每支攝影機各自的 H、version、confidence、
rolling evidence、rollback history 與 calibration-window metrics，狀態依序為
`UNCALIBRATED → COLLECTING → ESTIMATING → WARMING_UP → LOCKED`；證據不足則進入
`LOW_CONFIDENCE` 並凍結 H，累積更多觀測後再估計。只有 optimizer proposal 通過時
才增加 version；收斂後必須再連續出現足夠 stable windows 才 LOCKED。LOCKED 不再持續
調 H，只監看 traffic residual、normalized motion centroid/spread、direction tensor，以及
可選的 static-background motion signal。

單一異常 window 不會觸發重校正；連續達 `DRIFT_WINDOWS` 才轉成
`DRIFT_DETECTED → RECALIBRATING`。資料不足、optimizer disabled、candidate invalid 或
coverage 不足都保留舊 H。每次成功更新前保存 rollback snapshot。若 frame resolution/crop
尺寸改變，舊 H 與舊 buffer 不會混用，狀態會停在 `DRIFT_DETECTED` 並要求明確 `reset()`
後以新影像尺寸重新蒐證。這個 state 仍完全隔離於事件歸因。

Phase 8 提供預設關閉的 `StaticBackgroundStabilizer`。呼叫端必須提供涵蓋已知
vehicle/person/litter 的 exclusion mask；否則不估計。模組只在剩餘 static background
上抽 ORB features，以 ratio test + RANSAC 求 `G_current_to_reference`，並限制 inlier
ratio、translation、rotation、scale 與 orientation。每支攝影機只保留一組 reference
keypoints/descriptors 和最後有效 G，不保存 frame history；新估計失敗時保留最後有效 G。
Runtime transform 定義為 `H_runtime=H_calibration@G_current_to_reference`。

`capture_event_snapshot()` 可建立唯讀的 H version、confidence、calibration H、G 與
runtime H 快照，確保事件反追蹤不會混用不同座標版本。其 consumer 由
`SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY` 獨立控制，未達條件會留下原因並安全 fallback。
`render_calibration_debug()` 在 frame copy 上同時顯示 image-space
tracks、local motion/flow clusters、pseudo-ground inset 與 calibration statistics，完全不
進入正常 renderer。

`DYNAMIC_HOMOGRAPHY=0` 與 `SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=0` 均為安全預設。
Phase 1–8 的診斷本身不會覆蓋 backtrack H；只有通過事件級安全閘門的 opt-in consumer
才改用 relative pseudo-ground spatial features，且不修改垃圾確認或 OCR。完整架構、公式、
狀態、效能與驗收對照見
[`scripts/pipeline/calibration/FLOW.md`](scripts/pipeline/calibration/FLOW.md)。只有 reviewed
continuous multi-camera replay 通過後，才能考慮預設啟用。

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
| `RTDETR_IMGSZ` | 空值 | 選用的 PyTorch RT-DETR inference size；空值沿用 checkpoint/runtime 預設，固定 shape engine 不受此值改變 |
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
| `SMART_BACKTRACK_MASK_DIAGNOSTICS` | `0` | 只記錄 observed YOLO-Seg mask 與 raw litter bbox 的純量關係；不進入成本、route 或 NULL 判定 |
| `SMART_BACKTRACK_STUDY_STAGE` | `full` | Research ablation stage；production 預設不變 |
| `SMART_BACKTRACK_RAW_PREFIX` | `1` | confirmed event 才能使用 pre-postprocessing RT-DETR bbox 補 release trajectory；不參與 event confirmation |
| `SMART_BACKTRACK_DT_DISTANCE_WEIGHT` | `1.0` | D+T stage 的 gate-normalized distance weight |
| `SMART_BACKTRACK_DT_TIME_WEIGHT` | `1.0` | D+T stage 的 hybrid observation-time weight |
| `SMART_BACKTRACK_TIME_COST_MODE` | `soft` | `soft` 使用連續超界懲罰；`hard` 僅供重播舊 AND gate |
| `SMART_BACKTRACK_TIME_SOFT_KAPPA` | `4.0` | `z>1` 後的平方超界曲率；須以 event-grouped validation 校正 |
| `SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY` | `0` | opt-in 使用事件級 frozen H；未 LOCKED、低信心或無效 H 逐事件退回 image-space |
| `SMART_BACKTRACK_HOMOGRAPHY_MIN_CONFIDENCE` | `0.65` | 事件可使用 dynamic H 的最低 state confidence；不是 accuracy probability |
| `DYNAMIC_HOMOGRAPHY` | `0` | 啟用 Phase 1 mask 接地點／軌跡／local-motion 蒐集；目前不影響歸因 |
| `DYNAMIC_HOMOGRAPHY_WINDOW_SEC` | `100` | 每支攝影機的輕量 rolling trajectory window |
| `DYNAMIC_HOMOGRAPHY_BOTTOM_PERCENTILE` | `98` | vehicle mask 底部接地帶 percentile |
| `DYNAMIC_HOMOGRAPHY_MIN_MASK_AREA` | `64` | calibration mask 最小像素面積 |
| `DYNAMIC_HOMOGRAPHY_MIN_CONF` | `0.45` | calibration vehicle observation 最低信心 |
| `DYNAMIC_HOMOGRAPHY_QUALITY_THRESHOLD` | `0.60` | 綜合 observation quality 下限 |
| `DYNAMIC_HOMOGRAPHY_GRID_ROWS` / `GRID_COLS` | `12` / `20` | Phase 2 spatial motion field 網格 |
| `DYNAMIC_HOMOGRAPHY_DIRECTION_TRIM_FRAC` | `0.15` | cell circular residual 的 robust trimming 比例 |
| `DYNAMIC_HOMOGRAPHY_FLOW_POSITION_RADIUS` | `0.10` | flow clustering 的影像尺寸正規化空間半徑 |
| `DYNAMIC_HOMOGRAPHY_FLOW_ANGLE_DEG` | `30` | 不同 track local segments 的最大方向差 |
| `DYNAMIC_HOMOGRAPHY_FLOW_MIN_TRACKS` | `2` | flow cluster 最少獨立 vehicle tracks |
| `DYNAMIC_HOMOGRAPHY_MAX_CONDITION` | `1000000` | Phase 3 H condition-number 上限 |
| `DYNAMIC_HOMOGRAPHY_MIN_DENOMINATOR` | `1e-6` | homogeneous projection 安全分母下限 |
| `DYNAMIC_HOMOGRAPHY_MAX_DENOM_RATIO` | `100` | image ROI 內最大／最小投影分母比例 |
| `DYNAMIC_HOMOGRAPHY_REQUIRE_ORIENTATION` | `1` | 拒絕鏡射 pseudo-ground coordinate system |
| `DYNAMIC_HOMOGRAPHY_*_WEIGHT` | `1.0` | Phase 4 motion/speed/curvature/direction loss 權重；相等值只是中性起點，須用獨立 replay 校正 |
| `DYNAMIC_HOMOGRAPHY_MOTION_HUBER_DELTA` | `0.50` | 相鄰 velocity change／track median speed 的 Huber 轉折點 |
| `DYNAMIC_HOMOGRAPHY_SPEED_HUBER_DELTA` | `0.35` | `abs(log(v_i/v_{i-1}))` 的 Huber 轉折點 |
| `DYNAMIC_HOMOGRAPHY_CURVATURE_HUBER_DELTA` / `DIRECTION_HUBER_DELTA` | `0.261799` | 角度 residual 的 15° robust 轉折點；為初始 prior，非 accuracy 證明 |
| `DYNAMIC_HOMOGRAPHY_DIRECTION_PROBE_FRAC` | `0.01` | 用影像對角線 1% 的局部 probe 近似 H 在 observation 周圍的方向映射 |
| `DYNAMIC_HOMOGRAPHY_LANE_WEIGHT` | `1.0` | Phase 5 cross-vehicle local centerline loss 權重；中性起點，非已驗證常數 |
| `DYNAMIC_HOMOGRAPHY_LANE_HUBER_DELTA` | `0.03` | 以 projected footprint scale 正規化後的 lane residual Huber 轉折點 |
| `DYNAMIC_HOMOGRAPHY_LANE_NEIGHBORHOOD_FRAC` | `0.10` | 以影像對角線比例定義的固定局部 peer 搜尋半徑 |
| `DYNAMIC_HOMOGRAPHY_LANE_MIN_PEER_OBS` / `LANE_MIN_PEER_TRACKS` | `2` / `1` | 每個 leave-one-track-out centerline 的最低其他車輛證據 |
| `DYNAMIC_HOMOGRAPHY_OPTIMIZE` | `0` | 執行 Phase 6 bounded candidate search；只產生建議，不套用 production H |
| `DYNAMIC_HOMOGRAPHY_OPT_*_STEP` | `0.08` / `0.04` / `0.08` | anisotropy、shear、perspective 的初始 dimensionless search step |
| `DYNAMIC_HOMOGRAPHY_OPT_ITERATIONS` / `STEP_DECAY` | `3` / `0.5` | coordinate search 輪數與每輪 step 衰減 |
| `DYNAMIC_HOMOGRAPHY_OPT_DATA_WEIGHT` | `1.0` | Phase 4/5 traffic-data loss 在 optimizer objective 的權重 |
| `DYNAMIC_HOMOGRAPHY_OPT_PERSPECTIVE_WEIGHT` / `TEMPORAL_WEIGHT` | `0.25` / `0.25` | 局部投影正則化與 previous-H temporal prior 權重；皆為待 replay 校正的初始 prior |
| `DYNAMIC_HOMOGRAPHY_OPT_MIN_VALID_TRACKS` / `MIN_FLOW_CLUSTERS` | `4` / `1` | safe-update 最低跨車輛與獨立 flow 證據 |
| `DYNAMIC_HOMOGRAPHY_OPT_MIN_SPATIAL_COVERAGE` / `MIN_DURATION_SEC` | `0.01` / `10` | safe-update 最低 motion-grid 覆蓋與觀察時間 |
| `DYNAMIC_HOMOGRAPHY_OPT_MIN_REL_IMPROVEMENT` | `0.01` | candidate objective 至少相對改善 1%，否則 freeze |
| `DYNAMIC_HOMOGRAPHY_OPT_FREEZE_CONFIDENCE` / `MAX_UPDATE_ALPHA` | `0.55` / `0.20` | confidence freeze threshold 與單次 theta-space update 上限 |
| `DYNAMIC_HOMOGRAPHY_EVAL_INTERVAL_SEC` | `30` | Phase 7 calibration evaluation 間隔；逐幀只收集輕量 observation |
| `DYNAMIC_HOMOGRAPHY_LOCK_MIN_UPDATES` / `LOCK_STABLE_WINDOWS` | `2` / `2` | 進入 LOCKED 前的最低成功更新與連續穩定 window |
| `DYNAMIC_HOMOGRAPHY_DRIFT_WINDOWS` | `3` | LOCKED 後必須連續出現 drift signal 的 window 數 |
| `DYNAMIC_HOMOGRAPHY_DRIFT_RESIDUAL_RATIO` / `DRIFT_ABS_INCREASE` | `2.0` / `0.05` | 相對及絕對 residual drift 門檻，取較嚴格者 |
| `DYNAMIC_HOMOGRAPHY_DRIFT_CENTROID_FRAC` / `DRIFT_SPREAD_FRAC` | `0.08` / `0.08` | normalized traffic geometry 中心與分布改變門檻 |
| `DYNAMIC_HOMOGRAPHY_DRIFT_DIRECTION_TENSOR` | `0.25` | 車流方向二階矩陣的 Frobenius drift 門檻 |
| `DYNAMIC_HOMOGRAPHY_ROLLBACK_HISTORY` / `METRICS_HISTORY` | `8` / `100` | 每攝影機保留的 H rollback 與 calibration-window 紀錄上限 |
| `DYNAMIC_HOMOGRAPHY_CONFIDENCE_ALPHA` / `DRIFT_CONFIDENCE_DECAY` | `0.25` / `0.80` | persistent confidence EMA 與 drift-window 衰減 |
| `DYNAMIC_HOMOGRAPHY_STABILIZE` | `0` | Phase 8 static-background ORB stabilization；caller 必須提供完整 dynamic exclusion mask |
| `DYNAMIC_HOMOGRAPHY_STAB_MIN_MATCHES` / `MIN_INLIER_RATIO` | `20` / `0.50` | RANSAC 前最低 static matches 與接受 G 的最低 inlier 比例 |
| `DYNAMIC_HOMOGRAPHY_STAB_MAX_TRANSLATION_FRAC` | `0.08` | G translation 相對影像對角線的安全上限 |
| `DYNAMIC_HOMOGRAPHY_STAB_MAX_ROTATION_DEG` / `MAX_SCALE_CHANGE` | `5.0` / `0.08` | 單次 background transform 的旋轉與尺度安全上限 |
| `SMART_BACKTRACK_TWO_POINT_MAX_BACK_SEC` | `0.4` | 舊 replay 相容欄位；新版不再作為兩點軌跡的物理截止 |
| `SMART_BACKTRACK_TWO_POINT_PRIOR_COST` | `1.0` | 兩點常速 release hypothesis 基礎 prior cost |
| `SMART_BACKTRACK_MAX_FORWARD_RELEASE_SEC` | `0.5` | ballistic release window 可晚於 detector birth 的上限；仍受已觀測 airborne 軌跡限制 |
| `SMART_BACKTRACK_RELEASE_WINDOW_WEIGHT` | `0.35` | 超出 B0/B1 零成本窗後，每一個 observation-gap 的軟性 prior 增量 |
| `SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT` | `0.0` | `full/reverse` 中 release 點位於 vehicle bbox 深處的軟成本；`0` 關閉，須經 reviewed replay 後才啟用 |
| `LITTER_DEBUG` | `0` | 設為 `1` 時輸出逐幀診斷；annotated video 顯示 actor track ID，並以洋紅框顯示 RT-DETR 通過 class/confidence、但尚未經 geometry/motion/holding/tracker 後處理的 litter bbox 與 confidence |
| `LITTER_FP_CONTAINMENT_THR` | `0.999` | 車輛容納判定門檻；命中時不丟棄候選，而是進入無法直接 confirmation 的 tracker-only quarantine |
| `LITTER_VEHICLE_QUARANTINE_MIN_OBSERVATIONS` | `3` | 同載體相對軌跡至少 observation 數，不足時保持 pending |
| `LITTER_VEHICLE_QUARANTINE_MIN_RELATIVE_DISPLACEMENT` | `25` | 解除隔離所需的載體相對總位移下限（px），並取物件初始尺寸的較嚴格值 |
| `LITTER_VEHICLE_QUARANTINE_MIN_RELATIVE_DOWNWARD` | `15` | 解除隔離所需的載體相對向下位移下限（px） |
| `LITTER_VEHICLE_QUARANTINE_MIN_SCALE_RATIO` | `1` | 依初始 litter bbox 尺寸縮放的位移／下落證據下限 |
| `LITTER_VEHICLE_QUARANTINE_MIN_DOWNWARD_STEPS` | `2` | 解除隔離前至少連續向下的 step 數 |
| `LITTER_VEHICLE_QUARANTINE_MAX_GAP_SEC` | `0.35` | 隔離證據的最大 detector 中斷；超過後重置證據段，避免將無關 bbox 接成落下軌跡 |
| `LITTER_CANDIDATE_SIDECAR` | `0` | 研究用逐 candidate JSONL，記錄 gate reason、tracker ID 與可重現設定；不供前端或 ground truth 使用 |
| `LITTER_CANDIDATE_DEDUP` | `0` | 實驗性同幀 IoU 去重；2026-09-12 僅與 streak probation 候選組合通過 58 正片 development replay，缺 reviewed negatives／camera holdout，production 仍關閉 |
| `LITTER_CANDIDATE_DEDUP_IOU` | `0.5` | 同幀 candidate 去重 IoU 門檻；僅在 `LITTER_CANDIDATE_DEDUP=1` 時生效 |
| `LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR` | `0` | 8/27 recovery profile 允許 actor 在垃圾首個 observation 後才被偵測到；confirm 當下仍須有 actor 且通過其餘證據 gates |
| `LITTER_MIN_CONFIRM_AGE_VEHICLE` | `2` | vehicle/scooter thrower 的最少 observation 次數；短軌跡須通過尺寸正規化絕對位移或「起點近 actor、終點已脫離」證據，兩點 fast-drop 另須 ≥35 px downward |
| `LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE` | `7` | vehicle/scooter thrower 的最小向下位移（px） |
| `LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT` | `1` | confirm 所需最小水平位移（px），允許近垂直落下案例 |
| `LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE` | `10` | vehicle thrower 水平／向下位移最大比例，保留極端水平滑動抑制 |
| `LITTER_MIN_VEHICLE_RELATIVE_SEPARATION` | `0` | 關閉載體車輛相對分離 hard gate；相對分離仍可保留為診斷資料 |
| `LITTER_FP_STREAK_RATIO` | `10` | 前處理水平 streak 與向下位移比例門檻 |
| `LITTER_FP_STREAK_MIN_OBSERVATIONS` | `2` | 包含本幀的最小 streak 判決 observation 數；`2` 保持 production 行為，較大值只延後拒絕且不能直接 confirmation |
| `LITTER_FP_STREAK_DEFER_MAX_STEP_DIAGONALS_PER_FRAME` | `1.1` | probation 期間相鄰候選的最大尺度正規化跳躍：`px / (source-frame × current-bbox-diagonal-px)`；只在 observation 門檻大於 `2` 時影響結果，屬未獨立校正研究值 |
| `LITTER_ALLOW_SHAKE_CANDIDATES` | `1` | camera-shake frame 的 candidate 仍提交給下游 motion/holding/tracker gates |
| `OUTPUT_ROOT` | `.` | Output directory；建議明確設為 `output` |

其餘 action smoothing、video I/O、writer、Smart Backtrack 與 legacy research 開關都已列在
`.env.example`。演算法內部固定 class mapping、4-channel preprocessing contract 與低頻物理 gate
仍保留在 live code，避免一般部署調參意外改變證據責任。型別化 production 預設以
`scripts/pipeline/config.py` 為準。

### 反追蹤研究 replay

`scripts/backtrack_study.py` 只重播 confirmed event sidecar 內的 frozen
resolver input；不重跑 detector，也不改變 litter confirmation。`--candidates`
可接受單一 JSONL，或遞迴尋找 production batch `case_<id>/` 內的 sidecar。先由新 sidecar
建立不可變 group split，再只在 development/validation 調參，最後才讀 test：

```bash
conda run -n rtdetr python scripts/backtrack_study.py manifest \
  --candidates artifacts/backtrack_candidates --output artifacts/study_manifest.json

conda run -n rtdetr python scripts/backtrack_study.py replay \
  --candidates artifacts/backtrack_candidates --manifest artifacts/study_manifest.json \
  --config artifacts/distance_time.json --split validation \
  --output artifacts/distance_time_validation.jsonl
```

多組 frozen replay 完成後，歷史研究曾用已封存的
`report_backtrack_route_ablation.py` 合併相同 58 案 readiness/paired comparison，
並輸出 accepted-event 的 GT route rank、cost gap 與 `C_BA/C_AC/C_BC` raw feature
拆解。若來源 metadata 無法建立獨立 camera group，結果只能標為 development
characterization，不得宣稱 validation/test accuracy 或修改 production weight。

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
`review_state` 在當時仍未寫入；該 metadata 遺漏已於 2026-09-13 依人工複核者確認補正。

`exit_code=0` 只代表 pipeline 完成。58 筆既有正片事件已由唯一負責人確認先前完成
人工複核，並於 2026-09-13 將遺漏的 `review_state` metadata 更正為 `reviewed`；此更正
沒有改動 event、release、actor bbox、route 或 ID。其餘未經人工 reviewed route 的資料
仍只能比較 candidate coverage、route 變化與 margin，不可宣稱 attribution accuracy。

2026-08-26 的研究 recovery replay（完整 63 部）使用 sidecar 記錄的放寬門檻，
得到 41/63 confirmed clips（usable 41/58，Wilson 95% CI 52.75%--75.67%），達到
「超過 40 部」的短期 coverage 目標。該批輸出後續經專案人員人工檢查，於
2026-09-07 提升為 production confirmation profile；這項決策提高 coverage，但因仍缺少
足夠 negative/background clips，不得把 coverage 解讀為 precision、普適 accuracy 或
自動開罰安全性。完整命令、paired CI 與原始回滾方式記於
`versions/2026-08-26_codex_confirmation_recovery_replay.md`；機器可讀結果位於
`artifacts/litter_postprocess_calibration/recovery_horiz1_full_20260826/`。

2026-09-14 的 58 個 reviewed positive clips 完整 paired regression 使用 production
RT-DETR 權重，以及 `LITTER_CANDIDATE_DEDUP=1`、
`LITTER_FP_STREAK_MIN_OBSERVATIONS=6` 的 streak probation 研究設定，
assignment-conditioned accepted event match
由 41/58 提升至 45/58（77.59%，Wilson 95% CI 65.34%--86.41%），新增案例為
23、30、69、75，accepted-event loss=0。58/58 pipeline jobs 完成且輸出 MP4 皆可解碼；
逐案結果與鎖定 manifest 位於 `artifacts/target45_full_20260913/readiness_v2/` 及
`artifacts/target45_full_20260913/experiment_manifest_v2.json`。這是 assignment-conditioned
event sensitivity，不是 detector-only recall；正樣本資料不能估 precision/FPR，且目前沒有
獨立 camera-group holdout、reviewed negative set 或 plate OCR ground truth，因此不可據此宣稱
production enforcement ready。

同日的 development-only cascade 保留上述 primary 輸出，並在 primary
`confirmed_event_count` 不在 `[1, 3]` 時選用 alternate 4-channel checkpoint 的
1216-pixel secondary 結果。選擇規則只讀模型輸出，不讀人工標註；一次性的
`build_litter_cascade_view.py` 已隨該研究封存，既有 evidence 與版本紀錄仍保留。

完整 58 案得到 50/58 assignment-conditioned accepted event matches（86.21%，
Wilson 95% CI 75.07%--92.84%），相對 45/58 新增 18、34、135、143、193，loss=0；
route correctness 為 33/58。這不是 production 預設或 enforcement release：觸發規則與
secondary 尺度均在同一 positive development cohort 選定，尚缺 reviewed negatives、
independent camera-group holdout 與 plate OCR ground truth；cascade 另新增四個未 accepted
的 exploratory confirmations，沒有人工負樣本可判定是否為 false positive。證據位於
`artifacts/target50_full_20260914/cascade_readiness/`、
`artifacts/target50_full_20260914/cascade_paired/` 與
`output/target50_cascade_composite_20260914/cascade_selection.jsonl`。

將人工 actor ID 表與 recovery sidecar 對齊，可使用：

```bash
conda run -n rtdetr python scripts/summarize_actor_ground_truth_metrics.py \
  --sidecar-dir output/rtdetr_recovery_horiz1_full_20260826 \
  --acknowledge-run-local-ids \
  --output artifacts/actor_ground_truth_metrics/recovery_horiz1_full_20260826
```

輸出 `actor_clip_metrics.csv`（以影片為獨立單位）與
`actor_event_metrics.csv`（含 release/birth 座標、距離、時間、margin 與 D/T/A
components）；`UNUSED` 與 `?` 不會被放入 accuracy 分母。此工具只供重現當初抄錄
numeric tracker ID 的同一輪歷史輸出；tracker ID 每輪皆可能改變，禁止把這份 ID 表
套到新推論輸出或用於產品 readiness。

新推論輸出必須以同輪 actor bbox→tracklet IoU 對映、固定正樣本分母與 fail-closed
事件配對重新評估：

```bash
conda run -n rtdetr env PYTHONPATH=scripts python \
  scripts/evaluate_reviewed_readiness.py \
  --ground-truth runs/grounding_truth \
  --candidates output/groundtruth_batch_dynamic_h_20260907 \
  --output artifacts/product_readiness_same_run_20260907 \
  --target 0.85
```

工具只接受 strict/moderate 時空 event match；miss、exploratory、actor mapping 失敗、
NULL 與錯誤 route 全部留在分母。輸出 Wilson 95% CI、標註 review state、輸入 hash 與
release gate。沒有 reviewed event labels、reviewed negative set 或獨立攝影機 holdout 時，
`ready_for_enforcement` 必為 false；沒有 reviewed plate/OCR truth 時，可開罰案件正確率
也一律標為未評估。研究用 release-window replay可加
`--replay-max-release-back-seconds 1.0`，但不會改動 production 預設。

2026-09-07 對 dynamic-H 58 案輸出套用此協定後，strict/moderate event sensitivity
為 32/58（55.17%，Wilson 95% CI 42.45%--67.25%），暫定端到端 exact route 為
23/58（39.66%，Wilson 95% CI 28.09%--52.51%）。1.0 秒 release replay 為
25/58，但 paired gain/loss=3/1、exact p=0.625，且 event match 有退化；0.5 秒為
23/58、gain/loss=2/2。兩者均未通過 promotion gate，production 參數未變。
上述 58 筆 event annotation 的 `review_state` 已於 2026-09-13 依唯一人工複核者的確認
更正為 `reviewed`。這是 metadata 補正，不是重新盲標或獨立複核；資料仍缺 reviewed
negatives 與跨攝影機 holdout，因此上述數值只能作 development evidence，不能作 85%
上線聲明。

2026-09-14 的 Smart Backtrack Phase 0 新增逐點 observation provenance/lineage 診斷。
既有 `history_sources=accepted_tracker/raw_rtdetr_recovered` 保持相容；新增
`history_provenance` 區分 `detector`、`visual_bridge`、raw recovery 與舊資料未知來源，
`history_lineage` 則標出父觀測、independence group 與是否為獨立 measurement。所有
history 平行陣列在排序及 raw prefix 補點後維持同長對齊。Resolver 不讀新欄位；相同
task 加入或移除這些欄位時，route、person、vehicle 與全部 costs 必須完全相同。這只是
後續「釋放狀態與多假設平滑」研究的資料合約，沒有新增權重、門檻或 production 決策。
Phase 1A 另加入無 production caller 的純數學模組，僅提供嚴格 Gaussian NLL、caller-defined
不規則時間 prior mass 與包含 NULL 的 log-sum-exp route evidence；它不選 route、不讀
lineage 自動相乘觀測，也不把 posterior 稱作準確率。

2026-09-13 以較新 streak6/jump11/dedup frozen sidecar 執行 16 組單因子反追蹤研究。
baseline 為 31/58 exact route；`direct_vehicle_penalty=1.1` 單獨增加 case 194，
`release_window_prior_weight=0.2` 單獨增加 case 41，兩者合併為 33/58（56.90%，
Wilson 95% CI 44.12%--68.82%），在 41 筆 accepted event 中為 33/41（80.49%，
Wilson 95% CI 65.99%--89.77%），相對 baseline gain/loss=2/0。因調參與評估使用同一批
development 正片，且沒有 reviewed negatives／camera holdout，兩項只保留為 fine-tune
候選，production 預設不變。confirmation-time actor hint 另經 10 組 frozen rerank 驗證後
出現淨退化，已拒絕且未接入 production。combined 的 strict match 另由 16 降至 13、
moderate 由 25 增至 28；目前 event match 使用 selected route 的 release frame/point，
所以 41/58 是 assignment-conditioned match，不是純 detector sensitivity。

後續單幀 wrist-release 研究重用現有 YOLO-Pose、沒有新增 CNN，並在同輪 10 案 sidecar
上鎖定 12 組 `distance scale × weight` replay。12/12 均未修正預先指定的 case 141/16；
8 組反而使正確 control case 194 退化，另有一組使 case 42 strict→moderate。該方向已依
預設 rollback gate 完整移除，production/data-path/config 均不保留 wrist 欄位。完整負面
證據位於 `artifacts/wrist_release_study_20260913/`。下一方向只先研究 YOLO-Seg temporal
mask 的局部遮擋順序，禁止用「畫面較低／框較大」充當深度；該簡化規則已出現 gain 2、
loss 1（破壞 case 25）。

同日後續研究加入預設關閉的 `SMART_BACKTRACK_MASK_DIAGNOSTICS`。它只在原始
YOLO-Seg polygon 被丟棄前，計算 mask overlap、bbox containment、mask fill ratio 與
尺度正規化 signed distance；只接受 `observed=True` 的 `seg_track/seg_predict`，不使用
cache/Kalman，也不保存 polygon 或 bitmap。診斷資料完全不進 resolver。case 9 顯示錯車
bbox 可覆蓋垃圾約 90%，但 visible mask overlap 為 0，確認 bbox proximity 會產生假支持；
case 168 則顯示垃圾在人工正確背景車之前的另一車 visible mask 內，證明單幀 mask 不能
直接當「深度」或加權來源。case 67 在每幀 actor inference 的獨立 run 中改為正確 route，
但 mask overlap 仍為 0，改善來源是 detection cadence/release evidence，不是 mask。
目前沒有獨立 reviewed ordinal front/behind 標註，因此 temporal occlusion 仍只具研究潛力，
尚未提升或重新宣稱 58 案準確率。

已完成的 RT-DETR intermediate 研究使用預設不接入 production 的 `.pt` 中間張量
驗證器；一次性 `analyze_rtdetr_intermediates.py` 與其輔助模組已封存。該研究攔截
query-aligned encoder/final-decoder box/score 與 final-query
embedding，以 frozen confirmed-litter history 作 silver temporal label；輸出 manifest、逐案例
執行紀錄、逐 query/pair CSV 與 clip-cluster bootstrap 摘要。silver label 不是人工物件身分真值，
且工具不支援未額外匯出 hidden bindings 的 TensorRT engine，因此結果只能決定是否值得進入
camera-separated 研究，不能直接設定歸因加分或宣稱準確率提升。

第二階段 query-embedding 驗證只讀取 `resolver_input.history_sources` 中的
`accepted_tracker`，排除 `raw_rtdetr_recovered`；用前三個 accepted query 以不等間隔
OLS 常速模型預測第四點，並在 candidate universe、幾何距離與 embedding 距離凍結後才
揭露 target 的一對一 IoU label。缺少 positive query 仍計為失敗；primary 每條軌跡只取
第一個 holdout，rolling 只作敏感度描述。現有 62 條軌跡只有 5 條具四個 accepted 點，
production score floor 下 median candidate count 為 1，因此 embedding 沒有增量空間；
研究 floor 的 2 rescue / 0 harm 只有 5 clips（clip sign-flip `p=0.25`）。本方向目前維持
production weight 0，不能解讀為 58 案準確率提升。

車輛短軌跡 safety guard 的最終 58 案重跑維持 32/58 accepted events 與 23/58
exact routes，paired gain/loss 皆為 0/0；case 42 的一筆 exploratory confirmation 被移除，
且沒有新增 exploratory confirmation。這通過 development safety-change gate，但不是
accuracy 改善或 reviewed false-positive reduction。完整紀錄位於
`artifacts/groundtruth_batch_vehicle_evidence_final_20260908/REPORT.md`。

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
正式 NULL safety 或自動開罰 threshold。報告另列相對 D+T 的逐事件 exact gain/loss；
即使 NLL 改善，只要目標 exact top-1 退化，就不能升級為 route selector。

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

要把既有事件標註整理為正式 release validation 複核包，可使用
`scripts/build_release_validation_package.py`。工具會對正片原始影片建立 SHA-256、產生
兩位 reviewer 的空白事件表與獨立 adjudication 欄位，並把舊的 unreviewed seed 隔離在
`adjudication_only/`。指定的 normal 資料只會成為
`unreviewed_negative_candidate`；資料夾名稱不會被提升為 ground truth。三幀視覺 hash
只提供同攝影機人工分組的檢索建議，也永遠保持 unreviewed：

```bash
conda run -n rtdetr env PYTHONPATH=scripts python \
  scripts/build_release_validation_package.py \
  --ground-truth runs/grounding_truth \
  --positive-source-root /path/to/original/litter/clips \
  --negative-candidate-root /path/to/normal/candidates \
  --output artifacts/release_validation_YYYYMMDD
```

目前由唯一負責人完整看片並填寫 primary reviewer 表即可建立 reviewed event、route、
NULL 與 negative labels；secondary/adjudication 表只保留給未來可選的獨立稽核，不是現行
人力要求。攝影機分組仍須由該負責人確認後才能凍結 development/validation/test，且必須
以 camera group 切分，禁止同一攝影機洩漏到不同 split。

### Release 時間與距離成本優化研究

可透過 `StudyConfig` 啟用車輛 `C_BC` 的 `D/0.4`、`T_E/0.25` 成本，以及
release 回推的分段平方 prior（0.25 秒內不加罰，最多 1 秒）。新時間 prior
取代原本的 backward window 成本，保留短軌跡 prior、forward penalty 與 NULL。
Production 預設使用 D=0.4 與 observation-gap prior；新的 release-time 組合先作明確的
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

### 優化版反追蹤技術說明簡報（2026-09-15）

以 `08_27 專題進度回報.pptx` 的白底、細線、灰階卡片與右上角「反向追蹤」標籤為
版型參考，說明目前優化版（研究候選）的 Kalman/RTS、release hypotheses、
C_BA/C_AC/C_BC、Gaussian/Mahalanobis、Min-Cost Flow、NULL、provenance/lineage
與 pseudo-homography 邊界。簡報中的 48/58 是 frozen development replay，並明確
標示尚未通過 85% 與獨立 holdout gate；不代表 production accuracy。

可用下列命令重建 13 頁簡報（需要系統的 LibreOffice 與 ffmpeg）：

```bash
/usr/bin/python3 artifacts/optimized_backtrack_presentation_20260915/build_deck.py
```

輸出為 `artifacts/optimized_backtrack_presentation_20260915/優化版反追蹤算法_技術說明_2026-09-15.pptx`；
同目錄的 `rendered_svg/`、`rendered_png/` 與 `preview_v2/` 僅供版面檢查，沒有任何
production caller。

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

### Backtrack 候選空間稽核（歷史研究）

研究成本函數前，曾以已封存的 `analyze_route_oracle_gap.py` 確認正確 actor tuple 是否
存在於同一 confirmed event 的 valid routes。該研究沿用 product-readiness 的
fixed-denominator、same-run actor mapping 與 strict/moderate event-match 規則；結果與
限制保留在版本紀錄，工具不再位於 active `scripts/`。

輸出的 `frozen_match_candidate_oracle_correct` 只回答「凍結目前 event match 後，重排既有
valid routes 的理論上限」，不是可部署準確率，也不能用來選取真實路徑。若此上限仍低於
目標，繼續調整 route cost 無法達標，必須先改善 actor mapping、candidate generation 或
event confirmation。

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
