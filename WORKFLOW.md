# scripts-old-test Workflow

本文件依「實際程式碼」梳理 `scripts-old-test/` 目前架構、執行流程，並完整整理 **litter 前處理 / 後處理篩除鏈**，最後列出**潛在漏洞與可優化、可縮減項目**。所有條目皆標注對應檔案與行號，供直接定位。

> 實作目標不變：`scripts-old-test/` 只用 YOLO-Pose + STGCN 判斷 `normal` / `urinate`，亂丟垃圾只由 litter object-event branch 確認。RTMW 不在 production keypoint path。

---

## 1. 核心合約

| 項目 | 內容 | 來源 |
|------|------|------|
| Entry | `scripts-old-test/main.py` | — |
| Runtime env | `conda run -n rtdetr ...` | — |
| Actor detection | YOLO-Seg，解析 `person` / `vehicle` / `scooter` | `detect.py:48` |
| Pose source | YOLO-Pose only（`yolo26x-pose.pt`） | `main.py:34` |
| STGCN classes | `{0: "normal", 1: "urinate"}` | `action.py:14` |
| Litter detector | RT-DETR 4-channel（`best-rtdetr-4c.pt`），輸出只當候選 | `main.py:39` |
| Litter confirm | `GlobalLitterTracker` 依 motion / temporal / holding evidence | `litterTracker.py:79` |
| OCR | 只在 confirmed violator vehicle/scooter 上啟動 | `licensePlate.py:179` |
| RTMW | legacy，未進入此 pipeline 任何路徑 | — |

---

## 2. 檔案職責對照

| 檔案 | 行數 | 職責 |
|------|------|------|
| `main.py` | 1486 | 模型載入/warmup、影片 I/O、async reader、FFmpeg writer、batch 主迴圈、統計 summary |
| `detect.py` | 1245 | 單幀 / 批次偵測核心：actor 解析、4ch litter 偵測、**前處理篩除**、tracker 串接、渲染 |
| `litterTracker.py` | 1707 | `GlobalLitterTracker`：軌跡配對、**後處理確認**、thrower 反推、backward worker、違規者狀態 |
| `smallFunction.py` | 830 | 幾何工具：`motion_evidence`、IoU/IoM、mask overlap、`litter_holding`、`validate_trajectory` |
| `action.py` | 533 | `STGCNActionModule`：pose 擷取、骨架序列、STGCN 推理、urinate 時間確認 |
| `licensePlate.py` | 298 | 車牌偵測 + PaddleOCR 背景工作、`until_plate_found` 持續搜尋 |
| `timeUtils.py` | 348 | `PipelineProfiler` 分組計時與瓶頸排行 |

---

## 3. 高階流程

```text
Input video
  -> 模型 preload + warmup（STGCN→pose→YOLO-seg→RTDETR→OCR）
  -> VideoCapture（FFmpeg, 可硬解）
  -> AsyncVideoFrameReader 背景執行緒
        └ 解碼 + 建 fg_mask（temporal diff，預設 0.5x）-> queue
  -> 主迴圈每次取 --batch（預設 8）幀
     -> detect_batch
         ├ actor：YOLO-Seg（predict/track），可 yolo_seg_frame_skip 跳幀沿用快取
         │    ├ person branch -> STGCNActionModule.update -> normal/urinate
         │    └ vehicle/scooter branch -> person_vehicle_map（IoM>0.7）
         ├ litter：RT-DETR 4ch 批次推理（BGR + change_map）
         │    └ 前處理篩除（幾何 -> 全框 motion -> 核心 motion -> holding）
         │    └ GlobalLitterTracker.update（後處理確認 pending->confirmed）
         │         └ confirm 後標記 thrower + 連動 vehicle + 派 backward worker
         ├ OCR：只對 active violator + backward ROI 派工
         └ render：confirmed litter 紫框、violator 紅框、person STGCN 標籤
  -> AsyncFFmpegVideoWriter（NVENC 優先，否則 libx264）
  -> summary log + 可選 --summary-json
  -> profiler 分組計時
```

執行緒共 4 條：主推論、`AsyncVideoFrameReader`（讀檔+fg_mask）、`AsyncFFmpegVideoWriter`（編碼）、`litter-backward-resolver`（thrower 反推幾何評分）。

---

## 4. 模型載入與 fallback

- 預設權重依 `--batch` 自動切換（`main.py:855` `_default_model_paths_for_batch`）：batch>1 用 `modules_weight/batch/` 版。
- Engine 優先策略（`main.py:225` `_model_path_candidates_for_batches`）：`.pt` 先找同 batch 的 `*_bN.engine`，缺 CUDA/TensorRT 自動退回 `.pt`。`--no-engine` 強制 `.pt`。
- Actor batch 自動估算：`ceil(batch / yolo_seg_frame_skip)` 再 round 到 `{1,2,4,8}`（`main.py:213`）。
- RT-DETR 為 **4-channel** 模型：warmup 前先讀 engine binding shape 或 `.pt` 的 `yaml["channels"]` 來決定 dummy frame 通道（`main.py:781`、`main.py:1107`）。channel/imgsz 不符會在 batch predict 直接 raise（`detect.py:951`）。
- 每個模型載入後立即 warmup，避免首幀吃初始化成本（`main.py:263`）。

---

## 5. 預設「極速路徑」

未加 `--extreme-speed-off` 時，`main.py:1005` 會強制：

```python
args.actor_mode = "predict"        # 用 YOLO.predict + 輕量 IoU id，跳過 BoT-SORT per-frame 成本
args.rtdetr_zero_repair = "off"    # 不修補 batch 中掉成 0-box 的 frame
```

- `predict` 模式 id 由 `_assign_fast_actor_track_ids` 以前一幀 bbox IoU（`--actor-track-iou` 預設 0.3）指派（`detect.py:270`）。
- `track` 模式保留 BoT-SORT id，較穩定但較慢。

---

## 6. Person Action 分支（STGCN）

`action.py` `STGCNActionModule` 擁有所有 STGCN 狀態。

**流程**（`action.py:378` `update`）：
1. 對整張 frame 跑 YOLO-Pose（`conf=0.3`），取得 pose boxes。
2. 用 IoU（`track_iou_threshold=0.2`）把 pose 偵測配對回主流程 tracked person。
3. 配到才取 17 點骨架 append 進 `track_history[track_id]`（`deque(maxlen=window_size)`，預設 30）。
4. 序列滿窗（`== window_size`）且 `frame_index % predict_interval == 0` 才跑 STGCN 推理（`action.py:473`）。
5. urinate 時間確認（`_record_urination_evidence`，`action.py:347`）：最近 `urination_window_sec` 內 positive 累積需 `>= urination_min_sec` 才 alert。
6. 與 vehicle 關聯的 person（`blocked_urination_track_ids`，來自 `person_vehicle_map`）會清除並抑制 urinate（`action.py:453`、`489`）。

**輸出**（`action.py:512`）：
```python
{track_id: {"action": "normal"|"urinate", "raw_action", "conf", "stgcn_conf",
            "alert": bool, "urination_evidence_sec", ...}}
```

**門檻現況（程式碼預設）**：
```bash
--urination-window-sec 10.0
--urination-min-sec     8.0
--action-window         30
--action-threshold      0.5
```
> 注意：`CLAUDE.md` 提到「preferred 8s 窗 / 6s positive」，但程式碼預設仍是 10s / 8s。依規範**保留現有行為、不靜默更改**，需更動請先確認。

STGCN 不可用（缺權重/載入失敗）時，`update` 仍回傳全 `normal`，主流程不需分支（`action.py:400`）。

---

## 7. Litter 前處理篩除鏈（detect.py，進 tracker 前）

litter 候選必須**依序通過以下每一關**才會進入 `GlobalLitterTracker`。任一關失敗即丟棄。

### 7.1 偵測階段
- RT-DETR 4ch 推理：輸入 `dstack(BGR, change_map)`，`change_map = compute_pixel_change_map(prev_frame, frame)`（`detect.py:20`，放大 2x 不做 normalize 以保留動態幅度）。
- `conf >= --trash-conf`（預設 0.4）。
- class name 必須是 `litter`（`detect.py:504`）。

### 7.2 幾何篩除（`detect.py:509`）
丟棄：`aspect_ratio > 6.0` 或 `< 0.15` 或 `width < 3` 或 `height < 3`（極端扁長/過小雜訊）。
→ 通過者計入 `raw_litter_candidates`。

### 7.3 全框 motion evidence（`detect.py:528`）
`motion_evidence(fg_mask, 全框, threshold=moving_threshold=0.25, min_component_area=4, min_largest_component_ratio=0.25)`：
- 前景像素比例需 `>= 0.25`；
- 連通元件需集中（最大元件 / 白點數 `>= 0.25`），濾掉細碎感測器噪聲（`smallFunction.py:96`）。
- 不動 → 丟棄（排除靜止舊垃圾）。

### 7.4 核心區 motion evidence（`detect.py:540`）
只在 bbox `>= 8x8` 時，對中心 60% 區再驗一次（`threshold=core_moving_threshold=0.3`）。
→ 抑制「旁邊人車移動、舊垃圾被連帶判定為動」的誤觸發。

### 7.5 Holding 篩除（`detect.py:557`）
1. 從 `litter_tracker.active_litters` 找距離 `< distance_threshold(250)` 且最近的前一幀中心，取其 `prev_center` / `missed` / `history`。
2. **新出生**（`prev_litter_center is None`）：直接放行建立 anchor（第二幀起才判斷相對分離，避免剛丟出的第一點被擋）。
3. 否則呼叫 `litter_holding(...)`（`smallFunction.py:407`），判定為持有中 → 丟棄。
→ 通過者計入 `filtered_litter_candidates`，送入 tracker。

### 7.6 `litter_holding` 行為判定（`smallFunction.py:407`）
對每個 actor（person/vehicle/scooter）採不同 holding 與 release gate：

- **共用速度特徵**：`litter_vx/vy`（相對前幀），`abs < 3.5` 視為靜止；`_motion_points` 重建軌跡支援「先拋後落」拋物線（`smallFunction.py:360`、`380`）。
- **Vehicle/scooter**（`smallFunction.py:451`）：
  - 解綁前置條件：litter 靜止在地 + 車仍在動 + 不在車 bbox 內 → 直接跳過（非持有）。
  - **release gate**（須有向下+水平+足夠位移才允許脫離 holding）：分 relative（相對車身）/ absolute（車不動時）/ arc（拋物線）三類；側邊釋放另有 `side` / `strong_side` gate，含 mask gap、overlap、low-edge ratio、anchor missed 上限（常數見 `smallFunction.py:26-59`）。
  - 上半部 attached（後照鏡/車頂物件，`norm_y <= 0.55`）且無 release → 判持有。
  - mask overlap `>= 0.08` 或 signed-distance 在膨脹範圍內 → 車身部件持有。
- **Person**（`smallFunction.py:527`、`766`）：anchor 為 bbox 上方略偏，距離門檻 `PERSON_DIST_THRESHOLD=55`；mask 內 / overlap `>= 0.20` / bbox 內 / 距離內 → 持有（手持寶特瓶、鞋、衣物誤判）。

> 行為驅動原則：噪聲修正應加強 motion/shape/component 證據，**不是單純降門檻**。

---

## 8. Litter 後處理確認鏈（GlobalLitterTracker.update，litterTracker.py:110）

候選進 tracker 後，從 `pending` 轉 `confirmed` 才算違規。

### 8.1 軌跡配對（`litterTracker.py:144`）
- 距離 `< distance_threshold(250)` 且形狀一致（pending 比例 `0.60`、confirmed `1.20`，`litterTracker.py:14`）配到既有軌跡。
- `confirmed` 軌跡距離 `< 0.7x` 門檻時可只靠距離續接，避免尺寸波動斷 id。
- `stationary_locked` 的 pending litter **不吸收**新偵測，避免靜止舊垃圾鎖住真正丟棄軌跡起點（`litterTracker.py:184`）。

### 8.2 thrower 重估（pending 每幀）
`_find_thrower_for_litter`（`litterTracker.py:1414`）：用車輛底邊估簡化 homography，把 litter 與 actor 投影到 pseudo-ground，以 `world_dist / threshold` 評分，含：前一 thrower bonus（0.85）、軌跡方向因子（2D）、release-origin（起點貼 actor、終點離開）加權，以及 fallback / release 多級 score limit。

### 8.3 Confirm 多路 gate（任一成立即 confirmed，`litterTracker.py:437`）
共同前提：`thrower_key is not None`、非 `is_static_candidate`、非 `stationary_locked`、`vehicle_relative_ok`、`is_step_velocity_ok`、`is_horiz_ratio_ok`。

| 路徑 | 關鍵條件 | 行號 |
|------|----------|------|
| `can_confirm_by_trajectory` | `age>=eff_min_age` + `validate_trajectory` 通過 + 向下+水平位移達標 | `335` |
| `can_confirm_by_motion` | 移動距離 `> max(2.5*邊長, 14)` + 向下+水平達標 | `347` |
| `can_confirm_vehicle_fast_drop` | vehicle thrower + `age==2` + 連續幀(gap≤2) + release-origin + 向下`>=35` + 水平/向下 ratio`>=0.15` | `404` |
| `can_confirm_fall_then_stable` | vehicle thrower + `age>=8` + 曾下落`>=25` + 末 4 幀穩定(jitter`<15`) | `373` |

**vehicle thrower 加嚴**（FP 多為車輛部件）：`MIN_CONFIRM_AGE_VEHICLE=3`、向下門檻 `12`、水平/向下 ratio 上限 `3.5`、單步位移上限 `200px`（`litterTracker.py:24-27`）。

### 8.4 靜止舊垃圾抑制
- `is_static_candidate`：`age>=3` 且歷史最大跨距 `< 10`（`litterTracker.py:249`）。
- `stationary_locked`：`age>=10` 且跨距 `< 8` → 永久標記，後續單次大抖動無法衝過 confirm（`litterTracker.py:256`）。

### 8.5 車身/貨物分離判別（`litterTracker.py:309`）
`_carrier_vehicle` 找重疊最高的載體車輛（mask overlap 或 bbox 內含取大），若 overlap `>= 0.15` 才檢查 `_litter_vehicle_separation`（扣除車輛自身位移後的淨位移）。淨位移 `< 60` → 視為隨車部件，`vehicle_relative_ok=False` 擋下。

### 8.6 軌跡 miss 容忍
本幀未配對的舊 litter `missed += 1`，`< MAX_MISSED_FRAMES(10)` 仍保留（`litterTracker.py:537`）。

### 8.7 confirmed 後動作
1. 標記 thrower 為 violator（action=`littering`），若 person 綁定車輛則連動標車（`litterTracker.py:449`）。
2. 派 `_submit_backward_resolution`（`litterTracker.py:668`）給 backward worker。

### 8.8 Backward thrower 反推（worker thread）
`_resolve_backward_task`（`litterTracker.py:735`）：在 `[birth-24, confirm+18]` 幀窗（`litterTracker.py:69-72`）內，用 homography pseudo-ground 對所有 actor 重新評分，挑最佳 thrower + 連動 vehicle + 蒐集車牌 ROI（每結果上限 3 個）。結果只在 main/update thread 套用（`_drain_backward_results`，`litterTracker.py:911`），避免 shared dict 競爭。車牌被遮時設 `plate_search_until_found` 持續搜尋。

### 8.9 違規者顯示連續性（`litterTracker.py:545`）
TTL 60 幀，同 key 跳動 `> MAX_VIOLATOR_JUMP(80)` 改走 rebind；同類別近距離（`< 60`）可繼承 id；`miss > VIOLATOR_MAX_MISSED(5)` 才釋放。

---

## 9. 渲染與 STGCN 旁路

- `register_action_violators`（`litterTracker.py:1131`）**只接受 `urinate`**；littering 一律由 litter 分支確認，符合合約。urinate 可連動標記歷史車輛並排車牌 backtrack。
- 渲染（`detect.py:723`）：violator 紅框 + `-{LABEL}-`；正常 person 顯示 `person {action} STGCN {conf}`；只畫 `state == 'confirmed'` 的 litter（紫框），pending 不畫（`detect.py:776`）。

---

## 10. CLI 參數速查

```bash
conda run -n rtdetr python scripts-old-test/main.py resources/resize.mp4 --batch 8
```

| 參數 | 預設 | 作用 |
|------|------|------|
| `--batch` | 8 | pipeline batch（1/2/4/8） |
| `--disable-action` | off | 關閉 YOLO-Pose + STGCN 分支 |
| `--disable-plate` | off | 關閉 OCR（litter-only 提速） |
| `--yolo-seg-frame-skip` | 2 | actor 每 N 幀才推理，其餘沿用快取 |
| `--actor-mode` | predict（極速路徑） | predict / track |
| `--trash-conf` / `--bbox-conf` | 0.4 / 0.45 | litter / actor 信心門檻 |
| `--action-window` | 30 | STGCN 序列窗 |
| `--urination-window-sec` / `--min-sec` | 10 / 8 | urinate 時間確認 |
| `--no-engine` | off | 強制 `.pt` |
| `--extreme-speed-off` | off | 關掉極速路徑（恢復 track + zero-repair=all） |
| `--debug-tracker` | off | 開啟 per-frame confirm debug |

---

## 11. Validation / Regression

```bash
# 編譯檢查
conda run -n rtdetr python -m py_compile \
  scripts-old-test/main.py scripts-old-test/detect.py \
  scripts-old-test/litterTracker.py scripts-old-test/action.py

# litter 確認回歸
conda run -n rtdetr python validate_old_test_videos.py --expect-positive resize.mp4 manyFast.mp4
```

回歸基線（每次改動後必跑）：

| 影片 | confirmed_ids | first_confirmed_frame |
|------|--------------|----------------------|
| resize.mp4 | 1 | fi=74 |
| manyFast.mp4 | 1 | fi=82 |
| litter_case_10.mp4（絕對路徑） | 1 | fi=150 |

STGCN/action：`urinate.mp4` / `normal_case1.mp4` / `normal_case2.mp4` / `best_urinate.mp4`。
- normal：keypoints 可有，但不得 alert。
- urinate：STGCN 預測 urinate 後須時間確認。
- litter clips：STGCN 不得輸出 `littering`，litter 警示只來自 confirmed litter event。

---

## 12. 潛在漏洞與可優化 / 可縮減

> 以下為「依現有程式碼觀察到」的事項，**尚未修改**；多為文件化與提案，動手前請依 CLAUDE.md 規範跑回歸基線。

### A. 正確性 / 穩健性風險

1. **STGCN 序列時間不連續**
   pose 配不到時不 append skeleton（`action.py:465`），序列由「非連續幀」的骨架拼成；滿窗判定只看長度 `==30`，不檢查時間跨度。長時間遮擋後拼出的序列可能讓 STGCN 判讀失真。建議：記錄 frame gap，過大時清窗或標記低信心。

2. **pose 擷取無跳幀，是已知主要成本**
   `ACTION_PREDICT_INTERVAL` 只降低 STGCN 推理頻率，**pose YOLO 仍每有 person 的幀都全圖跑一次**（`action.py:425`）。符合 CLAUDE perf note 所述「pose extraction can still dominate cost」。建議：加 pose 擷取 interval 或與 `yolo_seg_frame_skip` 對齊。

3. **兩套 frame-difference 重複計算**
   每幀同時算：(a) `MotionMaskBuilder` temporal fg_mask（reader thread，供 `motion_evidence`）與 (b) `compute_pixel_change_map`（detect，供 4ch RT-DETR）。兩者都是 grayscale absdiff。可評估共用一次灰階差分以省 per-frame CPU（注意 fg_mask 預設 0.5x scale、change_map 為全解析度且 ×2，需對齊尺度）。

4. **backward task queue 滿載靜默丟棄**
   `_backward_tasks`（maxsize 64）滿時 `put_nowait` 失敗即放棄該次反推（`litterTracker.py:708`），高密度 confirm 場景可能漏掉 thrower/車牌 backtrack。建議：滿載時記一筆 stat 以利觀測。

5. **長影片字典無上限成長**
   `person_to_vehicle_history`（`litterTracker.py:90`）與 main 的 `vehicle_history` defaultdict（`main.py:1224`，內層 centroids 有界但外層 entry 不回收）會隨 track_id 線性成長。短片無感，超長串流需加老化清理。

### B. 設計取捨（非 bug，但值得標注）

6. **新出生 litter 一律放行**（`detect.py:580`）：第一幀不做 holding 檢查，完全倚賴 tracker 後處理 gate。利於捕捉「剛離手」的第一點，但也讓人/車身上的誤判都先進 tracker，confirm gate 的嚴謹度是唯一防線。

7. **urinate 門檻文件不一致**：程式碼預設 10s/8s，`CLAUDE.md` 寫 preferred 8s/6s。需正式拍板，目前保留程式碼行為。

8. **極速路徑預設改寫 CLI**（`main.py:1005`）：未加 `--extreme-speed-off` 時，使用者就算顯式指定 `--actor-mode track`，仍會被強制改成 `predict`。屬「合約」但易讓除錯者困惑，建議在 log 明確提示被覆寫的值。

### C. 可優化（效能）

9. 依 CLAUDE perf 排序，首要瓶頸通常是 **pose 擷取 → RT-DETR → actor YOLO → STGCN → OCR → encode**。在不動責任分工前提下，最高槓桿是第 2 點（pose 跳幀）與第 3 點（共用 frame diff）。
10. `--disable-plate` 已能有效砍掉 OCR 背景成本；litter-only 量測時應預設帶上。
11. TensorRT engine 僅在 sibling `.engine` 存在且 runtime smoke 通過時採用，`--no-engine` 必須維持可用 fallback（現況符合）。
