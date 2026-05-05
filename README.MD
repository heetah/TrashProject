# trashProject / scripts-old-test

`scripts-old-test/` 是目前主要影片推理流程，負責整合 YOLO actor tracking、RTDETR litter detection、litter 軌跡確認、丟擲者反追蹤、STGCN 動作辨識、車牌 OCR、影片輸出與耗時摘要。

## 執行入口

```bash
conda run -n rtdetr python scripts-old-test/main.py TThrow.mp4 --batch 1
conda run -n rtdetr python scripts-old-test/main.py TThrow.mp4 --batch 2
```

影片檔預設從 `resources/` 讀取，輸出到 `output/{影片名}_annotated.mp4`。

## Batch 模式與模型路徑

`--batch 1` 使用一般單幀模式：

- YOLO actor model: `modules_weight/best-yolo-seg_v3.pt`
- RTDETR litter model: `modules_weight/best-rtdetr-seg.pt`

`--batch 2` 使用 batch-2 模式：

- YOLO actor model: `modules_weight/batch/best-yolo-seg_v3.pt`
- RTDETR litter model: `modules_weight/batch/best-rtdetr-seg.pt`

若同位置存在 TensorRT engine，`main.py` 會優先載入：

- batch 1: `*.engine`
- batch 2: `*_b2.engine`

可用 `--no-engine` 強制使用 `.pt`。也可用 `--bbox-model` / `--trash-model` 手動覆蓋預設路徑。

## 常用參數

- `--batch {1,2}`: 選擇單幀或 batch-2 推理。
- `--bbox-conf`: YOLO actor 信心門檻，預設 `0.45`。
- `--trash-conf`: RTDETR litter 信心門檻，預設 `0.4`。
- `--yolo-seg-frame-skip`: 每 N 幀跑一次 YOLO actor tracking，其餘幀沿用快取。
- `--disable-action`: 關閉 STGCN 動作辨識。
- `--action-threshold`: STGCN 判定 `littering` 的分數門檻，預設 `0.5`。
- `--disable-plate`: 關閉車牌偵測與 OCR。
- `--skip-plate-preload`: 不在影片開始前預載車牌模型。
- `--disable-speed-filter`: 關閉車輛速度相關過濾。

## 檔案責任

`main.py`

- 解析 CLI 參數。
- 依 `--batch` 自動選擇 batch 1 或 batch 2 model path。
- 優先嘗試 TensorRT engine，失敗或不存在時回退 `.pt`。
- 預載並 warmup YOLO、RTDETR、STGCN、車牌模型。
- 讀取影片、建立前景遮罩、呼叫 `detect()` 或 `detect_batch()`。
- 寫入暫存 AVI，最後用 ffmpeg 壓成 MP4。
- 在暫存檔清理後輸出分段耗時摘要。

`detect.py`

- 管理單幀與批次偵測流程。
- 從 YOLO tracking 解析 `person / scooter / vehicle`。
- 建立 person-to-vehicle 關聯。
- 執行 RTDETR litter detection。
- 對 litter 做 motion filter 與 holding filter。
- 呼叫 `GlobalLitterTracker` 更新 pending/confirmed 狀態。
- 渲染 actor、violator、confirmed litter、STGCN 分數與車牌。

`litterTracker.py`

- 維護 active litter 軌跡、missed frame、shape reference 與 `pending / confirmed` 狀態。
- 使用物理軌跡與向下位移確認 litter。
- 以 pseudo-ground homography 估計最可能 thrower。
- confirmed 後標記 person / vehicle / scooter 違規者，並用 TTL 維持畫面紅框。

`smallFunction.py`

- 提供 motion check、IoU、polygon overlap、holding 判斷與 `validate_trajectory()`。
- `litter_holding()` 是 polygon-aware gate，負責判斷 litter 是否仍貼在人車上。

`action.py`

- 封裝 YOLO pose + STGCN++。
- 累積每個 person track 的骨架序列。
- 當動作為 `littering` 且分數達 `action_threshold` 時，回傳 alert。

`licensePlate.py`

- 只對已鎖定違規車輛派工。
- 背景 thread 執行車牌 YOLO 與 PaddleOCR。
- OCR 結果寫回 `vehicle_history`，供畫面顯示。

`timeUtils.py`

- 提供 `PipelineProfiler` 與 `profile_block()`。
- 支援中文欄位寬度對齊。
- 最後輸出模型載入、影片處理、影像寫入與瓶頸 Top 3。

`export_tensorrt.py`

- 匯出 YOLO actor 與 RTDETR litter 的 TensorRT engine。
- 預設支援 batch 2，輸出檔名會是 `*_b2.engine`。

## 逐幀主流程

1. `main.py` 讀取 frame，並用 MOG2 背景減除器產生前景遮罩。
2. `detect.py` 執行 YOLO actor tracking，或依 `--yolo-seg-frame-skip` 沿用 actor 快取。
3. 解析 actor bbox、track id、class name 與 segmentation polygon。
4. 建立 person 與 vehicle/scooter 的 IoU 對應。
5. 選擇性執行 STGCN 動作辨識。
6. RTDETR 對全圖偵測 litter。
7. litter 候選先通過 motion filter 與 holding filter。
8. `GlobalLitterTracker` 更新 litter 軌跡與 confirmed 狀態。
9. confirmed 後反推 thrower，標記 person / vehicle / scooter 違規者。
10. 車牌流程只對違規車輛派工。
11. 渲染紅框、confirmed litter、STGCN 分數與車牌。

## Litter 確認流程

### 前處理

RTDETR 只負責提出 litter bbox 候選，不能直接當 confirmed。候選進 tracker 前會先被 `detect.py` 過濾：

1. 只接受 class name 為 `litter` 的結果。
2. 過濾極端長寬比與過小 bbox，避免明顯雜訊進入 tracker。
3. 使用前景遮罩檢查 litter bbox 內的 moving pixel ratio。
4. 對較大的 litter bbox 再檢查核心區域 motion，避免旁邊人車移動誤觸發舊垃圾。
5. 從既有 tracker 找最近上一幀 litter center。
6. 若是新出現 litter，先放入 tracker 建立 pending history anchor。
7. 若已有上一幀中心，呼叫 `litter_holding()` 判斷是否仍被人、車或機車持有。

### Holding Gate

`litter_holding()` 會用 actor bbox、segmentation polygon、litter 位移、車輛位移與相對速度判斷是否仍是 holding：

1. litter anchor 在 actor polygon 內，通常視為仍被持有。
2. litter 貼近 vehicle/scooter mask 底部或側邊時，若沒有明確 release motion，視為 holding。
3. vehicle release 需要向下位移、水平位移、相對車體分離或明確絕對位移。
4. 靜止 litter 加上車輛仍在移動時，不視為 holding，避免地上垃圾被移動車輛黏住。
5. 通過 holding gate 後，litter 才會交給 tracker。

### Tracker 確認

`GlobalLitterTracker.update()` 會把通過前處理的 litter 串成軌跡：

1. 以中心距離與尺寸一致性配對既有 active litter。
2. 新 litter 建立 `pending` 狀態，記錄 bbox、history、shape 與出生幀 thrower。
3. pending 更新時累積中心點 history，並重新評估最可能 thrower。
4. confirmed 需要足夠觀測幀數與向下位移。
5. confirmed 可由兩條路成立：
   - `validate_trajectory()` 通過：軌跡平滑、總位移足夠、Y 軸往下。
   - motion route 通過：位移明顯、Y 軸往下、且能關聯到 thrower。
6. confirmed 後才會在畫面畫出紫色 `Litter {id} (confirmed)`。

### 後處理

confirmed 後會進入違規者後處理：

1. tracker 用 pseudo-ground homography 估計 litter 與 actor 的地面距離。
2. 選出最可能 thrower，寫入 violator TTL cache。
3. 若 thrower 是 person，且曾與 vehicle/scooter 有 IoU 關聯，會同步標記該車輛。
4. 違規者用紅框顯示，TTL 期間可容忍短暫 miss。
5. 若 tracking ID 短暫切換，會以同類別近距離 rebind 延續違規標記。
6. 車牌辨識只對已鎖定違規車輛執行，避免每幀掃描所有車輛。
