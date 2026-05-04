# trashProject / scripts-old-test

本專案目前的主要即時流程位於 `scripts-old-test/`，負責把人/車追蹤、垃圾偵測、垃圾軌跡確認、丟擲者反追蹤、STGCN 動作辨識、車牌辨識與輸出影片整合成單一 pipeline。

## 執行入口

```bash
conda run -n rtdetr python scripts-old-test/main.py
```

常用參數：

- `--bbox-model`: 指定 YOLO-seg actor 模型，負責 `person / scooter / vehicle`。
- `--trash-model`: 指定 RTDETR litter 模型，負責 `litter`。
- `--bbox-conf` / `--trash-conf`: 分別控制 actor 與 litter 偵測信心門檻。
- `--yolo-seg-frame-skip`: 每 N 幀執行一次 YOLO actor tracking，其餘幀沿用快取以加速。
- `--disable-speed-filter`: 關閉車輛相對速度過濾。
- `--enable-action` / `--disable-action`: 控制 STGCN 動作辨識是否啟用。
- `--profile`: 輸出分段耗時與瓶頸排序。

## 整體架構

`main.py`

- 解析 CLI 參數與影片輸入/輸出設定。
- 優先載入 TensorRT `.engine`，若不存在則回退 `.pt`。
- 建立 YOLO actor model、RTDETR litter model、背景減除器、`GlobalLitterTracker`、STGCN action module 與 profiler。
- 逐幀呼叫 `detect.detect(...)`，寫入暫存 AVI，最後壓縮輸出影片並列印耗時摘要。

`detect.py`

- 使用 YOLO-seg tracking 取得 `person / scooter / vehicle`，並保留 actor bbox、track id、class 與 mask polygon。
- 以 person bbox 和 vehicle bbox IoU 建立 `person_vehicle_map`，讓違規 person 可同步標記其乘坐車輛。
- 選擇性呼叫 STGCN 動作辨識，取得每個 person 的 `normal / littering` 分數。
- 呼叫車牌辨識流程，並更新 `vehicle_history`。
- 使用 RTDETR 在全圖偵測 litter，先經過 motion filter 與 holding filter，避免舊垃圾或仍被持有的物件進入 tracker。
- 呼叫 `GlobalLitterTracker.update(...)` 做垃圾狀態更新與 thrower 反追蹤，再把 active violator 畫成紅框。

`litterTracker.py`

- 維護 active litter 狀態：`pending`、`confirmed`、missed frame、bbox、history、shape reference。
- 使用軌跡物理條件確認垃圾：足夠觀測幀數、明顯位移、Y 軸向下位移與 `validate_trajectory(...)`。
- 在垃圾出生與 pending 更新階段重新計算最可能 thrower。
- confirmed 後將 thrower 加入 violator TTL cache，並支援同類別近距離 rebind，降低 tracker ID 短暫切換造成的漏標。
- 若 thrower 是 person，且 `person_vehicle_map` 曾對應到車輛，會同步標記對應 vehicle/scooter，讓畫面可標紅框並觸發車牌顯示。

`smallFunction.py`

- 提供 motion check、IoU matrix、mask overlap、holding 判斷與軌跡驗證。
- holding 會同時考慮 actor bbox/mask、litter 移動方向、車輛移動方向與相對速度。

`action.py`

- 封裝 STGCN++ 動作辨識流程。
- 目前違規條件是 action 為 `littering` 且 STGCN 分數達到 `action_threshold`，預設門檻為 `0.35`。

`licensePlate.py`

- 對 vehicle/scooter ROI 做車牌偵測與 OCR。
- 結果寫入 `vehicle_history[track_id]['license_plate']`，供 render 階段顯示。

`timeUtils.py`

- 提供 `PipelineProfiler` 與 `profile_block(...)`。
- 最終輸出模型載入、影像處理、影片寫入、總影片處理時間與瓶頸 Top 3。

`export_tensorrt.py`

- 匯出 YOLO actor model 與 RTDETR litter model 的 TensorRT engine。
- `main.py` 會優先使用同名 `.engine`，不存在時才使用 `.pt`。

## 逐幀流程

1. `main.py` 讀取影片 frame。
2. `detect.py` 執行或沿用 YOLO actor tracking 快取，取得 persons 與 vehicles。
3. 建立 person-to-vehicle 對應與 vehicle history。
4. 視設定執行 STGCN action module。
5. 執行 RTDETR litter detection。
6. 對 litter 做 motion filter 與 holding filter。
7. `GlobalLitterTracker` 更新 litter 軌跡與 confirmed 狀態。
8. tracker 依 homography pseudo-ground 座標反推最可能 thrower。
9. confirmed litter 會把 thrower 與可能對應車輛加入 violator cache。
10. `detect.py` 渲染 actor、violator、confirmed litter 與 STGCN/車牌資訊。
11. `main.py` 寫入暫存影片、壓縮輸出並清理暫存檔。

## Homography Thrower 反追蹤

舊邏輯只用 2D 影像距離在 litter 出生幀選最近 actor。現在 `litterTracker.py` 改成以下流程：

1. 從目前畫面所有 `vehicle / scooter` bbox 取得最底部中心點，這些點視為地面接觸點。
2. 根據車輛底部點的 X/Y 分布、bbox 高度與 litter 底部點，估計一個 3x3 homography matrix。
3. 將 litter bottom-center、actor bottom-center 投影到 pseudo-ground 座標。
4. 以投影後距離除以投影後 actor 寬度門檻計算 score。
5. 若同一個 thrower key 已在上一幀被選中，score 會有連續性加權，降低短暫抖動。
6. 若 litter 已有兩個以上 history points，會用 pseudo-ground 軌跡方向微調 score，偏好「垃圾從 actor 方向離開」的候選。
7. 若沒有 vehicle/scooter 可估計 homography，會退回等比例 identity transform，保留原本 2D 幾何關聯能力。

這個設計不需要額外相機校正檔，但它不是精準測量真實 3D 座標；它的目的，是在多車、多透視深度的畫面中，比單純 pixel distance 更穩定地選出最可能丟擲者。
