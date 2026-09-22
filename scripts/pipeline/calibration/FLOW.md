# Online Self-Calibrating Pseudo-Homography：完整流程與驗收

本文件是 Phase 1–8 完成後的整體流程。這套模組從長期動態車流建立統計幾何約束，
不把相鄰時間的移動車輛位置當成相同 world point correspondence。

## 1. 完整資料流

```text
YOLO-Seg vehicle/scooter mask（既有推論）
  -> mask bottom 98-percentile band median ground point
  -> confidence / area / border / continuity / teleport quality filter
  -> bounded rolling per-camera trajectory metadata
  -> local motion vectors and tangents（保留彎道）
  -> robust spatial motion field
  -> position + direction + track-continuity flow clustering
  -> normalized relative pseudo-ground H0
  -> candidate-H hard geometric validation
  -> motion / speed / curvature / direction / lane robust losses
  -> bounded theta-space candidate search
  -> perspective + previous-H temporal regularization
  -> evidence confidence and independent hard gates
  -> confidence-gated small update + validation + objective re-check
  -> COLLECTING / ESTIMATING / LOW_CONFIDENCE / WARMING_UP / LOCKED state
  -> persistent drift -> DRIFT_DETECTED -> RECALIBRATING
  -> optional static-background G_current_to_reference
  -> H_runtime = H_calibration @ G_current_to_reference
  -> immutable event snapshot API（production attribution 尚未啟用）
```

## 2. 主要類別與函式

| 責任 | API |
|---|---|
| Mask 接地點 | `extract_vehicle_ground_point` |
| 輕量軌跡 | `CalibrationBuffer`, `VehicleTrajectoryCollector` |
| Motion field | `TrafficMotionField`, `robust_direction_summary` |
| Flow clustering | `cluster_traffic_flows` |
| 投影與安全 | `image_to_ground`, `validate_homography` |
| Robust objective | `evaluate_calibration_losses` |
| Bounded search | `estimate_candidate_homography` |
| Persistent state | `DynamicHomographyCalibrator` |
| Drift | `detect_calibration_drift` |
| Static stabilization | `StaticBackgroundStabilizer`, `build_dynamic_exclusion_mask` |
| Runtime composition | `compose_runtime_homography` |
| Event freeze | `capture_event_snapshot` |
| Debug | `render_calibration_debug` |

## 3. Homography 與最佳化數學

影像點 `p=[u,v,1]^T` 投影為：

```text
q = H p
(X,Y) = (q0/q2, q1/q2)
```

初始矩陣 `H0=diag(1/W,1/H,1)` 只建立 `[0,1]x[0,1]` 相對座標，不代表公尺。
候選以 `H_candidate=P(theta)H_previous` 產生，theta 依序為 log-anisotropy、x/y
shear、x/y perspective。平移與旋轉未納入搜尋，因為目前 traffic loss 無法識別它們。

```text
L_data = weighted_mean(L_motion, L_speed, L_curvature, L_direction, L_lane)

L_opt = weighted_mean(
    w_data * L_data,
    w_perspective * L_perspective,
    w_temporal * L_temporal
)
```

搜尋只做有界、逐輪衰減的 coordinate perturbation。每個 candidate 與最後 small-step
proposal 都要通過 hard validation；proposal objective 必須比 baseline 小。

## 4. 各 loss 的用途

| Loss | Dimensionless residual | 防止的錯誤 |
|---|---|---|
| Motion | `norm(v_i-v_(i-1))/median_track_speed` | ground-plane velocity 突跳 |
| Speed | `abs(log((speed_i+eps)/(speed_(i-1)+eps)))` | 短時間速度比例爆炸 |
| Curvature | consecutive wrapped turning-angle difference | 折線式曲率不連續；不強迫彎道變直 |
| Direction | flow cluster 內 projected local tangent angular residual | 局部方向發散 |
| Lane | leave-one-track-out centerline perpendicular distance / footprint scale | 跨車輛局部 curve-family 發散 |
| Perspective | local-Jacobian anisotropy and area variation | 極端 shear/stretch/collapse |
| Temporal | previous/candidate image-grid projection displacement | 固定攝影機 H 劇烈跳動 |

以上使用 quality weighting 與 Huber penalty。Huber 讓 lane change、tracking noise 和
局部離群點的 influence 有界，不表示它們被刪除。

## 5. Confidence

每次 Phase-6 評估輸出十個 `[0,1]` component：tracks、flows、coverage、duration、
quality、field stability、residual、lane、improvement、historical stability。Instant
confidence 是它們的平均；persistent state 再以 EMA 緩慢更新，drift window 則衰減。

Confidence 不是 accuracy probability。它是「目前 evidence 是否足以更新 H」的控制量。

## 6. Update 與 freeze

以下 hard gates 必須同時通過：minimum valid tracks、independent flows、spatial coverage、
median track duration、relative objective improvement、freeze confidence，以及 candidate H
和 proposed H validation。通過時：

```text
alpha = max_update_alpha * confidence * improvement_score
H_proposed = P(alpha * theta_best) @ H_previous
```

其餘情況全部 freeze。成功更新前保存 rollback snapshot。

## 7. State machine

```text
UNCALIBRATED
  -> first accepted observation -> COLLECTING
  -> evaluation window -> ESTIMATING
  -> weak evidence -> LOW_CONFIDENCE -> keep collecting
  -> validated proposal -> WARMING_UP
  -> enough updates + stable windows -> LOCKED
  -> persistent residual/geometry/background shift -> DRIFT_DETECTED
  -> next evaluation -> RECALIBRATING
  -> validated proposals -> WARMING_UP -> LOCKED
```

LOCKED 只監控、不持續最佳化。單一異常不改狀態。影像尺寸改變會 freeze 並要求 reset，
避免把不同 coordinate systems 的 observation 混入同一 buffer。

## 8. Static background stabilization

此功能預設關閉。呼叫端必須先以 vehicle、person、litter mask/polygon/box 建立 exclusion
mask。ORB 只在剩餘背景取 feature；ratio test 與 RANSAC 求 current-to-reference partial
affine G。低 inlier 或過大 translation/rotation/scale 不更新 G。這與長期 traffic H 分工：

```text
fast shake: current frame --G--> reference frame
long-term geometry: reference frame --H--> pseudo-ground
```

## 9. Runtime 與記憶體

- 每 frame：重用既有 YOLO-Seg/tracker 輸出，只存數值 metadata。
- 每 evaluation window：建立 motion field、flow clusters，評估少量 3x3 candidate。
- Motion/flow/lane 鄰居使用 grid index；典型為 `O(N*k)`，不是全域 all-pairs。
- 預設 3 輪、5 維搜尋最多 `1+3*2*5=31` 個 candidate。
- 每 track、track 數、rollback、metrics 均有 hard cap。
- Stabilizer 只保存一組 reference keypoints/descriptors 與 3x3 G，不保存 frame history。
- 不新增任何 deep-learning inference 或 central video upload。

## 10. Debug visualization

`render_calibration_debug(frame, tracks, observations, field, clusters, H, state)` 回傳 copy，
包含 image ground points/tracks、motion arrows、flow colors、pseudo-ground inset，以及 status、
confidence、tracks、coverage、loss、version、alpha。它不修改輸入，也不進 production renderer。

## 11. Metrics

每個 evaluation window 保存 timestamp、status、confidence、track/motion/flow counts、coverage、
五個 data loss、perspective/temporal loss、total、candidate improvement、alpha 與 H version。
另記錄 normalized-H Frobenius update magnitude，供 debug 顯示收斂步幅。歷史長度有上限，
可直接匯出做跨 camera/event 的統計分析。

## 12. Config 分類

`.env.example` 提供 observation quality、buffer、motion grid、flow clustering、matrix safety、
五種 data loss、lane peers、optimizer steps/weights/gates、confidence/state/drift、history 與
background stabilization 參數。所有數值都是顯式 initial engineering priors；沒有 reviewed
multi-camera validation 前，不得稱為最佳常數。

## 13. Synthetic / integration 驗收

測試涵蓋 robust mask point、segmentation jitter、teleport/ID-switch proxy、彎曲 local tangent、
反向與分區 flow、稀疏 evidence、safe projection、motion/speed/curvature/direction/lane loss、
尺度不變性、bounded optimization、低 coverage freeze、confidence accumulation、LOCKED、
persistent drift、rollback、resolution change、static ORB translation、exclusion mask、immutable
snapshot、debug copy，以及 GlobalLitterTracker 的 opt-in integration。

## 14. 已知限制與 promotion boundary

- Relative pseudo-ground 沒有 absolute metric scale；`meters_per_unit` 僅保留接口。
- 沒有 lane label 時，flow cluster 可能包含相鄰實體 lanes；lane loss 不可單獨決定 H。
- Traffic-only loss無法可靠識別全域平移/旋轉，因此由 optional static background G 處理。
- Vanishing-point loss 尚未啟用：彎道與交叉口不能被強迫共享 VP；未先建立可靠的
  straight-local-segment RANSAC evidence 前，不加入這項約束。
- ORB 在低紋理、夜間、雨霧下可能無 evidence；此時 freeze，不猜 transform。
- 現行 thresholds/weights 尚未以 reviewed multi-camera data 校正。
- `DYNAMIC_HOMOGRAPHY=0`、`DYNAMIC_HOMOGRAPHY_OPTIMIZE=0`、
  `DYNAMIC_HOMOGRAPHY_STABILIZE=0` 仍是安全預設。
- Event snapshot 已可由 `SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=1` 接入 spatial costs，
  但預設關閉，且僅接受 `LOCKED`、達最低信心並通過矩陣驗證的 H；其餘事件逐一回退
  image-space。58 部短片只驗證 fallback/no-regression，尚未證明真實 H 的準確率收益；
  在 continuous multi-camera paired replay 通過前不得預設啟用。

## 禁止的舊邏輯

本模組沒有也不得加入：

```text
vehicle_position(t-1) <-> vehicle_position(t)
-> treat as the same world point
-> directly solve Homography
```

動態車輛只能提供長期 trajectory statistics、local tangent、flow 與 curve-family constraint。
