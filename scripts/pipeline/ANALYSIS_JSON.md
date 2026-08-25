# Analysis JSON 輸出格式

- Schema version：`2.1.0`
- 產生位置：與 annotated MP4 相同資料夾
- 命名：`<影片名>_annotated_analysis.json`
- 數量：每支輸入影片一份，不跨影片累積

假設輸入為 `resize.mp4`：

```text
output/resize_annotated.mp4
output/resize_annotated_analysis.json
```

Production 不再另外產生 `summary.json` 或 `events.jsonl`。JSON 先寫入
`.tmp`，完成後 atomic replace，避免網頁讀到未完成內容。

## 完整範例

```json
{
  "schema_version": "2.1.0",
  "video": {
    "file": "resize_annotated.mp4",
    "duration_sec": 135.0
  },
  "litter_detection": {
    "rtdetr_4channel": {
      "enabled": true,
      "confidence_threshold": 0.4,
      "evaluated_frame_count": 4020,
      "vehicle_gate_skipped_frame_count": 30,
      "candidate_count": 3,
      "detected_frame_count": 2,
      "detected_frames": [
        {"frame_index": 870, "candidate_count": 1},
        {"frame_index": 871, "candidate_count": 2}
      ]
    },
    "geometry_passed": {
      "candidate_count": 2,
      "detected_frame_count": 2,
      "detected_frames": [
        {"frame_index": 870, "candidate_count": 1},
        {"frame_index": 871, "candidate_count": 1}
      ]
    },
    "motion_holding_passed": {
      "candidate_count": 2,
      "detected_frame_count": 2,
      "detected_frames": [
        {"frame_index": 870, "candidate_count": 1},
        {"frame_index": 871, "candidate_count": 1}
      ]
    },
    "confirmed_event_count": 1
  },
  "summary": {
    "litter_event_count": 1,
    "urinate_event_count": 0,
    "passed_vehicle_count": 4,
    "average_litter_confidence": 0.88,
    "detection_accuracy": null,
    "accuracy_status": "not_evaluated",
    "littering_plates": ["ABC1234"],
    "review_required": true
  },
  "events": [
    {
      "type": "litter",
      "id": 1,
      "start_sec": 29.0,
      "end_sec": 30.0,
      "confidence": 0.88,
      "vehicle": "vehicle:7",
      "plate": "ABC1234",
      "plate_confidence": 0.91,
      "plate_status": "recognized",
      "attribution_status": "resolved",
      "review_required": true
    }
  ]
}
```

## 欄位說明

### `video`

| 欄位 | 型別 | 說明 |
|---|---|---|
| `file` | string / null | Annotated MP4 檔名 |
| `duration_sec` | number | Annotated MP4 秒數 |

### `litter_detection`

這一區是 RT-DETR candidate 到 confirmed event 的執行診斷，不是人工標註或 accuracy。
所有 `frame_index` 都是從 `0` 開始；同一幀可能有多個 bbox，因此
`candidate_count` 是 bbox observation 數，不是影片中不重複的垃圾物件數。

| 欄位 | 型別 | 說明 |
|---|---|---|
| `rtdetr_4channel.enabled` | boolean | 本次是否啟用 RT-DETR 4-channel branch |
| `rtdetr_4channel.confidence_threshold` | number / null | 本次 `TRASH_CONF`；模型輸出低於此值不會進入 candidate |
| `rtdetr_4channel.evaluated_frame_count` | integer | 車輛 gate 開啟且實際交由此 branch 評估的幀數 |
| `rtdetr_4channel.vehicle_gate_skipped_frame_count` | integer | 因 vehicle gate 不 active 而略過 litter branch 的幀數；這些幀不能解讀為 RT-DETR 漏檢 |
| `rtdetr_4channel.candidate_count` | integer | RT-DETR 輸出的 `litter` bbox 總數，尚未套用 geometry/motion/holding |
| `rtdetr_4channel.detected_frame_count` | integer | 至少有一個 RT-DETR `litter` bbox 的幀數 |
| `rtdetr_4channel.detected_frames` | object[] | 各辨識幀的 `frame_index` 與該幀 `candidate_count` |
| `geometry_passed` | object | 通過 bbox 尺寸與長寬比後的相同三個統計欄位 |
| `motion_holding_passed` | object | 再通過 camera-shake、motion、core-motion、vehicle FP 與 holding gate 後的相同三個統計欄位 |
| `confirmed_event_count` | integer | 最終由 `GlobalLitterTracker` 確認的事件數，與 `summary.litter_event_count` 相同 |

### `summary`

| 欄位 | 型別 | 說明 |
|---|---|---|
| `litter_event_count` | integer | `GlobalLitterTracker` confirmed 垃圾事件數，不含 raw candidate |
| `urinate_event_count` | integer | STGCN temporal confirmation 事件數 |
| `passed_vehicle_count` | integer | Unique vehicle/scooter tracker ID 數 |
| `average_litter_confidence` | number / null | Confirmed litter detector confidence 平均值 |
| `detection_accuracy` | number / null | 只有 human-reviewed ground truth 才可填；目前固定 `null` |
| `accuracy_status` | string | 目前固定 `not_evaluated` |
| `littering_plates` | string[] | 已可靠辨識且關聯到垃圾事件的車牌；無結果時為空陣列 |
| `review_required` | boolean | 有 confirmed AI event 時為 `true` |

### `events`

垃圾事件欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `type` | string | `litter` |
| `id` | integer | Litter event ID |
| `start_sec` / `end_sec` | number | Estimated release/candidate birth 到 confirmation 的影片片段 |
| `confidence` | number / null | RT-DETR confidence，不是 accuracy |
| `vehicle` | string / null | 關聯 vehicle/scooter，例如 `vehicle:7`；不足時為 `null` |
| `plate` | string / null | OCR 車牌；失敗時為 `null`，不得猜測 |
| `plate_confidence` | number / null | OCR confidence |
| `plate_status` | string | `recognized`、`pending`、`attempted_no_result`、`not_requested` 或 `not_applicable` |
| `attribution_status` | string / null | Smart Backtrack 狀態，例如 `resolved`、`dustbin`、`pending` |
| `review_required` | boolean | 固定為 `true` |

Urinate 事件欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `type` | string | `urinate` |
| `track_id` | integer / null | STGCN confirmed person track |
| `time_sec` | number / null | Temporal confirmation 時間 |
| `start_sec` / `end_sec` | number / null | 已累積 STGCN evidence 到 confirmation 的可剪輯區間 |
| `confidence` | number / null | STGCN confidence，不是 accuracy |
| `vehicle` | string / null | Confirmed 人物經 action actor history 回追到的 vehicle/scooter |
| `plate` | string / null | 關聯車輛的 OCR 車牌；失敗時為 `null` |
| `plate_confidence` | number / null | OCR confidence |
| `plate_status` | string / null | 與垃圾事件相同的 OCR 狀態集合 |
| `attribution_status` | string / null | 找到 action 車輛候選時為 `resolved`，否則 `dustbin` |
| `review_required` | boolean | 固定為 `true` |

## 證據限制

- `rtdetr_4channel.candidate_count=0` 只能代表「有評估的幀沒有通過 `TRASH_CONF` 的
  litter bbox」；若 `enabled=false` 或 `evaluated_frame_count=0`，不能解讀為模型漏檢。
- `candidate_count` 是跨幀 bbox observation 數；同一實體可能連續多幀被計數。
- Confidence 是模型分數，不是準確率。
- `detection_accuracy` 沒有人工 reviewed labels 時必須保持 `null`。
- `passed_vehicle_count` 以 tracker ID 計算；ID fragmentation 可能高估實際車數。
- `attribution_status=resolved` 仍是模型歸因，不是 ground truth。
- Urinate 的 `start_sec` 由已累積 evidence 秒數回推，供看片與剪輯，不代表人工標定的
  行為開始時間。
- Person、vehicle 或 plate 證據不足時保留 `null`，交由人工複核。

## 研究 sidecar

`SMART_BACKTRACK_SIDECAR=0` 為 production 預設，因此一般執行只有 annotated MP4
與一份 analysis JSON。研究標註需要完整成本資料時，可明確設為
`SMART_BACKTRACK_SIDECAR=1`，額外產生 `_backtrack_candidates.jsonl`；該檔不供
網頁使用，也不是 ground truth。
