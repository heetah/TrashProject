# Analysis JSON 輸出格式

- Schema version：`2.0.0`
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
  "schema_version": "2.0.0",
  "video": {
    "file": "resize_annotated.mp4",
    "duration_sec": 135.0
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

Urinate 事件只保留：`type`、`track_id`、`time_sec`、`confidence`、
`review_required`。

## 證據限制

- Confidence 是模型分數，不是準確率。
- `detection_accuracy` 沒有人工 reviewed labels 時必須保持 `null`。
- `passed_vehicle_count` 以 tracker ID 計算；ID fragmentation 可能高估實際車數。
- `attribution_status=resolved` 仍是模型歸因，不是 ground truth。
- Person、vehicle 或 plate 證據不足時保留 `null`，交由人工複核。

## 研究 sidecar

`SMART_BACKTRACK_SIDECAR=0` 為 production 預設，因此一般執行只有 annotated MP4
與一份 analysis JSON。研究標註需要完整成本資料時，可明確設為
`SMART_BACKTRACK_SIDECAR=1`，額外產生 `_backtrack_candidates.jsonl`；該檔不供
網頁使用，也不是 ground truth。
