# Debug Actor Track IDs

- 日期：2026-08-18
- 作者：Codex
- Branch：heetah-dev
- Commit：基準 `d638365`，本變更未提交
- 類型：feat / test / docs

## 問題背景

`LITTER_DEBUG=1` 原本只輸出 tracker 診斷文字。Annotated video 的 confirmed litter
已有 litter ID，但 person、vehicle、scooter 標籤沒有 track ID，難以人工核對反追蹤
route 與畫面 actor。

## 實作內容

- Debug mode 的 actor 標籤加入 `ID:<track_id>`。
- 正常模式標籤與 production 歸因流程不變。
- 新增 GPU-free renderer test，確認 person/vehicle ID 會出現在 debug 標籤。

## API／Config／Schema 變更

```text
LITTER_DEBUG=1
```

只改 annotated video 與診斷輸出；analysis JSON schema 不變。

## 測試證據

- `tests/pipeline/test_detect_characterization.py`：`3 passed`。
- `scripts/pipeline/detect.py` compile smoke 通過；`git diff --check` 通過。
- 以兩張 RTX 4090、`LITTER_DEBUG=1`、`SMART_BACKTRACK_SIDECAR=0` 重跑
  case 25、42、50、51、165、168；6/6 child run exit 0。
- 6/6 annotated MP4 通過 `ffprobe`，frame count 與來源一致。
- 六案的 compact summary/events 與非 debug full-stage run 完全相同。
- 人工抽查 case 25 frame，person/vehicle/scooter label 已顯示 track ID。

輸出：`output/backtrack_six_debug_ids_20260818_full/`。

## 已知限制

Track ID 是 tracker runtime identity，不是車牌、身分或人工 ground truth。

## 回滾方式

移除 debug label 的 `ID:<track_id>` 拼接與對應測試、文件即可。
