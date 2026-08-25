# Multi-Actor Gate-Normalized D+T Weight Study

- 日期：2026-08-04
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交
- 類型：feat / test

## 問題背景

固定 63 支 multi-actor clips，比較三組 distance/time 權重。原 D+T stage
仍把尺度化距離與秒直接相加，且 `C_AC` 內部選最佳 observation 時會使用
未啟用的 full-model components，因此不能嚴格解釋權重實驗。

## 實作內容

- D+T distance/time 各自除以 hard gate，形成 0--1 無因次 feature。
- 三矩陣共同接受顯式 distance/time trial weights。
- `C_AC` observation selection 與最終 cell 使用同一組啟用 components/weights。
- production `full` stage 保持既有 raw feature 與預設權重。
- 固定案例存於 `artifacts/backtrack_study/multi_actor_v1.json`。

## API／Config／Schema 變更

```text
SMART_BACKTRACK_DT_DISTANCE_WEIGHT=1.0
SMART_BACKTRACK_DT_TIME_WEIGHT=1.0
```

`StudyConfig` 新增 `distance_weight`、`time_weight`。兩者必須非負，且至少
一者大於零。

## 測試證據

- Unit/production suite：targeted `26 passed`；full `131 passed, 12 skipped`。
- Runtime：兩張 RTX 4090，以 tmux 跑完三組各 63 支 `multi-actor-v1` clips；
  189/189 child runs 為 exit status 0，189/189 annotated MP4 通過 `ffprobe`
  decode smoke。
- 三組均得到 60 confirmed litter events，分布完全相同：22 `person_vehicle`、
  38 `direct_vehicle`、60 resolved、0 dustbin；39/63 clips 有 confirmed event。
- 三組的每一個 selected `route_id` 完全相同。由 `0.50/0.50` 提升至
  `0.75/0.25` 時，mean margin 由 1.694 升至 1.733、median margin 由
  0.1765 升至 0.278；但仍有 19/60 events 的 margin 小於 0.1，8 個為 exact
  zero-margin tie。
- 因為 winner 沒有改變，上述 margin 改善不構成 attribution accuracy 證據；
  要以盲標 admissible route 檢查低 margin cases，尤其 case 25、42、50、51、
  135、138、145、168。

## 已知限制

resolved/dustbin 只能表示 route coverage。未完成人工 admissible-route 標註前，
三組 trial 不可宣稱 attribution accuracy。

## 回滾方式

移除 DT weight 環境變數並使用 `SMART_BACKTRACK_STUDY_STAGE=full`，即可維持
production historical feature units 與權重。
