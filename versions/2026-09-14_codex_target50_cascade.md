# Litter confirmed-event 50/58 development cascade

- 日期：2026-09-14
- 作者：Codex
- Branch：`heetah-dev`
- Commit：`a149c5975898643e5131caf0579bd0aefe228d75` + working tree
- 類型：feat / test / docs / research validation

## 問題

既有 45/58 run 的實際 shell 設定未完整寫進 experiment manifest；candidate sidecar 證明
該 run 使用 `LITTER_CANDIDATE_DEDUP=1` 與
`LITTER_FP_STREAK_MIN_OBSERVATIONS=6`。未帶這兩項的重跑只有 40/58。先前嘗試擴大
single-seed visual chain 雖補回 case 135，卻造成六個 accepted-event regressions，已撤回。

## 變更

- `PipelineConfig` 新增 optional `RTDETR_IMGSZ`，只影響 PyTorch RT-DETR；空值保持既有
  checkpoint/runtime 行為，固定 shape TensorRT engine 不受影響。
- 新增 `scripts/build_litter_cascade_view.py`。它只讀 primary analysis 的
  `confirmed_event_count`；count 不在 `[1,3]` 時選 secondary，否則保留 primary。
- Builder 不讀 ground truth、不重寫推論證據，以 symlink 建立 candidate view，並輸出逐案
  `cascade_selection.jsonl`。需要的 secondary 缺失或 analysis malformed 時 fail closed。
- production litter tracker 的 medium-seed／horizontal-excursion 擴張已撤回；既有大型 seed
  四步 visual chain 行為不變。

## 驗證證據

- Reproduced primary regression screen：cases 9、13、25、42、49、75 在正確 dedup/streak
  設定下全數恢復 strict/moderate accepted。
- Secondary screen：13 個 primary misses 中，cases 18、34、135、143、193 accepted。
- Machine-only trigger 唯一額外選到的既有成功案 case 76，在 secondary 仍為 strict 且 route
  correct。
- 完整 composite：50/58 = 86.21%，Wilson 95% CI 75.07%--92.84%；相對 45/58 gains=5、
  losses=0。Route correctness 33/58；correctness gains=2、losses=0。
- Targeted tests：79 passed；完整 pipeline suite：395 passed、12 skipped；cascade view 的
  58/58 annotated MP4 均可開啟並讀取首幀，broken symlink=0。

## 限制

- 58 案全為 positive development clips，不能估 precision 或 false-positive rate。
- Trigger、alternate checkpoint 與 1216 input size 都在同一 cohort 選定，沒有獨立攝影機
  holdout；paired exact test 也未證明顯著改善。
- 四個仍未 accepted 的 secondary confirmations 是 safety warning，不是已審核 false positive。
- Plate OCR correctness 未評估，`ready_for_enforcement=false`；cascade 不得直接作為自動開罰
  依據或 production 預設。

## 回滾

移除 `scripts/build_litter_cascade_view.py`、其測試、`RTDETR_IMGSZ` config/warmup override 與
本 README 段落即可。保留 output/artifact 供稽核；它們不是 ground truth。
