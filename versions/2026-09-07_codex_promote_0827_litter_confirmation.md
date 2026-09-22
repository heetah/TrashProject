# Promote the 8/27 litter-confirmation recovery profile

- 日期：2026-09-07
- 作者：Codex
- Branch：current working branch
- Commit：uncommitted working-tree changes
- 類型：feat(litter)

## 問題背景

目前保守 production confirmation 預設在 58 部 usable、幾乎都含真實垃圾的
ground-truth clips 中，只讓 20/58 部進入 confirmed litter 階段。2026-08-26/27
的 recovery replay 使用較寬鬆、可重現的 confirmation profile，得到 41/58
usable clips confirmed；專案人員後續表示已人工檢查該版輸出，未發現明顯問題，
因此決定將該 profile 提升為 production 預設。

這是 confirmed-event coverage 決策，不是 Smart Backtrack attribution 公式調整。
目前未擴張 vehicle bbox、`D <= 0.4`、actor evidence
`<= 3 frames AND <= 0.25 s` 的設定保持不變。

## 實作內容

Production 的程式內預設與 `.env.example` 現在採用：

```text
LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR=0
LITTER_MIN_CONFIRM_AGE_VEHICLE=2
LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE=7
LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT=1
LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE=10
LITTER_MIN_VEHICLE_RELATIVE_SEPARATION=0
LITTER_FP_STREAK_RATIO=10
LITTER_ALLOW_SHAKE_CANDIDATES=1
```

Candidate sidecar 的 run metadata 同步使用以上預設，確保未設定環境變數時，
記錄值與 tracker/detector 的實際行為一致。所有設定仍可由 shell、UI worker 或
`PIPELINE_ENV_FILE` 明確覆寫。

## API／Config／Schema 變更

- CLI 與 analysis JSON schema 無變更。
- `LITTER_CANDIDATE_SIDECAR` 與 `SMART_BACKTRACK_SIDECAR` 仍預設關閉；它們只影響研究記錄。
- Litter 仍須通過 motion、holding、trajectory/displacement、actor association 與
  non-stationary evidence 才能 confirmed。
- Confirmed litter 仍不等於歸因正確、OCR 成功或可自動開罰。

## 測試證據

- 原始 recovery replay：41/58 usable clips confirmed；完整歷史與命令見
  `versions/2026-08-26_codex_confirmation_recovery_replay.md`。
- 專案人員回報已人工檢查該版輸出，未發現明顯問題。此為人工複核紀錄，並非
  negative-set false-positive 統計。
- Python compile smoke：`scripts/main.py`、`pipeline/detect.py`、
  `pipeline/geometry.py`、`pipeline/litter_tracker.py` 通過。
- Production-default targeted tests：5 passed。
- 完整 `tests/pipeline`：240 passed、12 skipped（以 `LITTER_DEBUG=0` 執行；
  repository 本機 `.env` 的 debug overlay 會刻意改變 characterization image hash）。
- Case 164 production runtime smoke：RT-DETR 43 candidates、geometry 43、
  motion/holding 6、confirmed event 1；目前 Smart Backtrack 選到人工 GT 的
  `vehicle:3`。輸出 MP4 為 H.264、2592x1944、519 frames，可由 ffprobe 解碼。
- Case 164 candidate sidecar 的 run record 確認八個 production recovery 預設值
  均與 8/27 profile 相同。

## 已知限制

- 目前資料幾乎都是 positive litter clips，仍不足以估計 background/negative clip
  false-positive rate。
- 同一 clip 可能產生多筆 confirmed event；event records 不是彼此獨立樣本。
- `LITTER_ALLOW_SHAKE_CANDIDATES=1` 與關閉 relative-separation hard gate 會提高
  false-positive exposure，因此所有事件仍須人工複核，不得直接自動開罰。

## 回滾方式

不需修改程式即可用環境變數恢復先前保守 profile：

```text
LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR=1
LITTER_MIN_CONFIRM_AGE_VEHICLE=3
LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE=12
LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT=5
LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE=3.5
LITTER_MIN_VEHICLE_RELATIVE_SEPARATION=60
LITTER_FP_STREAK_RATIO=5
LITTER_ALLOW_SHAKE_CANDIDATES=0
```
