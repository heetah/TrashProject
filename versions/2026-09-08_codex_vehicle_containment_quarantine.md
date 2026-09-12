# Vehicle-contained litter evidence quarantine

- 日期：2026-09-08
- 作者：Codex
- Branch：`heetah-dev`
- Commit：uncommitted working-tree changes
- 類型：`feat(litter)`

## 問題背景

Production 原先將幾乎完全位於 vehicle/scooter bbox 內的 RT-DETR
litter candidate 直接丟棄。這會壓制車燈、車體細節等 false positive，但也會
丟失從車內或車體邊緣開始的真實下落軌跡。直接降低門檻不符合執法安全原則，
因此本次改為 fail-closed evidence quarantine。

## 實作內容

- `vehicle_contained` candidate 改送入 tracker-only quarantine，不再直接淘汰。
- Quarantine 軌跡在解除前無法進入 litter confirmation。
- 只有在同一個不可變的 carrier track 座標系中，至少 3 個 observation、
  2 個向下 step，且同時通過尺寸正規化的相對位移與下落證據才解除。
- Detector 中斷超過 0.35 秒會重置證據段，避免將不同物件錯接成落下軌跡。
- Quarantine 與 ordinary pending 軌跡進行雙向 identity isolation：隔離歷史不會
  觸發普通候選過濾或確認，普通軌跡也不會因單幀 containment 抖動被隔離。
- Candidate sidecar run record 新增六個 quarantine 參數，供後續執行重現。

## API／Config／Schema 變更

- CLI 與 production analysis JSON schema 無變更。
- `GlobalLitterTracker.update()` 新增 optional `quarantined_litters` 參數。
- 新增六個 `LITTER_VEHICLE_QUARANTINE_*` 環境變數，預設值見 `.env.example`。
- Candidate sidecar 仍為 research evidence，不是 ground truth。

## 驗證證據

- 聚焦單元與 characterization：59 passed。
- `tests/pipeline`：332 passed、12 skipped。
- 完整 `tests`：359 passed、12 skipped。
- Python compile smoke 與 `git diff --check`：passed。
- 63 支固定批次：63 success，63/63 MP4 可由 `ffprobe` 解碼。
- 58 支 usable positive clips：
  - strict/moderate event sensitivity：36/58 = 62.07%（Wilson 95% CI 49.20–73.44%）。
  - provisional end-to-end route correctness：27/58 = 46.55%
    （Wilson 95% CI 34.33–59.20%）。
  - accepted-event conditional route correctness：27/36 = 75.00%
    （Wilson 95% CI 58.93–86.25%）。
- 與舊 baseline 23/58 成對比較：4 gains、0 losses，event-match 4 gains、0 losses；
  exact paired two-sided p = 0.125，未達統計顯著。

## Release decision 與已知限制

**BLOCKED — 不得宣稱已達 85% 或可用於自動執法。**

- 58 筆 event label 全部仍為 unreviewed，指標只是 provisional。
- Positive-only set 無法估計 precision/false-positive rate。
- 尚無 independent camera-group holdout。
- 尚無 reviewed plate OCR ground truth，因此 finable-case correctness 未評估。
- 85% point estimate 需至少 50/58；Wilson 95% lower bound 達 85% 需至少
  55/58。目前分別尚差 23 與 28 筆正確端到端案例。

## 回滾方式

最小回滾是將 `scripts/pipeline/detect.py` 的 `vehicle_contained` 路徑恢復為
pre-tracker rejection，並移除 tracker 的 quarantine state。回滾後應重跑同一個
58-case paired evaluation，不可只比較 confirmed count。
