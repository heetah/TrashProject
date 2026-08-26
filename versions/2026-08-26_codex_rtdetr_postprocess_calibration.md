# RT-DETR 後處理校正（2026-08-26）

- 作者：Codex
- Branch：`heetah-dev`（工作樹尚未整合至 `main`）
- 目的：以 `runs/grounding_truth` 的人工 litter bbox/frame 對齊 63 部影片，定位 RT-DETR candidate 在 geometry、motion/holding、tracker 前的淘汰原因，並驗證安全的後處理修正。

## 變更

1. `scripts/pipeline/detect.py` 增加 opt-in `litter-candidate-gates/v1` sidecar，逐 candidate 保存 frame、bbox、confidence、filter outcome/reason、tracker ID/state。
2. `scripts/main.py` 在 `LITTER_CANDIDATE_SIDECAR=1` 時寫出 sidecar，並保存 threshold/dedup 設定。
3. `scripts/pipeline/geometry.py` 將 vehicle containment threshold 改為可由 `LITTER_FP_CONTAINMENT_THR` 控制，預設從 `0.85` 調整為 `0.999`：完全（含像素誤差）被 vehicle bbox 包住的候選仍淘汰，非完全重疊者交由後續 motion、holding、relative-separation 與 tracker confirmation 判斷。
4. `scripts/calibrate_litter_postprocess.py` 對 GT bbox/frame 配對 tracker ID，輸出 event-level Wilson 95% CI、clip-level coverage、paired bootstrap、exact McNemar 與 gate counts；不改寫 canonical ground truth。
5. replay runner 支援 `SOURCE_DIRECTORY`、`CASE_IDS`、`SKIP_EXISTING_SUCCESS`。

## A/B 證據

使用同一 RT-DETR/actor 模型與其他參數，從 `/home/under115a/under115a/under115a/litter_vidshort/litter` 重跑全部 63 部：

| 指標 | 舊版 threshold 0.85 | 校正版 threshold 0.999 |
|---|---:|---:|
| usable event（58）正確 tracker ID | 12/58 | 13/58 |
| event-level Wilson 95% CI | 12.25%--32.77% | 13.59%--34.66% |
| confirmed clip（63） | 19/63 | 20/63 |
| confirmed clip Wilson 95% CI | 20.24%--42.36% | 21.59%--44.00% |
| paired gain/loss | — | +1 / 0 |
| paired bootstrap 95% CI（rate difference） | — | 0--5.17 percentage points |
| exact McNemar two-sided p | — | 1.000 |
| unverified confirmed-track safety proxy | 14 | 14 |

新增正確 confirmed 為 case 167。去重實驗（同幀 IoU 0.5）為 12/58、未驗證額外
confirmed 14→15，因此保持關閉。

## 限制與回滾

- 63 部影片中 58 部 usable、5 部因 `extremely small` 不可判定；usable clip 沒有
  negative-only control，因此 14→14 只是 safety proxy，不是 false-positive rate。
- `event_annotations.jsonl` 的 58 筆 `review_state` 現值為 `unreviewed`，工具只標示
  此問題，不把它們升格為 reviewed ground truth。
- paired gain 的 p 值不顯著，不能宣稱 0.999 是普適或最優常數；需要更多事件、negative
  clips、跨攝影機 test 與人工 route review。
- 回滾：設定 `LITTER_FP_CONTAINMENT_THR=0.85` 即重現舊版 gate；或將
  `LITTER_CANDIDATE_SIDECAR=0` 關閉研究 sidecar。sidecar 與校正報告為新增 artifacts，
  不參與前端事件判定。

## 驗證

```bash
conda run -n rtdetr python -m py_compile \
  scripts/main.py scripts/pipeline/detect.py scripts/pipeline/geometry.py \
  scripts/calibrate_litter_postprocess.py
conda run -n rtdetr python -m pytest -q \
  tests/pipeline/test_detect_characterization.py \
  tests/pipeline/test_calibrate_litter_postprocess.py
```
