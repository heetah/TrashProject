# Seven-case Smart Backtrack sequential ablation

- 日期：2026-08-18
- 作者：Codex
- Branch：heetah-dev
- Base commit：d638365
- 類型：fix / research / test

## 問題

七個人工複核案例中，case 24 是瞬時 false litter；case 25、42、50、51、
165、168 需要可重建的 actor attribution。舊 sidecar 會重複保存 OCR 影像，
Min-Cost Flow 的 0.001 成本精度也會把不同 actor 變成 route-ID tie。

## 依序完成

1. sidecar 排除 `plate_actor_frames`、`plate_roi` 像素，保留 resolver 所需
   actor 幾何、confidence、ID 與 litter history。
2. 建立 `seven_cases_v1` reviewed annotation；case 50 litter 8 標為
   `not_litter`。
3. confirmation 加入 birth/current actor support、兩點 actorless fast-drop
   限制與 arc descent 驗證。七案中保留 7/7 reviewed true litter，排除
   6/6 已標 false candidates；此數字只適用這七案。
4. 兩點軌跡使用 bounded constant velocity；最大回推由 0.3 s 調成 0.4 s。
5. `C_BC` 新增 reverse/exit/relative diagnostics，但跨案方向不一致，
   production weight 維持 0。
6. ballistic release 加入最多 0.5 s forward hypotheses，使 case 42 的
   person 2 進入候選。
7. `C_AC overlap` 消融 0.8/0.4/0.2/0.0；0.0 排除錯誤深度 bbox overlap，
   case 42 改為正確 `person 2 -> vehicle 1`。
8. Min-Cost Flow cost scale 由 1e3 提升至 1e6；case 50 litter 5 因此由
   tie-break 的 vehicle 5 改為原始成本較低且正確的 vehicle 9。
9. sidecar 新增 route、distinct-person、distinct-vehicle、NULL margins；
   actor margin 先按 actor collapse，同 actor 的不同 route/release 不算
   identity tie。

## 驗證結果

- 七案 conditional attribution：Exact Route Top-1 `5/7`；Recall@3 `7/7`；
  release interval hit `7/7`、MAE `0 frame`。
- 仍錯：case 25 選 vehicle 1（正確 vehicle 2）；case 168 選 vehicle 3
  （正確 vehicle 1）。兩案都是多車遮擋；延長回推與現有 C_BC
  direction/exit/relative 訊號均無一致解。
- 固定 63 clips production scan：`63/63` 真實案例 exit 0；18 clips 共有
  23 confirmed litter，另有 7 urinate events。21 個輸出 clips 產生 run
  sidecar，23 個 confirmed litter 產生 candidate record。
- 固定集 replay：候選設定只改 case 42 route，以及 case 25 release frame
  38 -> 36；其餘 21/23 route 不變。固定集尚無完整 reviewed routes，
  因此不是 accuracy 證據。
- Final pipeline suite：`158 passed, 12 skipped`；另通過 backtrack/litter
  modules compile smoke。

## 限制與下一步

case 25/168 的正確車均已在 Recall@3，但 current bbox/track/release input
不足以穩定分離遮擋車。不可用單一 margin threshold：case 50/51 的正確
identity margin 也小於 0.01。下一步應加入 release-origin instance mask／
occlusion ordering 或人工複核標註，不應用兩案專屬權重硬修。

## 回滾

- Flow precision：`COST_SCALE` 改回 1000。
- AC ablation：`BacktrackCostConfig.ac_weights["overlap"]` 改回 0.8。
- Two-point horizon：預設與環境變數 fallback 改回 0.3。
