# 2026-09-15 — 優化版反追蹤算法技術簡報

- 日期：2026-09-15
- 作者：Codex
- 分支：`heetah-dev`
- 類型：研究成果溝通 artifact；無 production code 變更
- 參考版型：`/home/under115a/.codex/attachments/33dae050-8961-4d3b-8f7b-3a2c58ffe019/08_27 專題進度回報.pptx`
- 產出：`artifacts/optimized_backtrack_presentation_20260915/優化版反追蹤算法_技術說明_2026-09-15.pptx`

## 內容

簡報以白底、黑／灰文字、細分隔線、灰階卡片與右上角圓形「反向追蹤」標籤重現參考
風格，共 13 頁，涵蓋：

1. Kalman Filter 與 RTS smoothing 的狀態／covariance 更新；
2. release-time hypotheses（3 點 ballistic、2 點 constant velocity、1 點保守 fallback）；
3. 分層成本 `C_BA`、`C_AC`、`C_BC` 與 gate-before-cost 原則；
4. Gaussian NLL、Mahalanobis 與 release prior 的數學限制；
5. Min-Cost Flow、capacity、NULL 與 Hungarian 責任邊界；
6. pseudo-homography／深度的可行範圍與目前 `LOCKED` 失敗回退；
7. provenance／lineage、研究候選 48/58 結果與獨立 holdout 下一步。

## 證據與限制

簡報引用的研究候選為 58 部 development clips 的 frozen replay：43/58 → 48/58
route-correct、accepted event match 50/58 → 51/58、5 gains、0 losses；85% 點估計仍需
50/58，且 paired sign-test `p=0.0625`、Wilson 下界約 0.711。盲測來源預演為 18/18
來源 SHA 驗證、29 個事件紀錄中 1 筆路由變更；worksheet 仍 `unreviewed`，因此沒有
camera-independent accuracy、negative-set FPR 或 plate-OCR readiness 宣稱。

簡報是以 SVG 頁面轉成 PNG 後嵌入 PPTX 的視覺 artifact；重建來源保留於
`artifacts/optimized_backtrack_presentation_20260915/build_deck.py`。它不讀取或修改
resolver、cost、route schema、configuration、ground truth 或 production output。

## 驗證

- `/usr/bin/python3 artifacts/optimized_backtrack_presentation_20260915/build_deck.py`：成功產生 13 頁 PPTX。
- LibreOffice headless 轉換 PDF：13 頁成功。
- `pdftoppm` 產生 13 頁預覽，逐頁檢查中文、公式、流程圖、頁碼與邊界，未發現空白頁或文字溢位。
- `unzip -t`：PPTX 壓縮包完整。
- `git diff --check`：通過。

## 回滾

本項沒有 runtime migration。若不採用簡報，只需停止使用該 artifact 與重建腳本；
production Smart Backtrack 不受影響。
