# 2026-09-14 反追蹤模組修改重點與策略簡報

- 日期：2026-09-14
- 作者：Codex
- 分支：`heetah-dev`
- 類型：研究成果簡報；不包含 production code 變更
- 參考版型：`08_27 專題進度回報.pptx`
- 產出：`artifacts/target41_phase1b_research_20260914/反追蹤模組_修改重點與策略_2026-09-14.pptx`

## 內容範圍

簡報彙整 Phase 0 provenance/lineage 合約、Phase 1A 數學 primitives、58 部 reviewed
clips 的 frozen replay、paired comparison、未採用的研究方向，以及達成 41/58 前的下一步
校準策略。

簡報中的主要數字均來自同一份可重現研究證據：原始 cascade 33/58、最佳 frozen replay
38/58、frozen-match oracle 47/58；paired comparison 為 +5/0 回退，exact two-sided
sign-test p=0.0625。這些數字不是 enforcement readiness 或 85% 保證。

## 驗證

- PPTX 由 LibreOffice Impress 正常重新開啟並轉出 11 頁 PDF。
- 逐頁檢查中文、公式、表格、頁碼、數字與邊界；未發現空白首頁或文字重疊。
- `git diff --check` 通過。

## 限制與回滾

本簡報為單一研究報告 artifact，未接入 resolver、cost、route schema 或 runtime。若需
回滾，只需移除該 PPTX 與同目錄的 rendered preview/source artifact，不影響 production
pipeline。
