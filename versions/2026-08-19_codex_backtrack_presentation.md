# Smart Backtrack 版本差異與數學模型簡報

- 日期：2026-08-19
- 作者：Codex
- Branch：heetah-dev
- Commit：d638365（產製時基準；工作樹另含本次簡報與 README 變更）
- 類型：docs

## 問題背景

需要向具資工背景、但不熟悉 CV 反追蹤的客戶與老師，清楚說明 D+T 舊版與新版
Smart Backtrack 的差異、設計理由、成本矩陣及正式數學模型。

## 實作內容

- 新增 `scripts/create_backtrack_slides.py`，從 live backtrack contract 產生 16 頁簡報。
- 簡報包含 Kalman、confidence-aware covariance、RTS、release hypothesis、
  C_BA/C_AC/C_BC、route cost、Min-Cost Flow、NULL、margin、sidecar 與驗證指標。
- 同時輸出 ODP 與 PPTX；文字、方塊、連線與公式以簡報原生文字/圖形物件保存，
  不使用整頁截圖。
- 本次重新產製改為白色背景、黑色文字；灰階色塊只作區塊與流程層級提示。
- README 新增簡報重建指令與輸出位置。

## API／Config／Schema 變更

無 production API、模型權重或 attribution runtime 變更。新增的是文件產製工具與
`artifacts/presentations/` 下的研究溝通產物。

## 測試證據

- `python3 -m py_compile scripts/create_backtrack_slides.py`：通過。
- `python3 scripts/create_backtrack_slides.py`：成功產生 16-slide PPTX/ODP。
- `unzip -t artifacts/presentations/smart_backtrack_version_comparison_20260819.pptx`：通過。
- 40 個 PPTX XML/rels 全部可由 Python XML parser 解析；16 個 slide 與 16 個 slide
  relationship 均存在。
- LibreOffice headless conversion 在目前容器因 GUI/dconf runtime 限制返回非零，
  未將此結果宣稱為視覺相容性通過；使用者可在 PowerPoint/LibreOffice 桌面端開啟檢查。

## 已知限制

- 簡報內容是 live code contract 的技術說明，不是人工 reviewed ground truth。
- seven-case regression 的現有數字只能作案例診斷；不能把 coverage、resolved 或 margin
  直接宣稱成 attribution accuracy。
- case 25/168 的遮擋/釋放證據仍是下一階段模型研究問題。

## 回滾方式

刪除 `scripts/create_backtrack_slides.py`、本 version note 及 `artifacts/presentations/`
下的簡報產物；production `scripts/pipeline/backtrack/` 不受影響。
