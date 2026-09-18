# 反追蹤正式驗收雙人複核資料包

- 日期：2026-09-13
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交
- 類型：feat / test / docs

## 問題背景

既有 58 筆事件雖有人工種子標註，但 event records 全為 `unreviewed`，且沒有 reviewed
negative、NULL route 或已確認的 camera/site group。正片 coverage 或 provisional route
score 因此不能作為 85% 上線準確率證據。

## 實作內容

- 新增 model-blind release validation package builder。
- 58 支正片以原始未畫框影片建立 SHA-256，產生兩位 reviewer 的獨立空白欄位。
- 舊 event seed 與盲標 queue 分離，只放在 `adjudication_only/`。
- normal 資料夾影片只列為 unreviewed negative candidates，不自動標成 negative。
- 使用三幀 median dHash-128 產生 camera similarity retrieval suggestion；不得當作同攝影機真值。
- 所有 queue 在寫出前檢查無模型輸出、無 truth label 洩漏、無自動 promotion。

## API／Config／Schema 變更

新增 CLI：`scripts/build_release_validation_package.py`。

新增研究 schema：

- `release-validation-review/v1`
- `release-validation-negative-candidate/v1`
- `release-validation-camera-suggestion/v1`

不修改 production runtime、事件 schema、歸因成本或預設參數。

## 測試證據

- `tests/pipeline/test_release_validation_package.py`
- 與既有 annotation tests 合跑：10 passed、1 skipped。
- 完整 `tests/pipeline`：343 passed、12 skipped。
- 實際資料包：58 positive events、617 negative candidates；58/58 正片可解碼。
- 617 候選中 614 可解碼，3 支缺損；任何候選皆未被提升為 reviewed negative。

## 已知限制

- dHash threshold 18 僅是人工檢索 prior，未經 camera-ID 標註校正。
- 617 支 action-normal 影片不等於 litter-negative，必須完整雙人複核。
- 未完成雙人覆核、裁決、camera group 與獨立 holdout 前，不可計算或宣稱 85% 上線準確率。
- 此工具不建立 OCR ground truth。

## 回滾方式

移除新增 CLI、`release_validation.py`、對應測試與本版本文件即可；production 行為完全不受影響。
