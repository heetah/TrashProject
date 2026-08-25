# Smart Backtrack Cost Study Infrastructure

- 日期：2026-08-02
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交
- 類型：feat

## 問題背景

既有 sidecar 可檢視成本，但無法保證同一個 resolver input 被重播；研究也無法
循序比較 distance/time、confidence、uncertainty 與 reverse trajectory。

## 實作內容

- sidecar 新增 JSON-safe `resolver_input`，保存 immutable worker task。
- 新增 immutable `BacktrackCostConfig` 與 stage presets。
- 新增研究模式：baseline 使用 birth anchor/觀測 bbox，uncertainty 才開 Kalman/RTS，
  reverse 才開反向 trajectory。
- 新增 grouped manifest、frozen replay、reviewed-label evaluation CLI。
- 新增 blind annotation queue CLI；模板不複製 runtime prediction。
- evaluation 另報 wrong non-NULL rate 與 deterministic bootstrap 95% CI，避免
  以 resolved 比例掩蓋錯誤歸因。

## API／Config／Schema 變更

- `scripts/backtrack_study.py`：`manifest`、`replay`、`evaluate`。
- `smart-backtrack-candidates/v1` 新增 additive `resolver_input` 欄位；沒有此欄位的
  舊 sidecar 可閱讀但不可 replay。

## 測試證據

`conda run -n rtdetr python -m pytest -q tests/pipeline/test_backtrack_costs.py tests/pipeline/test_backtrack_kalman.py tests/pipeline/test_backtrack_study.py tests/pipeline/test_smart_backtrack_integration.py`

## 已知限制

目前沒有本機 reviewed dataset 或影片，未產生 accuracy；安全選模與 95% CI 必須在
人工 reviewed annotation 到位後執行。

## 回滾方式

移除研究工具與 additive sidecar 欄位即可；production full 預設成本與 Smart
Backtrack 路徑不需變更。
