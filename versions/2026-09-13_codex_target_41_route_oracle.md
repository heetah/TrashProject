# 41/58 目標與 route candidate oracle 稽核

- 日期：2026-09-13
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交 working tree（base `a149c5975898643e5131caf0579bd0aefe228d75`）
- 類型：research / test / docs

## 問題背景

研究目標由 45/58 調整為 41/58。為避免把 41 件 assignment-conditioned event matches
誤當成 41 件完整正確歸因，本研究把成功條件鎖定為 fixed 58 denominator 中，事件通過
strict/moderate match，且 route type、person key、vehicle key 與同輪人工 actor mapping
完全一致。

## 實作內容

新增 `scripts/analyze_route_oracle_gap.py`。工具沿用正式 readiness evaluator 的 event
matching 與 same-run IoU mapping，對每個 accepted event 檢查人工真值 tuple 是否已存在於
該 candidate record 的 valid、未 rejected routes。

分析最佳研究組合 `direct_vehicle_penalty=1.1`、
`release_window_prior_weight=0.2` 的 frozen sidecar，得到：

- 現行完整正確歸因：33/58（56.90%）。
- 另有 7 件 truth route 已存在但排序錯誤：141、16、168、67、7、78、9。
- frozen-match candidate oracle：40/58（68.97%）。
- case 26 為 actor mapping unavailable；vehicle mapping IoU=0.9272，但人工 person bbox 與
  最佳同幀 person tracklet IoU=0.2844，未通過既定 0.50 gate。
- 其餘為 15 missed events、2 exploratory/unaccepted event matches。

因此，即使在 7 件錯誤上用人工答案做完美重排，仍差 1 件；只調成本函數在目前候選空間
中無法達成 41/58。case 26 影像顯示人工 person bbox 同時橫跨多個人物／車體區域，無法由
現有標註唯一證明 person tracklet 身分；此案例需要重新裁定標註或建立可驗證的 mapping
規則，不能為達標而降低全域 IoU 門檻。

## API／Config／Schema 變更

- 新增研究輸出 schema：`route-oracle-gap/v1`。
- 未修改 production runtime、成本權重或環境變數預設。
- 研究目標為 41/58（70.69%）；不等同原產品 85% 上線門檻。

## 測試證據

```text
tests/pipeline/test_route_oracle_gap.py: 3 passed
tests/pipeline: 359 passed, 12 skipped
denominator: 58
current: 33/58
frozen-match candidate oracle: 40/58
target reachable by frozen-match reranking: false
```

完整輸出位於 `artifacts/route_oracle_gap_41_20260913/`。

## 已知限制

- Oracle 使用人工答案選 route，只能診斷候選覆蓋，不能部署。
- Event match 依 selected assignment 的 release frame/point；本稽核凍結 match，不能視為
  全域 joint oracle。
- 58 部均為同一 development positive set，沒有 reviewed negatives 與 camera holdout。
- 41/58 的點估計 Wilson 95% 下界約 57.99%，遠低於 85% 上線要求。

## 回滾方式

刪除新增的分析 script、targeted test、artifact 與本版本說明，並移除 README 的候選空間
稽核段落即可。Production 行為沒有變更，不需 runtime rollback。
