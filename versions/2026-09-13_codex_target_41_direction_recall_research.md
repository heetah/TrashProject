# 41/58 direction and low-confidence recall research

- 日期：2026-09-13
- 作者：Codex
- Branch：heetah-dev
- Commit：未提交 working tree（base `a149c5975898643e5131caf0579bd0aefe228d75`）
- 類型：research / test / docs

## 問題背景

在候選 oracle 證明現有 routes 的 frozen-match ceiling 為 40/58 後，研究兩條不偷看
真值的路徑：(1) 輕量方向特徵 learning-to-rank；(2) missed events 的低信心 temporal
candidate coverage。

## 實作內容

1. 以 40 件可映射 strict/moderate events 重跑 TrackFlow-inspired leave-one-event-out
   分析，特徵為 normalized distance、release time 與
   `A=(1-cos(theta))/2` direction penalty。
2. 修正研究報告內硬編碼的 63-clip／6-person 數字，改由當輪資料生成。
3. 新增 exact top-1 的 paired gain/loss 列表；NLL 與 target metric 分開裁決。
4. 對 cases 18、20、34、161、193 以 `TRASH_CONF=0.05` 執行有 hash provenance 的
   raw-candidate 診斷；case 36 使用既有等價 smoke 結果。

## 研究結果

- D+T exact top-1：16/40（40.0%）。
- D+T+A exact top-1：10/40（25.0%），gain/loss=4/10，net=-6。
- Direction 雖改善 choice NLL `-0.091`，bootstrap 95% CI
  `[-0.174,-0.002]`，且 sign-flip `p=0.0254`，但它破壞 exact target，故拒絕作
  route selector。
- 低信心五案全部完成；新增 confirmed records 2 件，但皆為 exploratory match，accepted
  recovery 為 0/5。
- case 193 在 0.05 產生 201 candidates，其中 189 件被 motion gate 淘汰，顯示大幅降低
  threshold 主要增加噪聲。

## API／Config／Schema 變更

- Research report 新增 `dtd_minus_dt_exact_changes`。
- Production API、環境預設、成本函數與 tracker gates 均未修改。

## 測試證據

- Likelihood targeted tests：8 passed。
- 五案 inference：5 completed、0 failed。
- 五支輸出 MP4 均可解碼。
- Reviewed accepted-event recovery：0/5。

## 已知限制

- Leave-one-event-out 事件來自同一 development set，不能替代跨攝影機 holdout。
- Positive-only clips 無法估 false-positive rate；低 threshold 不可 promotion。
- case 20/34 可能需要 held-object／dumping 狀態標註；現有 event label 不足以監督此分支。
- Sub-agents 因 workspace agent-credit exhausted 無法執行獨立批判。

## 回滾方式

Production 未變更。若撤回研究工具報告增補，移除
`paired_binary_changes`、對應測試、兩個新 artifact folders 與本版本紀錄即可。
