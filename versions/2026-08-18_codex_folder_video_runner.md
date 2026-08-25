# 2026-08-18 folder video runner

- 日期：2026-08-18
- 作者：Codex
- Branch：working tree（未提交）
- Commit：未提交

## 問題

`scripts/main.py` 的 production CLI 一次只接受一個影片路徑，需要批次處理
`/mnt/8tb_hdd/under115a/litter_vidshort/litter_order` 內的影片。

## 變更

- 新增 `scripts/run_litter_order.sh`。
- 遞迴尋找支援的影片副檔名，依檔名字典序逐支呼叫 `scripts/main.py`。
- 預設輸出到 `output/litter_order/`，單支失敗時繼續處理其餘影片，最後回報統計並以非零狀態結束。
- README 新增批次執行與環境變數覆寫說明。

## 介面/config

- `INPUT_DIR`：預設 `/mnt/8tb_hdd/under115a/litter_vidshort/litter_order`
- `OUTPUT_ROOT`：預設 `output/litter_order`
- `CONDA_ENV`：預設 `rtdetr`

## 測試證據

- `bash -n scripts/run_litter_order.sh`
- 未執行實際影片推論；本次只驗證腳本語法與文件內容。

## 限制與回滾

- 腳本為逐支、非平行執行；每支影片會重新啟動 `scripts/main.py` 並載入模型。
- 移除 `scripts/run_litter_order.sh` 並還原 README 對應段落即可回滾。
