# 隨地便溺車輛證據與已審核違規匯出

- 日期：2026-08-06
- 作者：Codex
- Branch：`heetah-dev`
- Commit：尚未提交
- 類型：feat

## 問題背景

人工複核介面仍提供「證據不足」判定，且隨地便溺事件雖然 runtime 會由 confirmed person
回查關聯 vehicle/scooter 並派送 OCR，analysis JSON 與 React 卡片沒有保留這段結果。
已審核案件也缺少一次下載所有成立違規片段與清單的方式。

## 實作內容

- 新人工判定只接受 `accepted`／`rejected`；既有 `uncertain` 紀錄保留但不再計入審核完成。
- `GlobalLitterTracker` 保存 confirmed urinate 實際解析到的 person→vehicle 關聯；每影片
  analysis JSON 的 urinate event 新增可剪輯區間、vehicle、plate、OCR 與 attribution 欄位。
- React 對垃圾與隨地便溺使用相同的車輛／車牌證據表格，兩類事件都可獨立人工修正車牌。
- 已審核頁新增匯出按鈕。後端收集完整審核案件中 `accepted` 的實際事件，以 FFmpeg 將每個
  事件剪成獨立 MP4，再以 openpyxl 建立 `違規清單.xlsx`，最後下載單一 ZIP。
- Excel 列出原始影片、違規類型、事件時間、模型分數、關聯車輛、AI／人工車牌、審核人、
  備註與片段狀態；人工文字會避免被 Excel 當成公式執行。

## API／Config／Schema 變更

- 新增 `POST /api/exports/reviewed`。
- 新增 `UI_EXPORT_ROOT`、`UI_FFMPEG_EXECUTABLE`、`UI_EXPORT_PRE_ROLL_SEC`、
  `UI_EXPORT_POST_ROLL_SEC`。
- `UI/backend/requirements.txt` 新增 `openpyxl==3.1.5`。
- Analysis schema version 維持 `2.0.0`，urinate event 新增向後相容的 optional 欄位；既有
  JSON 仍可讀取。

## 測試證據

- `conda run -n rtdetr python -m pytest -q tests/pipeline`：143 passed、12 skipped。
- `conda run -n rtdetr python -m pytest -q tests/ui`：14 passed。
- `conda run -n rtdetr python -m py_compile UI/backend/*.py scripts/main.py
  scripts/pipeline/action.py scripts/pipeline/events.py scripts/pipeline/litter_tracker.py`：通過。
- `cd UI/frontend && npm run build`：Vite production build 通過。
- 真實 FFmpeg smoke：3 秒 synthetic H.264 MP4 經 exporter 產生 ZIP，內含 Excel 與 1 支
  H.264 160×90 clip；`ffprobe` 可解碼且片段長度 0.934 秒。

未執行真實監視器影片 GPU inference；因此尚未驗證新 urinate 欄位在真實車輛回追與 OCR
輸出上的案例覆蓋率或正確率。

## 已知限制

- 匯出只包含已完整審核案件中的 accepted event，不包含 rejected 或 no-AI-event item。
- 舊 JSON 若無事件時間或 annotated MP4 已遺失，只會保留 Excel 列與失敗原因，無法補造片段。
- 匯出同步執行且 ZIP 保存在 `UI_EXPORT_ROOT`，尚無背景工作進度與自動清理。
- 車輛關聯、OCR 與人工 accepted 都是不同證據層級；本功能不把任一層宣稱為依法可開罰。

## 回滾方式

移除匯出 API／exporter／React 按鈕與 urinate optional 欄位，恢復 review verdict 驗證集合；
SQLite 舊 review 與 plate correction 資料不需修改。
