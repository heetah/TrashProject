# 對齊 RT-DETR 4-channel inference preprocessing

- 日期：2026-08-19
- 作者：Codex
- Branch：`heetah-dev`
- Commit：尚未提交（工作樹含使用者既有變更）
- 類型：fix / test / docs

## 問題背景

Production `scripts/pipeline/detect.py` 與
`/mnt/8tb_hdd/under115a/4c-yolo/run_4ch.py` 使用同一份
`best-rtdetr-4c-background.pt` 權重，但 inference input contract 不同。
Reference 先將 BGR frame 轉成 RGB，並將 temporal change map 做每幀
min-max normalization 後乘 `1.5`；production 原本傳入 BGR 並使用未
normalize 的 `2.0` gain map。Ultralytics 8.4.41 不會對 4-channel NumPy
input 自動執行 BGR-to-RGB，因此原本的前三個 channel 會以錯誤順序
直接送入模型。

## 實作內容

- 新增 `scripts/pipeline/litter/input4c.py`，集中實作 reference-compatible
  RGB + normalized temporal change map。
- 單幀 RT-DETR、batch RT-DETR、exception repair、zero-result repair 共用同一
  input builder。
- TensorRT export 後的 4-channel smoke test 改用同一 input builder；首幀
  第四通道也與 runtime 一致為全零。
- 更新 4-channel unit test，並直接驗證安裝中 Ultralytics preprocessing 後的
  final tensor channel order 為 `[R, G, B, change] / 255`。

## API／Config／Schema 變更

- 新增 `build_litter_model_input(prev_frame, curr_frame)` 共用函式。
- 沒有 CLI、environment variable 或 analysis schema 變更。
- `TRASH_CONF` production 預設仍為 `0.4`；參考程式的 `0.683` 較高，
  調高會減少 candidate recall，不作為本次漏檢修正。

## 測試證據

- `conda run -n rtdetr python -m pytest -q tests/pipeline/test_detect_4c.py`：
  `12 passed`。
- `conda run -n rtdetr python -m pytest -q tests/pipeline/test_import_smoke.py`：
  `9 passed`。
- `conda run -n rtdetr python -m pytest -q tests/pipeline/test_detect_characterization.py tests/pipeline/test_litter_tracker_4c.py`：
  `42 passed`。
- `conda run -n rtdetr python -m pytest -q tests/pipeline`：
  `163 passed, 12 skipped`。
- `conda run -n rtdetr python -m py_compile scripts/main.py scripts/export_tensorrt.py scripts/pipeline/detect.py scripts/pipeline/litter/input4c.py`：通過。
- 由 reference `run_4ch.py` AST 直接抽出 `compute_pixel_change_map()`，與新實作
  比對隨機 `72x96` frames：`reference_change_map_equal=True`。
- Production 與 reference checkpoint SHA-256 皆為
  `ec157ad5dea945a83a9b2139ca66593b2c12dd791b74ba30b72ce4b65e09c4d9`。

## 已知限制

- 當前執行環境無法看到 NVIDIA driver，`torch.cuda.is_available()` 為
  `False`，因此未在此環境重跑真實影片 GPU inference。
- Tensor-level parity 只證明 preprocessing contract 已對齊；真實 litter recall
  改善幅度仍需用明確 clips 與 reviewed labels 重跑比較。
- 本次不改變 downstream geometry、motion、holding 或 tracker confirmation gates。

## 回滾方式

回滾本版對 `scripts/pipeline/litter/input4c.py`、`scripts/pipeline/detect.py`、
`scripts/export_tensorrt.py`、`tests/pipeline/test_detect_4c.py` 與 `README.md` 的對應變更。
