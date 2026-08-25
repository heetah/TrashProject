# Production Pipeline 目錄與團隊規則統整

- 日期：2026-08-01
- 作者：team
- Branch：`refactor/pipeline-package`
- 基準 commit：`85693eb`
- 類型：`chore(repo)`、`docs(architecture)`

## 問題背景

原 production pipeline 位於 `scripts-old-test/`，測試與程式碼混在同一資料夾；團隊也需要明確區分正式程式、個人開發 branch、人類文件與 AI Agent Instruction。

## 實作內容

- 將 production pipeline 由 `scripts-old-test/` 改名為 `scripts/`，保留既有內部模組架構。
- 將 pipeline 測試抽至根目錄 `tests/pipeline/`。
- `dev/heetah`、`dev/pgdr` 作為兩位開發者的個人整合 branch，不建立本機個人程式碼資料夾。
- 移除對已不存在 `mmpose-rtmw/` 的文件依賴，YOLO-Pose 維持唯一 production keypoint source。
- 以根目錄 `AGENTS.md` 作為唯一 AI Coding Agent Instruction。
- 重寫 `README.md`，提供人類開發者目前架構、執行方式、責任邊界與 Git 流程。

## API／Config／Schema 變更

- Python model/event schema 無變更。
- Production entrypoint 路徑改為 `scripts/main.py`。
- Test path 改為 `tests/pipeline/`。
- Runtime 參數仍由既有 `PipelineConfig` 與環境變數控制。

## 測試證據

此版本已通過：

```bash
conda run -n rtdetr python -m py_compile \
  scripts/main.py scripts/pipeline/action.py \
  scripts/pipeline/detect.py scripts/pipeline/litter_tracker.py
```

```text
tests/pipeline: 124 passed, 12 skipped
tests/test_litter_regression.py: 12 passed
git diff --check: passed
targeted py_compile: passed
```

12 個 skip 包含需 `RUN_PIPELINE_TESTS=1` 的重型影片案例，以及此 checkout
未提供 `tools/apply_backtrack_ai_proposals.py`、
`tools/render_backtrack_annotation_previews.py`、
`tools/backtrack_annotations.py` 時自動跳過的 optional CLI tests。

## 已知限制

- `scripts-old-stable/` 暫時保留為 rollback/reference，不是 production。
- 模型、影片、輸出與研究 artifacts 仍是本機大型資料，不納入一般 Git。

## 回滾方式

回滾此結構 commit，即可恢復原 `scripts-old-test/` 與其內部測試路徑；不要手動複製兩套 production 程式。
