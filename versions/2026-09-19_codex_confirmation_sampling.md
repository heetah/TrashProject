# Deterministic confirmation sampling

- Date: 2026-09-19
- Author: Codex
- Branch: `heetah-dev`
- Base: working tree after `quarantine_evidence_module`
- Status: implementation in working tree; not committed

## Problem

模組化 reason 已接入 tracker，但舊的 406-case artifact 沒有 reason 欄位。
若直接從 candidate/confirmed aggregate count 猜測原因，會把診斷誤當成標註。

## Change

- Added `scripts/pipeline/confirmation_sampling.py`。
- 依 `clip_id` 排序並固定間距抽樣，不依賴 random seed。
- 只讀取明確 `confirmation_reason`／`confirmation_evidence`；舊 schema 使用
  `unavailable_legacy_artifact` marker。
- 若 confirmation `reason=supported`，抽樣群組改用明確 `rule`（例如
  `by_trajectory`），避免所有成功案例被壓成同一 generic label。
- 報告同時保留 error stage、candidate count、confirmed count，供逐案人工抽查。
- confirmed event 現在把 confirmation/quarantine evidence 凍結到事件 JSON，讓後續
  artifact 萃取可讀到明確 reason，而不必從 aggregate count 猜測。

## Validation

- `tests/pipeline/test_confirmation_sampling.py`: 5 passed。
- `tests/pipeline/test_events.py` covered reason propagation；compact analysis 欄位維持
  backward-compatible。
- 對 `artifacts/backtrack_testcase_production_20260919/case_results.csv` 做 12-case
  sample：population=406、sample=12；12/12 reason 為
  `unavailable_legacy_artifact`，不推斷舊資料原因。
- Sample stage：9 `not_recorded`、3 `tracker_confirmation_or_quarantine`。

## Limits

- 舊 artifact 仍需重新執行 production pipeline 才能取得逐案 reason。
- Counts 是診斷觀測，不是 reviewed label、precision、recall 或 attribution accuracy。
- 無 real video 可用時只做 fixture／artifact sampling。

## Rollback

移除 sampling helper 與測試即可；不影響 production inference 或模型權重。
