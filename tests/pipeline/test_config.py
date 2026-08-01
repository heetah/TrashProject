# -*- coding: utf-8 -*-
"""PipelineConfig 單元測試:預設值 = 原 main.py 固定值;env 覆寫;無 env 等同預設。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.config import PipelineConfig

# 這些預設必須等於翻新前 main.py 內的固定字面值,確保 config 化「行為不變」。
EXPECTED_DEFAULTS = {
    "batch_size": 8,
    "yolo_seg_frame_skip": 2,
    "actor_mode": "predict",
    "rtdetr_zero_repair": "off",
    "bbox_conf": 0.45,
    "trash_conf": 0.4,
    "actor_track_iou": 0.3,
    "moving_threshold": 0.25,
    "core_moving_threshold": 0.3,
    "violator_display_ttl": 60,
    "violator_display_max_jump": 80.0,
    "action_threshold": 0.5,
    "action_window": 100,
    "urination_window_sec": 8.0,
    "urination_min_sec": 5.0,
    "litter_distance_threshold": 250,
    "writer_queue_size": 16,
    "writer_preset": "fast",
    "writer_crf": 23,
    "writer_encoder": "auto",
}


def test_defaults_match_legacy_literals():
    cfg = PipelineConfig()
    for k, v in EXPECTED_DEFAULTS.items():
        assert getattr(cfg, k) == v, k


def test_from_env_without_env_equals_defaults(monkeypatch):
    for name in ("PIPELINE_BATCH", "BBOX_CONF", "TRASH_CONF", "ACTION_WINDOW",
                 "VIOLATOR_DISPLAY_TTL", "WRITER_CRF"):
        monkeypatch.delenv(name, raising=False)
    assert PipelineConfig.from_env() == PipelineConfig()


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("PIPELINE_BATCH", "4")
    monkeypatch.setenv("BBOX_CONF", "0.6")
    monkeypatch.setenv("WRITER_PRESET", "medium")
    monkeypatch.setenv("ACTION_WINDOW", "150")
    cfg = PipelineConfig.from_env()
    assert cfg.batch_size == 4
    assert cfg.bbox_conf == 0.6
    assert cfg.writer_preset == "medium"
    assert cfg.action_window == 150
    # 未覆寫者維持預設
    assert cfg.trash_conf == 0.4


def test_from_env_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("PIPELINE_BATCH", "not-an-int")
    monkeypatch.setenv("BBOX_CONF", "")
    cfg = PipelineConfig.from_env()
    assert cfg.batch_size == 8   # 無效 → 回退預設
    assert cfg.bbox_conf == 0.45  # 空字串 → 回退預設
