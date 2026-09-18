# -*- coding: utf-8 -*-
"""PipelineConfig 單元測試:預設值 = 原 main.py 固定值;env 覆寫;無 env 等同預設。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.config import PROJECT_ROOT, PipelineConfig, load_env_file

# 既有參數維持翻新前 main.py 固定值；新參數固定 production 預設，避免未設 env 時漂移。
EXPECTED_DEFAULTS = {
    "pose_model_path": str(PROJECT_ROOT / "modules_weight/yolo26x-pose.pt"),
    "stgcn_weight_path": str(PROJECT_ROOT / "modules_weight/best_stgcn_0623.pth"),
    "stgcn_config_path": str(
        PROJECT_ROOT / "mmaction2/configs/skeleton/stgcnpp/custom_trash_stgcnpp.py"
    ),
    "bbox_model_path": str(PROJECT_ROOT / "modules_weight/best-yolo-seg_v3.pt"),
    "trash_model_path": str(
        PROJECT_ROOT / "modules_weight/best-rtdetr-4c-background.pt"
    ),
    "bbox_model_path_batch": str(
        PROJECT_ROOT / "modules_weight/batch/best-yolo-seg_v3.pt"
    ),
    "trash_model_path_batch": str(
        PROJECT_ROOT / "modules_weight/best-rtdetr-4c-background.pt"
    ),
    "prefer_tensorrt": True,
    "rtdetr_enabled": True,
    "action_device": None,
    "batch_size": 8,
    "pipeline_queue_size": 8,
    "prepare_4c_in_reader": True,
    "yolo_seg_frame_skip": 2,
    "actor_mode": "predict",
    "rtdetr_zero_repair": "off",
    "rtdetr_imgsz": None,
    "bbox_conf": 0.45,
    "trash_conf": 0.4,
    "actor_track_iou": 0.3,
    "moving_threshold": 0.25,
    "core_moving_threshold": 0.3,
    "motion_mask_mode": "temporal",
    "fg_mask_scale": 0.5,
    "motion_diff_threshold": 10,
    "motion_dilate_iterations": 2,
    "motion_blur_kernel": 0,
    "motion_open_kernel": 3,
    "motion_open_iterations": 0,
    "motion_close_kernel": 0,
    "motion_close_iterations": 0,
    "motion_min_component_area": 4,
    "motion_min_largest_component_ratio": 0.25,
    "motion_mog2_detect_shadows": True,
    "violator_display_ttl": 60,
    "violator_display_max_jump": 80.0,
    "action_threshold": 0.5,
    "action_window": 100,
    "urination_window_sec": 8.0,
    "urination_min_sec": 5.0,
    "litter_distance_threshold": 250,
    "output_root": ".",
    "video_hw_accel": "any",
    "video_hw_device": None,
    "video_capture_buffer_size": 8,
    "video_read_threads": 0,
    "writer_queue_size": 16,
    "writer_preset": "fast",
    "writer_crf": 23,
    "writer_encoder": "auto",
    "ffmpeg_bin": None,
    "profile_enabled": True,
}


def test_defaults_match_legacy_literals():
    cfg = PipelineConfig()
    for k, v in EXPECTED_DEFAULTS.items():
        assert getattr(cfg, k) == v, k


def test_from_env_without_env_equals_defaults(monkeypatch):
    for name in (
        "PIPELINE_BATCH", "PIPELINE_QUEUE_SIZE", "PIPELINE_PREPARE_4C",
        "BBOX_CONF", "TRASH_CONF", "ACTION_WINDOW", "VIOLATOR_DISPLAY_TTL",
        "WRITER_CRF", "MODEL_BBOX_PATH", "MOTION_DIFF_THRESHOLD",
        "VIDEO_HW_DEVICE",
        "RTDETR_IMGSZ",
    ):
        monkeypatch.delenv(name, raising=False)
    assert PipelineConfig.from_env() == PipelineConfig()


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("PIPELINE_BATCH", "4")
    monkeypatch.setenv("PIPELINE_QUEUE_SIZE", "12")
    monkeypatch.setenv("PIPELINE_PREPARE_4C", "0")
    monkeypatch.setenv("BBOX_CONF", "0.6")
    monkeypatch.setenv("WRITER_PRESET", "medium")
    monkeypatch.setenv("ACTION_WINDOW", "150")
    monkeypatch.setenv("MODEL_BBOX_PATH", "modules_weight/custom-actor.pt")
    monkeypatch.setenv("MOTION_DIFF_THRESHOLD", "17")
    monkeypatch.setenv("VIDEO_HW_DEVICE", "1")
    monkeypatch.setenv("RTDETR_IMGSZ", "1536")
    cfg = PipelineConfig.from_env()
    assert cfg.batch_size == 4
    assert cfg.pipeline_queue_size == 12
    assert cfg.prepare_4c_in_reader is False
    assert cfg.bbox_conf == 0.6
    assert cfg.writer_preset == "medium"
    assert cfg.action_window == 150
    assert cfg.bbox_model_path == str(
        (PROJECT_ROOT / "modules_weight/custom-actor.pt").resolve()
    )
    assert cfg.motion_diff_threshold == 17
    assert cfg.video_hw_device == 1
    assert cfg.rtdetr_imgsz == 1536
    # 未覆寫者維持預設
    assert cfg.trash_conf == 0.4


def test_from_env_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("PIPELINE_BATCH", "not-an-int")
    monkeypatch.setenv("BBOX_CONF", "")
    cfg = PipelineConfig.from_env()
    assert cfg.batch_size == 8   # 無效 → 回退預設
    assert cfg.bbox_conf == 0.45  # 空字串 → 回退預設


def test_load_env_file_preserves_exported_values_and_parses_quotes(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local config\n"
        "PIPELINE_BATCH=4\n"
        "export MODEL_TRASH_PATH='modules_weight/custom trash.pt'\n"
        "INVALID-KEY=ignored\n",
        encoding="utf-8",
    )
    values = {"PIPELINE_BATCH": "2"}

    loaded = load_env_file(env_path, values)

    assert loaded == env_path.resolve()
    assert values["PIPELINE_BATCH"] == "2"
    assert values["MODEL_TRASH_PATH"] == "modules_weight/custom trash.pt"
    assert "INVALID-KEY" not in values


def test_model_paths_for_batch_use_separate_env_paths(monkeypatch):
    monkeypatch.setenv("MODEL_BBOX_PATH", "models/actor-single.pt")
    monkeypatch.setenv("MODEL_TRASH_PATH", "models/trash-single.pt")
    monkeypatch.setenv("MODEL_BBOX_PATH_BATCH", "models/actor-batch.pt")
    monkeypatch.setenv("MODEL_TRASH_PATH_BATCH", "models/trash-batch.pt")
    cfg = PipelineConfig.from_env()

    assert tuple(map(Path, cfg.model_paths_for_batch(1))) == (
        PROJECT_ROOT / "models/actor-single.pt",
        PROJECT_ROOT / "models/trash-single.pt",
    )
    assert tuple(map(Path, cfg.model_paths_for_batch(8))) == (
        PROJECT_ROOT / "models/actor-batch.pt",
        PROJECT_ROOT / "models/trash-batch.pt",
    )


def test_action_pose_and_normalization_thresholds_use_env(monkeypatch):
    from pipeline.action import STGCNActionModule

    monkeypatch.setenv("ACTION_POSE_CONF", "0.42")
    monkeypatch.setenv("ACTION_BBOX_NORM_PAD", "0.2")
    monkeypatch.setenv("ACTION_BBOX_NORM_CONF", "0.55")
    monkeypatch.setattr(STGCNActionModule, "_load_stgcn", lambda *args, **kwargs: None)

    module = STGCNActionModule(
        pose_model_path=None,
        stgcn_weight_path=None,
        stgcn_config_path=None,
        device="cpu",
    )

    assert module.pose_conf == 0.42
    assert module.bbox_norm_pad == 0.2
    assert module.bbox_norm_conf == 0.55


def test_plate_model_path_and_confidence_settings_use_env(monkeypatch):
    from pipeline import plate

    monkeypatch.setenv("PLATE_MODEL_PATH", "models/custom-plate.pt")
    monkeypatch.setenv("PLATE_DETECT_CONF", "1.5")

    assert plate._plate_model_path() == str(
        (PROJECT_ROOT / "models/custom-plate.pt").resolve()
    )
    assert plate._confidence_env("PLATE_DETECT_CONF", 0.6) == 1.0
