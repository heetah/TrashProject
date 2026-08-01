# -*- coding: utf-8 -*-
"""infra 子套件拆分的 call-level 冒煙測試(GPU-free)。

模組拆分最大的風險:函式體引用的 module-global 被留在別的子模組 → 只在「呼叫時」才
NameError,compile 與純 import 都抓不到。本測試實際「呼叫」跨子模組的純函式,並確認舊
路徑 `from pipeline.infra import ...`(含底線名稱)仍解析到同一物件。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pipeline.infra as infra
from pipeline.infra import constants, video_io, motion, models
from pipeline.profiling import PipelineProfiler


def test_reexport_identity_and_underscore_names():
    # main.py 從 pipeline.infra import 大量底線前綴名稱;必須指到子模組同一物件。
    assert infra._estimate_actor_batch_size is models._estimate_actor_batch_size
    assert infra.MotionMaskBuilder is motion.MotionMaskBuilder
    assert infra.AsyncFFmpegVideoWriter is video_io.AsyncFFmpegVideoWriter
    assert infra._resolve_video_path is video_io._resolve_video_path
    assert infra.SUPPORTED_BATCH_SIZES == constants.SUPPORTED_BATCH_SIZES
    assert infra.DEFAULT_READER_QUEUE_SIZE == constants.DEFAULT_READER_QUEUE_SIZE


def test_models_pure_functions_execute():
    # 呼叫(非只 import)→ 抓函式體內遺漏的跨模組引用。
    assert infra._estimate_actor_batch_size(8, 2) == 4
    assert infra._round_supported_batch_size(3) == 4
    assert infra._engine_batch_size_from_path("m_b8.engine") == 8
    assert infra._engine_batch_size_from_path("m.pt", fallback=2) == 2
    assert infra._model_path_candidates("/x/y.pt", prefer_engine=False, batch_size=1) == ["/x/y.pt"]
    dummy = infra._batched_dummy_frame(2)
    assert isinstance(dummy, list) and len(dummy) == 2 and dummy[0].shape == (640, 640, 3)


def test_video_io_pure_functions_execute():
    assert infra._odd_kernel_size(4) == 5
    assert infra._odd_kernel_size(2) == 0
    assert infra._nvenc_preset_from_generic("fast") == "p2"
    args = infra._build_ffmpeg_video_encoder_args("libx264", preset="fast", crf=23)
    assert args[:2] == ["-vcodec", "libx264"] and "-crf" in args
    nvenc = infra._build_ffmpeg_video_encoder_args("h264_nvenc", preset="fast", crf=23)
    assert nvenc[:2] == ["-vcodec", "h264_nvenc"] and "-cq" in nvenc


def test_motion_mask_builder_runs_end_to_end():
    # 實際 build 兩幀 → 走過 _odd_kernel_size + 常數 + cleanup,確認跨模組引用完整。
    prof = PipelineProfiler(enabled=True)
    builder = infra.MotionMaskBuilder(mode="temporal")
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    m1 = builder.build(frame, prof)              # 首幀 → 全前景
    assert m1.shape[:2] == (30, 40)              # scale_factor 0.5
    frame2 = frame.copy()
    frame2[10:20, 10:20] = 255                    # 製造動態
    m2 = builder.build(frame2, prof)
    assert m2.dtype == np.uint8 and m2.max() == 255


def test_resolve_video_path_prefers_resources(tmp_path, monkeypatch):
    # 絕對路徑原樣返回;相對純檔名找不到時回退第一候選(resources/<name>)。
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"x")
    assert infra._resolve_video_path(str(p)) == str(p)
    assert infra._resolve_video_path("no_such_clip.mp4").endswith("no_such_clip.mp4")
