# -*- coding: utf-8 -*-
"""Async reader 的 PreparedFrame、順序與 4-channel 跨 batch 邊界測試。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.infra.video_io import AsyncVideoFrameReader
from pipeline.litter.input4c import build_litter_model_input
from pipeline.profiling import PipelineProfiler


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class _FakeMotionMasker:
    def build(self, frame, _profiler):
        return np.full(frame.shape[:2], int(frame[0, 0, 0]), dtype=np.uint8)


def _frames():
    frames = []
    for value in (10, 20, 40):
        frame = np.full((4, 5, 3), value, dtype=np.uint8)
        frame[0, 0, 0] = value + 1
        frames.append(frame)
    return frames


def test_prepared_reader_preserves_order_and_previous_frame_across_batches():
    frames = _frames()
    capture = _FakeCapture(frames)
    reader = AsyncVideoFrameReader(
        capture,
        _FakeMotionMasker(),
        PipelineProfiler(enabled=True),
        queue_size=2,
        litter_input_builder=build_litter_model_input,
    )
    try:
        first_batch = reader.read_prepared_batch(2)
        second_batch = reader.read_prepared_batch(2)
        assert reader.read_prepared_batch(2) == []
    finally:
        reader.close()

    packets = first_batch + second_batch
    assert [packet.index for packet in packets] == [0, 1, 2]
    for index, packet in enumerate(packets):
        np.testing.assert_array_equal(packet.source_bgr, frames[index])
        np.testing.assert_array_equal(
            packet.litter_model_input,
            build_litter_model_input(frames[index - 1] if index else None, frames[index]),
        )
    assert capture.released is True


def test_legacy_read_batch_keeps_two_column_contract():
    frames = _frames()[:1]
    reader = AsyncVideoFrameReader(
        _FakeCapture(frames),
        _FakeMotionMasker(),
        PipelineProfiler(enabled=False),
        queue_size=1,
    )
    try:
        source_frames, masks = reader.read_batch(1)
    finally:
        reader.close()

    np.testing.assert_array_equal(source_frames[0], frames[0])
    assert masks[0].shape == frames[0].shape[:2]
