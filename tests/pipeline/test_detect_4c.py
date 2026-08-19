"""
TDD tests for 4c-specific detect.py logic.

Tests define expected behaviour BEFORE implementation is fixed.
Run with:
    conda run -n rtdetr python -m pytest tests/pipeline/test_detect_4c.py -v
"""
import sys
import os
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import cv2
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# compute_pixel_change_map
# ─────────────────────────────────────────────────────────────────────────────

def _import_fn():
    from detect import compute_pixel_change_map
    return compute_pixel_change_map


def _import_input_builder():
    from pipeline.litter.input4c import build_litter_model_input
    return build_litter_model_input


class TestComputePixelChangeMap:

    def test_first_frame_returns_all_zeros(self):
        """prev_frame=None (first frame) → all-zero output."""
        fn = _import_fn()
        curr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = fn(None, curr)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert result.max() == 0, "First frame must return all-zero change map"

    def test_output_shape_is_hw(self):
        """Output is (H, W) — no channel dim."""
        fn = _import_fn()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = fn(frame, frame.copy())
        assert result.shape == (480, 640)

    def test_output_dtype_uint8(self):
        """Output dtype must be uint8."""
        fn = _import_fn()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = fn(frame, frame.copy())
        assert result.dtype == np.uint8

    def test_near_static_peak_matches_reference_normalization(self):
        """Reference run_4ch.py min-max normalizes even a small isolated diff."""
        fn = _import_fn()
        base = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy = base.copy()
        noisy[10, 10, 0] = 130   # 2-unit diff in one pixel/channel
        noisy[20, 20, 1] = 129   # 1-unit diff
        noisy[50, 50, 2] = 130   # 2-unit diff

        result = fn(base, noisy)
        assert result.max() == 255

    def test_moving_object_produces_bright_pixels(self):
        """A clearly moving bright object must produce high values in the change map.

        A white box moves 20px diagonally: the changed region should have
        pixel values > 150.
        """
        fn = _import_fn()
        h, w = 200, 200
        prev = np.zeros((h, w, 3), dtype=np.uint8)
        curr = np.zeros((h, w, 3), dtype=np.uint8)
        prev[50:80, 50:80] = 255   # white box at (50,50)
        curr[70:100, 70:100] = 255  # moved 20px diagonally

        result = fn(prev, curr)
        motion_max = result[50:100, 50:100].max()
        assert motion_max > 150, (
            f"Motion region max={motion_max}, expected > 150. "
            "Change map must be bright where objects moved."
        )

    def test_static_scene_identical_frames_zero(self):
        """Identical frames → change map must be exactly zero everywhere."""
        fn = _import_fn()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = fn(frame, frame.copy())
        assert result.max() == 0, (
            "Identical frames must produce all-zero change map. "
            f"Got max={result.max()}."
        )

    def test_per_frame_peak_is_normalized_like_reference(self):
        """Separate frames with different amplitudes both reach the reference peak."""
        fn = _import_fn()
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        curr_small = prev.copy()
        curr_small[50, 50, 0] = 20   # small diff = 20
        curr_large = prev.copy()
        curr_large[50, 50, 0] = 200  # large diff = 200

        res_small = fn(prev, curr_small)
        res_large = fn(prev, curr_large)

        assert res_small[50, 50] == 255
        assert res_large[50, 50] == 255

    def test_matches_run_4ch_reference_formula(self):
        """Pin the exact change-map operations used by run_4ch.py."""
        fn = _import_fn()
        rng = np.random.default_rng(42)
        prev = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        curr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)

        diff = cv2.absdiff(prev, curr)
        expected = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        expected = cv2.normalize(expected, None, 0, 255, cv2.NORM_MINMAX)
        expected = np.clip(expected * 1.5, 0, 255).astype(np.uint8)

        np.testing.assert_array_equal(fn(prev, curr), expected)


class TestLitterModelInput:

    def _frames(self):
        prev = np.zeros((2, 2, 3), dtype=np.uint8)
        curr = np.empty_like(prev)
        curr[:] = [30, 20, 10]  # BGR
        curr[0, 0] = [230, 120, 60]
        return prev, curr

    def test_source_is_rgb_plus_reference_change_map(self):
        prev, curr = self._frames()
        source = _import_input_builder()(prev, curr)
        change_map = _import_fn()(prev, curr)

        assert source.shape == (2, 2, 4)
        assert source.dtype == np.uint8
        np.testing.assert_array_equal(source[..., :3], curr[..., ::-1])
        np.testing.assert_array_equal(source[..., 3], change_map)

    def test_ultralytics_final_tensor_preserves_rgb_change_order(self):
        """Validate the tensor after the installed Ultralytics preprocessing boundary."""
        torch = pytest.importorskip("torch")
        predictor_module = pytest.importorskip("ultralytics.engine.predictor")
        prev, curr = self._frames()
        source = _import_input_builder()(prev, curr)

        predictor = predictor_module.BasePredictor(
            overrides={"imgsz": 2, "rect": False}
        )
        predictor.model = SimpleNamespace(
            fp16=False,
            format="pt",
            dynamic=False,
            stride=32,
        )
        predictor.device = torch.device("cpu")
        predictor.imgsz = (2, 2)

        tensor = predictor.preprocess([source])

        assert tuple(tensor.shape) == (1, 4, 2, 2)
        expected = torch.from_numpy(
            source.transpose(2, 0, 1).copy()
        ).float().unsqueeze(0) / 255.0
        assert torch.equal(tensor, expected)

    def test_batched_runtime_passes_reference_inputs_to_predict(self):
        from pipeline.detect import _run_batched_trash_predict

        class CaptureModel:
            def __init__(self):
                self.source = None

            def predict(self, source, **_kwargs):
                self.source = source
                return [SimpleNamespace(boxes=[]) for _ in source]

        prev, first = self._frames()
        second = np.roll(first, 1, axis=1)
        model = CaptureModel()

        _run_batched_trash_predict(
            model,
            [first, second],
            trash_conf=0.4,
            export_batch_size=2,
            zero_repair="off",
            prev_frames=[prev, first],
        )

        assert isinstance(model.source, list)
        np.testing.assert_array_equal(
            model.source[0], _import_input_builder()(prev, first)
        )
        np.testing.assert_array_equal(
            model.source[1], _import_input_builder()(first, second)
        )

    def test_tensorrt_smoke_uses_same_input_contract(self):
        from export_tensorrt import _build_4ch_frames

        prev, first = self._frames()
        second = np.roll(first, 1, axis=0)
        built = _build_4ch_frames([first, second])

        np.testing.assert_array_equal(
            built[0], _import_input_builder()(None, first)
        )
        np.testing.assert_array_equal(
            built[1], _import_input_builder()(first, second)
        )
        assert not built[0][..., 3].any()
