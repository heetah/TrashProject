"""
TDD tests for 4c-specific detect.py logic.

Tests define expected behaviour BEFORE implementation is fixed.
Run with:
    conda run -n rtdetr python -m pytest tests/pipeline/test_detect_4c.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# compute_pixel_change_map
# ─────────────────────────────────────────────────────────────────────────────

def _import_fn():
    from detect import compute_pixel_change_map
    return compute_pixel_change_map


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

    def test_near_static_scene_not_amplified(self):
        """Near-static scene (1-2 pixel diff) must stay near-zero, NOT amplified.

        The old implementation used cv2.normalize which maps a tiny noise range
        like [0,2] to [0,255], destroying magnitude information.
        4c model relies on the change map having meaningful magnitude:
        static scene → dark channel, moving objects → bright channel.

        Max pixel value in near-static scene must be < 15.
        """
        fn = _import_fn()
        base = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy = base.copy()
        noisy[10, 10, 0] = 130   # 2-unit diff in one pixel/channel
        noisy[20, 20, 1] = 129   # 1-unit diff
        noisy[50, 50, 2] = 130   # 2-unit diff

        result = fn(base, noisy)
        assert result.max() < 15, (
            f"Near-static scene max={result.max()}, expected < 15. "
            "cv2.normalize is likely amplifying sensor noise to 0-255."
        )

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

    def test_magnitude_preserved_not_normalized(self):
        """Larger diffs must produce proportionally brighter output.

        If diff_A is 10 and diff_B is 100, result_B should be ~10x result_A.
        With cv2.normalize both would be mapped to the same range (0-255),
        destroying proportionality. After the fix, raw magnitude is preserved.
        """
        fn = _import_fn()
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        curr_small = prev.copy()
        curr_small[50, 50, 0] = 20   # small diff = 20
        curr_large = prev.copy()
        curr_large[50, 50, 0] = 200  # large diff = 200

        res_small = fn(prev, curr_small)
        res_large = fn(prev, curr_large)

        # Large diff should produce a noticeably brighter pixel
        assert res_large[50, 50] > res_small[50, 50] * 3, (
            f"Large diff ({res_large[50,50]}) should be much brighter than "
            f"small diff ({res_small[50,50]}). Normalize destroys proportionality."
        )
