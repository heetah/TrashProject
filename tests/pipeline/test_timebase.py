import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.timebase import SourceClock, resolve_source_fps


def test_source_fps_preserves_fractional_rate():
    assert resolve_source_fps(29.97) == pytest.approx(29.97)
    assert SourceClock(29.97).seconds_at(2997) == pytest.approx(100.0)


def test_invalid_source_fps_uses_positive_fallback():
    assert resolve_source_fps(0, fallback=25.0) == 25.0
    assert resolve_source_fps(float("nan"), fallback=25.0) == 25.0


def test_invalid_fallback_is_rejected():
    with pytest.raises(ValueError):
        resolve_source_fps(0, fallback=0)


def test_clock_delta_preserves_frame_gap_and_units():
    clock = SourceClock(10.0)
    assert clock.delta_seconds(8, 13) == pytest.approx(0.5)
    assert math.isfinite(clock.seconds_at(0))
