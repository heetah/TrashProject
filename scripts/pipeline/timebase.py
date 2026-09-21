"""Source-video time helpers.

Frame-domain logic must use source FPS, not a rounded display value.  This
module keeps conversion rules small and testable before wider timestamp wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def resolve_source_fps(raw_fps, fallback: float = 30.0) -> float:
    """Return finite positive source FPS without rounding it."""
    try:
        value = float(raw_fps)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0.0:
        value = float(fallback)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("fallback FPS must be finite and positive")
    return value


@dataclass(frozen=True)
class SourceClock:
    """Constant-FPS source clock; PTS/VFR support remains explicit future work."""

    fps: float

    def __post_init__(self):
        object.__setattr__(self, "fps", resolve_source_fps(self.fps))

    def seconds_at(self, frame_index: int) -> float:
        return float(frame_index) / self.fps

    def delta_seconds(self, first_frame: int, second_frame: int) -> float:
        return (int(second_frame) - int(first_frame)) / self.fps


__all__ = ["SourceClock", "resolve_source_fps"]
