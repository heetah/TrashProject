"""Fresh-observation gates for actor-dependent pipeline stages.

This module owns only gate freshness semantics.  It does not decide whether an
actor is correctly attributed, and it never treats a cached or predicted actor
as a new detector observation.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple


VEHICLE_LIKE_CLASSES = frozenset(("vehicle", "scooter"))
STALE_SOURCES = frozenset(("cache", "predicted", "prediction"))


def is_fresh_observation(
    actor: Mapping,
    *,
    accepted_classes: Iterable[str] = VEHICLE_LIKE_CLASSES,
) -> bool:
    """Return whether actor is a fresh detector observation for gate use.

    Legacy precomputed actors may omit ``observed``/``source``; those remain
    fresh for compatibility.  Explicit cache/prediction provenance always wins
    and is rejected, preventing stale state from extending a TTL.
    """
    if not isinstance(actor, Mapping):
        return False
    actor_class = str(actor.get("cls", "")).strip().lower()
    accepted = {str(value).strip().lower() for value in accepted_classes}
    if actor_class not in accepted:
        return False
    if actor.get("observed", True) is False:
        return False
    source = str(actor.get("source", "")).strip().lower()
    return source not in STALE_SOURCES


def has_fresh_observation(
    actors: Sequence[Mapping] | None,
    *,
    accepted_classes: Iterable[str] = VEHICLE_LIKE_CLASSES,
) -> bool:
    """Return whether actor sequence contains at least one fresh observation."""
    return any(
        is_fresh_observation(actor, accepted_classes=accepted_classes)
        for actor in (actors or ())
    )


def refresh_last_observation_frame(
    last_frame_index: Optional[int],
    actors: Sequence[Mapping] | None,
    frame_index: int,
    *,
    accepted_classes: Iterable[str] = VEHICLE_LIKE_CLASSES,
) -> Optional[int]:
    """Refresh gate timestamp only when a fresh actor was observed."""
    if not has_fresh_observation(actors, accepted_classes=accepted_classes):
        return None if last_frame_index is None else int(last_frame_index)
    current = int(frame_index)
    if last_frame_index is None:
        return current
    return max(int(last_frame_index), current)


def is_within_ttl(
    frame_index: int,
    last_observation_frame: Optional[int],
    ttl_frames: int,
) -> bool:
    """Check frame-domain TTL; caller supplies FPS-derived ``ttl_frames``."""
    if last_observation_frame is None:
        return False
    return int(frame_index) - int(last_observation_frame) <= max(int(ttl_frames), 0)


def batch_has_active_gate(
    actor_pairs: Sequence[Tuple[Sequence[Mapping], Sequence[Mapping]]],
    frame_indices: Sequence[int],
    last_observation_frame: Optional[int],
    ttl_frames: int,
    *,
    accepted_classes: Iterable[str] = VEHICLE_LIKE_CLASSES,
) -> bool:
    """Simulate gate freshness over a batch without mutating caller cache."""
    current = last_observation_frame
    for (_persons, vehicles), frame_index in zip(actor_pairs, frame_indices):
        current = refresh_last_observation_frame(
            current,
            vehicles,
            int(frame_index),
            accepted_classes=accepted_classes,
        )
        if is_within_ttl(int(frame_index), current, ttl_frames):
            return True
    return False


__all__ = [
    "VEHICLE_LIKE_CLASSES",
    "batch_has_active_gate",
    "has_fresh_observation",
    "is_fresh_observation",
    "is_within_ttl",
    "refresh_last_observation_frame",
]
