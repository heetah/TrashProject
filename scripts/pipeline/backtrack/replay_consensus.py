"""Research-only consensus checks for replay-selected routes.

The production resolver is deliberately not imported here.  A replay can run
under several caller-supplied perturbations (for example confidence, image
scale or codec changes), but a stable-looking top-1 route is not evidence that
the route is correct.  This module therefore returns a non-NULL route only
when every condition selects the same canonical route identity.  Any
disagreement is fail-closed to the explicit NULL route and requires review.

Route IDs are local labels and may legitimately change between reruns.  Route
identity is compared using ``(route_type, person_key, vehicle_key)`` instead;
the original route IDs are retained only as audit metadata.  No score,
likelihood, confidence or history lineage is combined by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence


ActorKey = tuple[str, int]
RouteTuple = tuple[str, ActorKey | None, ActorKey | None]
NULL_ROUTE: RouteTuple = ("null", None, None)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _actor_key(value: Any, label: str) -> ActorKey | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be null or [actor_type, track_id]")
    actor_type = _text(value[0], f"{label}.actor_type").casefold()
    track_id = value[1]
    if isinstance(track_id, bool) or not isinstance(track_id, Integral):
        raise ValueError(f"{label}.track_id must be an integer")
    return actor_type, int(track_id)


def canonical_route_tuple(route: Any) -> RouteTuple:
    """Validate and canonicalize one replay route.

    ``route`` may be a mapping with ``route_type``, ``person_key`` and
    ``vehicle_key`` fields, or a three-item tuple in that same order.  The
    route type and actor-type labels are compared case-insensitively after
    trimming whitespace.  Actor IDs remain exact integers.  The route ID is
    intentionally ignored because it is local to one resolver run.
    """

    if isinstance(route, Mapping):
        if "route_type" not in route:
            raise ValueError("route must contain route_type")
        route_type_value = route["route_type"]
        person_value = route.get("person_key")
        vehicle_value = route.get("vehicle_key")
    elif isinstance(route, (list, tuple)) and len(route) == 3:
        route_type_value, person_value, vehicle_value = route
    else:
        raise ValueError("route must be a mapping or a three-item tuple")

    route_type = _text(route_type_value, "route_type").casefold()
    person_key = _actor_key(person_value, "person_key")
    vehicle_key = _actor_key(vehicle_value, "vehicle_key")

    if route_type == "null":
        if person_key is not None or vehicle_key is not None:
            raise ValueError("NULL route must not contain person or vehicle keys")
        return NULL_ROUTE
    if person_key is None and vehicle_key is None:
        raise ValueError("non-NULL route must contain an actor key")
    if route_type == "direct_vehicle" and (
        person_key is not None or vehicle_key is None
    ):
        raise ValueError("direct_vehicle route has invalid actor keys")
    if route_type == "person" and (
        person_key is None or vehicle_key is not None
    ):
        raise ValueError("person route has invalid actor keys")
    if route_type == "person_vehicle" and (
        person_key is None or vehicle_key is None
    ):
        raise ValueError("person_vehicle route has invalid actor keys")
    return route_type, person_key, vehicle_key


def _route_id(decision: Mapping[str, Any], route: Any) -> str | None:
    """Read an optional local route ID while rejecting conflicting copies."""

    values: list[str | None] = []
    for container in (decision, route if isinstance(route, Mapping) else None):
        if container is not None and "route_id" in container:
            value = container["route_id"]
            values.append(None if value is None else _text(value, "route_id"))
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError("conflicting route_id values")
    return first


def _parse_decision(decision: Any) -> tuple[str, RouteTuple, str | None]:
    if not isinstance(decision, Mapping):
        raise ValueError("each replay decision must be a mapping")
    condition_id = _text(decision.get("condition_id"), "condition_id")
    route = decision.get("route")
    if route is None:
        # Accept a flat row as a convenience for CSV/JSONL adapters, while
        # still requiring the same strict route fields.
        if "route_type" not in decision:
            raise ValueError("decision must contain route")
        route = decision
    route_tuple = canonical_route_tuple(route)
    return condition_id, route_tuple, _route_id(decision, route)


@dataclass(frozen=True)
class ReplayConsensus:
    """Auditable result of a strict all-conditions route agreement check.

    ``safe_route`` is always a usable route tuple: it is the agreed route when
    ``consensus`` is true and the explicit NULL route otherwise.  The latter
    is a review hand-off, not a claim that NULL is the ground-truth route.
    ``consensus_route`` is ``None`` on disagreement so a caller cannot confuse
    a fail-closed fallback with a unanimous route.
    """

    consensus: bool
    consensus_route: RouteTuple | None
    safe_route: RouteTuple
    reason: str
    condition_ids: tuple[str, ...]
    observed_routes: tuple[RouteTuple, ...]
    decision_route_ids: tuple[str | None, ...]
    distinct_routes: tuple[RouteTuple, ...]

    @property
    def manual_review_required(self) -> bool:
        return not self.consensus

    @property
    def condition_count(self) -> int:
        return len(self.condition_ids)

    @property
    def distinct_route_count(self) -> int:
        return len(self.distinct_routes)


def replay_route_consensus(
    decisions: Sequence[Mapping[str, Any]],
) -> ReplayConsensus:
    """Return a fail-closed consensus result for caller-supplied replays.

    Every decision must have a unique non-empty ``condition_id`` and a valid
    route.  Input rows are sorted by condition ID for deterministic audit
    output.  No route is selected by plurality: one disagreement, including a
    disagreement with NULL, produces ``safe_route == NULL_ROUTE`` and sets
    ``manual_review_required``.
    """

    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise ValueError("decisions must be a non-empty sequence")
    if not decisions:
        raise ValueError("decisions must be a non-empty sequence")

    parsed = [_parse_decision(decision) for decision in decisions]
    condition_ids = [item[0] for item in parsed]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition_id values must be unique")
    parsed.sort(key=lambda item: item[0])

    sorted_condition_ids = tuple(item[0] for item in parsed)
    observed_routes = tuple(item[1] for item in parsed)
    route_ids = tuple(item[2] for item in parsed)
    distinct_routes = tuple(sorted(set(observed_routes), key=repr))
    consensus = len(distinct_routes) == 1
    consensus_route = distinct_routes[0] if consensus else None
    return ReplayConsensus(
        consensus=consensus,
        consensus_route=consensus_route,
        safe_route=consensus_route if consensus else NULL_ROUTE,
        reason="all_conditions_agree" if consensus else "route_disagreement",
        condition_ids=sorted_condition_ids,
        observed_routes=observed_routes,
        decision_route_ids=route_ids,
        distinct_routes=distinct_routes,
    )


# A descriptive alias keeps call sites readable when the result is used as a
# summary rather than as a route selector.  Both names are pure research APIs.
summarize_replay_consensus = replay_route_consensus


__all__ = [
    "ActorKey",
    "NULL_ROUTE",
    "ReplayConsensus",
    "RouteTuple",
    "canonical_route_tuple",
    "replay_route_consensus",
    "summarize_replay_consensus",
]
