from __future__ import annotations

import pytest

from pipeline.backtrack.replay_consensus import (
    NULL_ROUTE,
    canonical_route_tuple,
    replay_route_consensus,
)


def _decision(condition_id, route_id, route_type="direct_vehicle", vehicle=7):
    return {
        "condition_id": condition_id,
        "route": {
            "route_id": route_id,
            "route_type": route_type,
            "person_key": None,
            "vehicle_key": ["vehicle", vehicle] if vehicle is not None else None,
        },
    }


def test_unanimous_route_ignores_local_route_ids_but_keeps_audit() -> None:
    result = replay_route_consensus([
        _decision("jpeg_q50", "direct:old", vehicle=7),
        _decision("base", "direct:new", vehicle=7),
    ])

    assert result.consensus is True
    assert result.consensus_route == ("direct_vehicle", None, ("vehicle", 7))
    assert result.safe_route == result.consensus_route
    assert result.condition_ids == ("base", "jpeg_q50")
    assert result.decision_route_ids == ("direct:new", "direct:old")
    assert result.manual_review_required is False


def test_route_disagreement_fails_closed_to_explicit_null() -> None:
    result = replay_route_consensus([
        _decision("base", "direct:7", vehicle=7),
        _decision("scale512", "null", route_type="NULL", vehicle=None),
    ])

    assert result.consensus is False
    assert result.consensus_route is None
    assert result.safe_route == NULL_ROUTE
    assert result.manual_review_required is True
    assert result.reason == "route_disagreement"
    assert result.distinct_route_count == 2
    assert result.observed_routes[0][0] == "direct_vehicle"
    assert result.observed_routes[1] == NULL_ROUTE


def test_unanimous_null_is_preserved_as_a_real_consensus() -> None:
    result = replay_route_consensus([
        _decision("base", "null-a", route_type="null", vehicle=None),
        _decision("scale", "null-b", route_type="NULL", vehicle=None),
    ])

    assert result.consensus is True
    assert result.consensus_route == NULL_ROUTE
    assert result.safe_route == NULL_ROUTE
    assert result.manual_review_required is False


def test_input_order_does_not_change_audit_result() -> None:
    first = replay_route_consensus([
        _decision("z", "r-z", vehicle=3),
        _decision("a", "r-a", vehicle=3),
    ])
    second = replay_route_consensus([
        _decision("a", "r-a", vehicle=3),
        _decision("z", "r-z", vehicle=3),
    ])
    assert first == second


@pytest.mark.parametrize(
    "decisions,match",
    [
        ([], "non-empty"),
        ([_decision("same", "a"), _decision("same", "b")], "unique"),
        ([{"route": _decision("x", "a")["route"]}], "condition_id"),
        ([{"condition_id": "x"}], "route"),
    ],
)
def test_missing_or_duplicate_conditions_fail_closed(decisions, match) -> None:
    with pytest.raises(ValueError, match=match):
        replay_route_consensus(decisions)


@pytest.mark.parametrize(
    "route,match",
    [
        ({"person_key": None, "vehicle_key": ["vehicle", 1]}, "route_type"),
        ({"route_type": "null", "vehicle_key": ["vehicle", 1]}, "NULL"),
        ({"route_type": "direct_vehicle", "vehicle_key": ["vehicle", 1.5]}, "integer"),
        ({"route_type": "direct_vehicle", "person_key": ["person", 1]}, "direct_vehicle"),
        (("direct_vehicle", None), "mapping or a three-item"),
    ],
)
def test_malformed_route_fails_closed(route, match) -> None:
    with pytest.raises(ValueError, match=match):
        canonical_route_tuple(route)


def test_flat_decision_form_is_supported_without_relaxing_route_validation() -> None:
    result = replay_route_consensus([
        {
            "condition_id": "base",
            "route_id": "direct:7",
            "route_type": "direct_vehicle",
            "person_key": None,
            "vehicle_key": ["vehicle", 7],
        },
    ])
    assert result.consensus_route == ("direct_vehicle", None, ("vehicle", 7))
