import math
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.backtrack.flow import RouteCandidate, solve_event_routes


def test_same_vehicle_can_be_selected_for_two_people_and_events():
    routes = {
        101: [
            RouteCandidate("p1-car", 0.4, person_key="p1", vehicle_key="car-7"),
        ],
        102: [
            RouteCandidate("p2-car", 0.3, person_key="p2", vehicle_key="car-7"),
        ],
    }

    result = solve_event_routes(routes, null_cost=8.0)

    assert result[101].route.person_key == "p1"
    assert result[102].route.person_key == "p2"
    assert result[101].route.vehicle_key == "car-7"
    assert result[102].route.vehicle_key == "car-7"


def test_same_person_can_explain_multiple_litter_events():
    routes = {
        "litter-a": [RouteCandidate("person", 0.2, person_key=("person", 9))],
        "litter-b": [RouteCandidate("person", 0.25, person_key=("person", 9))],
    }

    result = solve_event_routes(routes, null_cost=6.0)

    assert result["litter-a"].route.person_key == ("person", 9)
    assert result["litter-b"].route.person_key == ("person", 9)


def test_direct_vehicle_and_synthetic_null_routes():
    routes = {
        "direct": [
            RouteCandidate(
                "vehicle-only",
                0.6,
                person_key=None,
                vehicle_key="truck",
                route_type="direct_vehicle",
            ),
            RouteCandidate("person", 1.4, person_key="p3"),
        ],
        "invalid-only": [
            RouteCandidate("nan", math.nan, person_key="bad"),
            RouteCandidate("inf", math.inf, vehicle_key="bad"),
        ],
    }

    result = solve_event_routes(routes, null_cost=3.0)

    assert result["direct"].route.is_direct_vehicle
    assert result["direct"].route.vehicle_key == "truck"
    assert result["invalid-only"].route.is_null
    assert result["invalid-only"].route.route_id == "__null__"


def test_explicit_null_can_win_and_margin_uses_scaled_objective():
    routes = {
        1: [
            RouteCandidate("person", 1.2344, person_key=4),
            RouteCandidate("dustbin", 1.0, route_type="null"),
        ]
    }

    assignment = solve_event_routes(routes)[1]

    assert assignment.route.route_id == "dustbin"
    assert assignment.scaled_cost == 1000
    assert assignment.margin_to_second == 0.234


def test_equal_cost_tie_is_deterministic_independent_of_input_order():
    route_a = RouteCandidate("a-route", 0.5, person_key="p-a")
    route_b = RouteCandidate("b-route", 0.5, person_key="p-b")

    first = solve_event_routes({"event": [route_b, route_a]}, null_cost=9.0)
    second = solve_event_routes({"event": [route_a, route_b]}, null_cost=9.0)

    assert first["event"].route.route_id == "a-route"
    assert second["event"].route.route_id == "a-route"
    assert first["event"].margin_to_second == 0.0
