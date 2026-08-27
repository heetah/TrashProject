import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from summarize_actor_ground_truth_metrics import _event_row, _summary_stats


def test_event_row_computes_route_and_release_birth_metrics():
    record = {
        "record_type": "candidate",
        "assignment": {
            "route_type": "direct_vehicle",
            "route_id": "direct:vehicle:2",
            "vehicle_key": ["vehicle", 2],
            "person_key": None,
            "cost": 1.2,
            "margin_to_second": 0.4,
            "release_frame": 8,
            "release_point": [3.0, 4.0],
            "actor_margins": {"vehicle": {"margin": 0.4}, "null": {"margin": 5.8}},
        },
        "event": {"litter_id": 0, "backtrack": {"components": {"costs": {"BC": {"raw_features": {"direct_distance": 0.1, "time": 0.2, "reverse_direction": 0.3}, "components": {}}}}}},
        "resolver_input": {"birth_frame": 10, "birth_centroid": [0.0, 0.0], "fps": 10.0, "confirm_frame": 10},
    }
    row = _event_row(7, 0, record, {"vehicle_diagonal": 10.0}, False)
    assert row["predicted_vehicle_id"] == 2
    assert row["route_match"] is True
    assert row["release_to_birth_delta_frames"] == 2
    assert row["release_to_birth_delta_sec"] == 0.2
    assert row["release_birth_distance_px"] == 5.0
    assert row["release_birth_distance_over_actor_diagonal"] == 0.5
    assert row["raw_direction_feature"] == 0.3


def test_summary_stats_empty_and_numeric():
    assert _summary_stats([], "x")["n"] == 0
    assert _summary_stats([{"x": 1}, {"x": 3}], "x")["median"] == 2


def test_unknown_vehicle_cases_keep_id_unknown_but_are_human_adjudicated():
    record = {
        "record_type": "candidate",
        "assignment": {
            "route_type": "direct_vehicle",
            "vehicle_key": ["vehicle", 1],
            "person_key": None,
            "release_frame": 8,
            "release_point": [3.0, 4.0],
        },
        "event": {"litter_id": 0},
        "resolver_input": {"birth_frame": 8, "birth_centroid": [3.0, 4.0], "fps": 10.0},
    }
    row = _event_row(174, 0, record, {}, False)
    assert row["gt_vehicle_id"] == "?"
    assert row["predicted_vehicle_id"] == 1
    assert row["human_verified_correct"] is True
    assert row["route_match"] is True
    assert row["vehicle_match"] is None
