import json
import math
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.backtrack.flow import RouteCandidate
from pipeline.backtrack.sidecar import (
    SCHEMA_NAME,
    build_candidate_record,
    build_run_record,
    read_jsonl,
    write_jsonl,
)


def test_build_run_record_is_strict_json_safe():
    record = build_run_record(
        "輸入.mp4",
        "output.mp4",
        fps=math.nan,
        frame_count="42",
        smart_summary={"resolved": 2, "bad": math.inf},
        extra={"labels": ("a", "b")},
    )

    assert record["schema"] == "smart-backtrack-candidates/v1"
    assert record["record_type"] == "run"
    assert record["video"] == {
        "input_video": "輸入.mp4",
        "output_video": "output.mp4",
        "fps": None,
        "frame_count": 42,
    }
    assert record["smart_summary"]["bad"] is None
    assert record["extra"]["labels"] == ["a", "b"]
    json.dumps(record, allow_nan=False)


def test_candidate_routes_are_ranked_scaled_selected_and_diagnostics_extracted():
    diagnostics = {
        "release_hypotheses": [{
            "uid": "R10",
            "frame": 10,
            "prior_cost": math.nan,
        }],
        "pair_costs": {
            "BA": [{
                "person_key": ("person", 1),
                "valid": False,
                "total": math.inf,
                "reject_reason": "release_distance_gate",
            }],
            "AC": [],
            "BC": [],
        },
    }
    null_route = RouteCandidate(
        "null",
        7.0,
        route_type="null",
        metadata={
            "candidate_diagnostics": diagnostics,
            "costs": {"BA": None, "AC": None, "BC": None},
        },
    )
    routes = [
        null_route,
        RouteCandidate(
            "selected",
            1.2345,
            person_key=("person", 1),
            route_type="person",
            metadata={"costs": {"BA": {"valid": True, "total": 1.2345}}},
        ),
        RouteCandidate(
            "first",
            1.2344,
            person_key=("person", 2),
            route_type="person",
        ),
        RouteCandidate(
            "invalid",
            math.inf,
            person_key=("person", 3),
            metadata={"valid": False, "reject_reason": "synthetic_reject"},
        ),
    ]
    event = {
        "litter_id": 8,
        "backtrack_status": "resolved",
        "backtrack": {
            "route_id": "selected",
            "route_type": "person",
            "person_key": ["person", 1],
            "vehicle_key": None,
            "score": 1.2345,
            "margin_to_second": 0.001,
            "components": {"candidate_diagnostics": diagnostics},
        },
    }

    record = build_candidate_record(
        {"fps": 10.0}, event, routes, "input.mp4", "output.mp4"
    )

    assert [route["route_id"] for route in record["routes"]] == [
        "first", "selected", "null", "invalid"
    ]
    assert [route["rank"] for route in record["routes"]] == [1, 2, 3, None]
    selected = next(route for route in record["routes"] if route["selected"])
    assert selected["route_id"] == "selected"
    assert selected["scaled_cost"] == 1234500
    invalid = record["routes"][-1]
    assert invalid["valid"] is False
    assert invalid["cost"] is None
    assert invalid["reject_reason"] == "synthetic_reject"
    assert all(
        "candidate_diagnostics" not in route["metadata"]
        for route in record["routes"]
    )
    assert "candidate_diagnostics" in null_route.metadata
    assert len(record["release_hypotheses"]) == 1
    assert record["release_hypotheses"][0]["prior_cost"] is None
    ba = record["pair_costs"]["BA"][0]
    assert ba["valid"] is False
    assert ba["total"] is None
    assert ba["reject_reason"] == "release_distance_gate"
    assert (
        "candidate_diagnostics"
        not in record["event"]["backtrack"]["components"]
    )
    assert record["assignment"]["scaled_cost"] == 1234500
    assert record["assignment"]["actor_margins"]["person"] == {
        "best_key": ["person", 2],
        "best_cost": 1.2344,
        "second_key": ["person", 1],
        "second_cost": 1.2345,
        "margin": pytest.approx(0.0001),
        "tie_count": 1,
    }


def test_candidate_contains_litter_history_and_allowed_raw_actor_tracklets():
    task = {
        "fps": 12,
        "history": [(10.0, 20.0), (12.0, 24.0)],
        "history_frames": [5, 6],
        "history_boxes": [(8, 18, 12, 22), (10, 22, 14, 26)],
        "history_confidences": [0.8, math.nan],
        "actor_frames": [
            {
                "frame_index": 5,
                "actors": [
                    {
                        "cls": "person",
                        "track_id": 1,
                        "tracklet_uid": "person:1@4",
                        "box": (0, 0, 10, 20),
                        "confidence": 0.9,
                    },
                    {
                        "cls": "litter",
                        "track_id": 99,
                        "box": (5, 5, 6, 6),
                    },
                ],
            },
            {
                "frame_index": 6,
                "actors": [
                    {
                        "cls": "person",
                        "track_id": 1,
                        "tracklet_uid": "person:1@4",
                        "box": (1, 0, 11, 20),
                        "confidence": math.inf,
                    },
                    {
                        "cls": "scooter",
                        "track_id": 3,
                        "box": (20, 20, 40, 40),
                    },
                ],
            },
        ],
    }
    event = {
        "litter_id": 2,
        "backtrack_status": "dustbin",
        "backtrack": {"route_id": "null", "route_type": "null"},
    }
    routes = [RouteCandidate("null", 7.0, route_type="null")]

    record = build_candidate_record(task, event, routes, "case.mp4")

    assert record["litter_history"][0] == {
        "frame_index": 5,
        "point_uv": [10.0, 20.0],
        "bbox_xyxy": [8, 18, 12, 22],
        "confidence": 0.8,
    }
    assert record["litter_history"][1]["confidence"] is None
    assert [actor["class_name"] for actor in record["candidate_actors"]] == [
        "person", "scooter"
    ]
    person = record["candidate_actors"][0]
    assert person["actor_key"] == ["person", 1]
    assert person["frame_range"] == [5, 6]
    assert len(person["observations"]) == 2
    assert person["observations"][1]["confidence"] is None
    assert record["routes"][0]["selected"] is True


def test_jsonl_roundtrip_and_half_away_from_zero(tmp_path):
    records = [
        build_run_record("a.mp4", None, 10, 3),
        build_candidate_record(
            {},
            {"backtrack": {"route_id": "negative"}},
            [RouteCandidate("negative", -1.2345, person_key=("person", 1))],
            "a.mp4",
        ),
    ]
    path = tmp_path / "nested" / "run.backtrack.candidates.jsonl"

    assert write_jsonl(records, path) == 2
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    loaded = read_jsonl(path)

    assert loaded[0]["schema"] == SCHEMA_NAME
    assert loaded[1]["routes"][0]["scaled_cost"] == -1234500
    assert loaded == records


def test_candidate_sidecar_excludes_plate_roi_pixels_but_remains_replayable():
    task = {
        "litter_id": 3,
        "fps": 10.0,
        "birth_frame": 10,
        "confirm_frame": 11,
        "history": [(20.0, 30.0), (21.0, 32.0)],
        "history_frames": [10, 11],
        "actor_frames": [{
            "frame_index": 10,
            "actors": [{
                "cls": "vehicle",
                "track_id": 2,
                "box": [0, 0, 40, 40],
                "plate_roi": np.full((48, 160, 3), 255, dtype=np.uint8),
            }],
        }],
        "plate_actor_frames": [{
            "frame_index": 10,
            "actors": [{
                "cls": "vehicle",
                "track_id": 2,
                "box": [0, 0, 40, 40],
                "plate_roi": np.full((48, 160, 3), 255, dtype=np.uint8),
            }],
        }],
    }
    event = {
        "litter_id": 3,
        "backtrack_status": "dustbin",
        "backtrack": {"route_id": "null", "route_type": "null"},
    }

    record = build_candidate_record(
        task,
        event,
        [RouteCandidate("null", 7.0, route_type="null")],
        "case.mp4",
    )
    encoded = json.dumps(record, allow_nan=False)

    assert "plate_actor_frames" not in record["resolver_input"]
    assert "plate_roi" not in encoded
    assert record["resolver_input"]["actor_frames"][0]["actors"][0]["box"] == [
        0, 0, 40, 40
    ]
    assert len(encoded) < 10_000
