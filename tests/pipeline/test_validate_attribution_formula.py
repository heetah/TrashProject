import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_attribution_formula.py"
SPEC = importlib.util.spec_from_file_location("validate_attribution_formula", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_route_rule_is_derived_only_from_actor_presence():
    assert MODULE.route_from_actor_types(["vehicle"]) == "direct_vehicle"
    assert MODULE.route_from_actor_types(["person", "vehicle"]) == "person_vehicle"
    assert MODULE.route_from_actor_types(["person"]) == "person"
    assert MODULE.route_from_actor_types([]) == "null"


def test_candidate_records_discovers_production_case_subdirectories(tmp_path: Path):
    case_dir = tmp_path / "case_7"
    case_dir.mkdir()
    (case_dir / "litter_case_7_annotated_backtrack_candidates.jsonl").write_text(
        '{"record_type":"candidate","event":{"litter_id":1}}\n',
        encoding="utf-8",
    )

    rows = MODULE.candidate_records(tmp_path)

    assert len(rows) == 1
    assert rows[0]["event"]["litter_id"] == 1


def test_wilson_interval_contains_observed_fraction():
    interval = MODULE.wilson_interval(19, 58)
    assert interval is not None
    assert interval[0] < 19 / 58 < interval[1]
    assert interval == pytest.approx([0.2208329108, 0.4557594532])


def test_bbox_and_point_geometry():
    assert MODULE.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert MODULE.bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert MODULE.point_to_rect_distance([5, 5], [0, 0, 10, 10]) == 0.0
    assert MODULE.point_to_rect_distance([13, 14], [0, 0, 10, 10]) == 5.0


def test_actor_geometry_is_resolution_invariant():
    release = [{"frame_index": 10, "mean_uv": [15.0, 5.0]}]
    person = {
        "class_name": "person",
        "observations": [{"frame_index": 10, "box": [0.0, 0.0, 10.0, 10.0]}],
    }
    scaled_release = [{"frame_index": 10, "mean_uv": [45.0, 15.0]}]
    scaled_person = {
        "class_name": "person",
        "observations": [{"frame_index": 10, "box": [0.0, 0.0, 30.0, 30.0]}],
    }
    first = MODULE.actor_geometry(person, release, fps=10.0)
    second = MODULE.actor_geometry(scaled_person, scaled_release, fps=10.0)
    assert first is not None and second is not None
    assert second["raw_distance"] == pytest.approx(first["raw_distance"] * 3.0)
    assert second["actor_scale"] == pytest.approx(first["actor_scale"] * 3.0)
    assert second["normalized_distance"] == pytest.approx(first["normalized_distance"])


def test_lower_distance_auc_handles_ties():
    assert MODULE.lower_is_positive_auc([0.0], [1.0]) == 1.0
    assert MODULE.lower_is_positive_auc([1.0], [0.0]) == 0.0
    assert MODULE.lower_is_positive_auc([1.0], [1.0]) == 0.5


def test_release_timing_keeps_b0_as_an_explicit_hypothesis():
    match = {
        "clip_id": "clip",
        "litter_id": 1,
        "match_tier": "strict",
        "event": {
            "release_point_frame": 10,
            "release_start_frame": 10,
            "release_end_frame": 10,
            "release_x": 5.0,
            "release_y": 5.0,
        },
        "candidate": {
            "resolver_input": {
                "history_frames": [10, 12],
                "history": [[5.0, 5.0], [9.0, 5.0]],
            },
            "assignment": {"release_frame": 10, "release_point": [5.0, 5.0]},
        },
    }
    rows = MODULE.release_timing_rows([match])
    by_method = {row["method"]: row for row in rows}
    assert by_method["b0"]["absolute_point_error_frames"] == 0.0
    assert by_method["b0_minus_half_h0"]["estimated_release_frame"] == 9.0
    assert by_method["b0_minus_h0"]["estimated_release_frame"] == 8.0


def test_same_frame_distance_compares_regions_and_points():
    match = {
        "clip_id": "clip",
        "litter_id": 1,
        "match_tier": "strict",
        "event": {
            "release_point_frame": 10,
            "release_x": 5.0,
            "release_y": 2.0,
        },
        "correct_person_key": ("person", 1),
        "correct_vehicle_key": None,
        "candidate": {
            "resolver_input": {
                "actor_frames": [{
                    "frame_index": 10,
                    "actors": [{
                        "actor_key": ["person", 1],
                        "cls": "person",
                        "box": [0.0, 0.0, 10.0, 20.0],
                    }],
                }],
            },
        },
    }
    rows = MODULE.same_frame_distance_rows([match])
    values = {row["distance_definition"]: row["normalized_distance"] for row in rows}
    assert values["upper72_region"] == 0.0
    assert values["full_bbox_region"] == 0.0
    assert values["upper_center_point"] > 0.0
    assert values["bbox_center_point"] > values["upper_center_point"]
