import pytest

from pipeline.backtrack.resolver import (
    SmartBacktrackConfig,
    SmartBacktrackResolver,
    _actor_specific_margins,
)
from pipeline.backtrack.flow import RouteCandidate


def test_actor_specific_margin_collapses_routes_for_the_same_identity():
    routes = [
        RouteCandidate(
            "vehicle-1-direct", 1.0002,
            vehicle_key=("vehicle", 1), route_type="direct_vehicle",
        ),
        RouteCandidate(
            "vehicle-1-person", 1.0004,
            person_key=("person", 7), vehicle_key=("vehicle", 1),
        ),
        RouteCandidate(
            "vehicle-2", 1.31,
            vehicle_key=("vehicle", 2), route_type="direct_vehicle",
        ),
        RouteCandidate(
            "person-8", 1.0003,
            person_key=("person", 8), route_type="person",
        ),
        RouteCandidate("null", 7.0, route_type="null"),
    ]

    margins = _actor_specific_margins(routes)

    assert margins["vehicle"]["best_key"] == ("vehicle", 1)
    assert margins["vehicle"]["second_key"] == ("vehicle", 2)
    assert margins["vehicle"]["margin"] == pytest.approx(0.3098)
    assert margins["vehicle"]["tie_count"] == 1
    assert margins["person"]["best_key"] == ("person", 8)
    assert margins["person"]["second_key"] == ("person", 7)
    assert margins["person"]["margin"] == pytest.approx(0.0001)
    assert margins["person"]["tie_count"] == 1
    assert margins["null"]["margin"] == pytest.approx(5.9998)


def test_study_config_can_ablate_ac_overlap_without_changing_other_weights():
    baseline = StudyConfig(stage="full").resolver_config(fps=10).cost_config
    trial = StudyConfig(
        stage="full", ac_overlap_weight=0.2
    ).resolver_config(fps=10).cost_config

    assert baseline.ac_weights["overlap"] == pytest.approx(0.0)
    assert trial.ac_weights["overlap"] == pytest.approx(0.2)
    assert trial.ba_weights == baseline.ba_weights
    assert trial.bc_weights == baseline.bc_weights


def test_study_config_can_ablate_bc_direction_features_in_sequence():
    config = StudyConfig(
        stage="full",
        bc_exit_deficit_weight=0.2,
        bc_relative_motion_deficit_weight=0.3,
        bc_reverse_direction_weight=0.4,
        bc_boundary_depth_weight=0.5,
    ).resolver_config(fps=10).cost_config

    assert config.bc_weights["exit_deficit"] == pytest.approx(0.2)
    assert config.bc_weights["relative_motion_deficit"] == pytest.approx(0.3)
    assert config.bc_weights["reverse_direction"] == pytest.approx(0.4)
    assert config.bc_weights["boundary_depth"] == pytest.approx(0.5)
from pipeline.backtrack.costs import BacktrackCostConfig, compute_c_ba
from pipeline.backtrack.trajectory import ReleaseHypothesis
from pipeline.backtrack.costs import ActorObservation
import numpy as np
from pipeline.backtrack.sidecar import build_candidate_record
from pipeline.backtrack.study import StudyConfig, build_manifest, replay_candidates
from pipeline.backtrack.annotations import AnnotationError


def _task():
    actors = []
    for frame in range(8, 13):
        actors.append({
            "frame_index": frame,
            "actors": [
                {"cls": "person", "track_id": 1, "box": [90, 50, 130, 170], "confidence": .9},
                {"cls": "vehicle", "track_id": 2, "box": [50, 120, 180, 230], "confidence": .9},
            ],
        })
    return {
        "litter_id": 9, "fps": 10.0, "birth_frame": 10, "confirm_frame": 12,
        "history": [(108, 95), (110, 100), (112, 108)],
        "history_frames": [10, 11, 12],
        "history_boxes": [[104, 91, 112, 99], [106, 96, 114, 104], [108, 104, 116, 112]],
        "history_confidences": [.9, .9, .9], "actor_frames": actors,
    }


def _record():
    task = _task()
    resolution = SmartBacktrackResolver(fps=10).resolve_task(task)
    event = {"litter_id": 9, "frame_index": 12, "backtrack": {
        "route_id": resolution.route_id, "route_type": resolution.route_type,
    }}
    return build_candidate_record(task, event, resolution.routes, "/dataset/cam_a/clip.mp4")


def test_sidecar_keeps_replayable_resolver_input():
    record = _record()
    assert record["resolver_input"]["actor_frames"] == _task()["actor_frames"]


def test_manifest_is_grouped_and_replay_is_deterministic():
    record = _record()
    manifest = build_manifest([record], seed=3)
    assert manifest["entries"][0]["split"] == "development"
    config = StudyConfig(name="default")
    first, first_summary = replay_candidates([record], config, manifest=manifest, split="development")
    second, second_summary = replay_candidates([record], config, manifest=manifest, split="development")
    assert first_summary["events"] == second_summary["events"] == 1
    assert first[0]["assignment"] == second[0]["assignment"]
    assert first[0]["study"]["config_digest"] == second[0]["study"]["config_digest"]


def test_replay_rejects_old_sidecar_without_frozen_input():
    record = _record()
    record.pop("resolver_input")
    with pytest.raises(AnnotationError, match="replayable"):
        replay_candidates([record], StudyConfig())


def test_distance_time_stage_excludes_non_intuitive_cost_components():
    release = ReleaseHypothesis(10, np.asarray([110.0, 90.0]), np.eye(2), np.zeros(2), "test", 5.0)
    person = ActorObservation("person", 1, 10, (90, 50, 130, 170), confidence=.2)
    cell = compute_c_ba([release], [person], 10, cost_config=BacktrackCostConfig.for_stage("distance_time"))
    assert set(cell.weights) == {"release_distance", "time"}
    assert set(cell.components) == {"release_distance", "time"}


def test_distance_time_stage_normalizes_by_gate_before_weighting():
    # Person height=120. Release is 51 px outside the bbox, hence the original
    # normalized distance is 51/120=.425 and consumes .425/.85=.5 of its gate.
    release = ReleaseHypothesis(
        10, np.asarray([181.0, 90.0]), np.eye(2), np.zeros(2), "test", 0.0
    )
    person = ActorObservation("person", 1, 11, (90, 50, 130, 170))
    config = BacktrackCostConfig.for_stage(
        "distance_time", distance_weight=.65, time_weight=.35
    )

    cell = compute_c_ba([release], [person], 10, cost_config=config)

    assert cell.valid
    assert cell.raw_features["release_distance"] == pytest.approx(.5)
    assert cell.raw_features["time"] == pytest.approx(.4)  # .1 s / .25 s
    assert cell.components["release_distance"] == pytest.approx(.325)
    assert cell.components["time"] == pytest.approx(.14)
    assert cell.total == pytest.approx(.465)


def test_full_stage_keeps_historical_feature_units():
    release = ReleaseHypothesis(
        10, np.asarray([181.0, 90.0]), np.eye(2), np.zeros(2), "test", 0.0
    )
    person = ActorObservation("person", 1, 11, (90, 50, 130, 170))

    cell = compute_c_ba([release], [person], 10)

    assert cell.raw_features["release_distance"] == pytest.approx(.425)
    assert cell.raw_features["time"] == pytest.approx(.1)


def test_runtime_distance_time_weights_are_explicit(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK_STUDY_STAGE", "distance_time")
    monkeypatch.setenv("SMART_BACKTRACK_DT_DISTANCE_WEIGHT", "0.75")
    monkeypatch.setenv("SMART_BACKTRACK_DT_TIME_WEIGHT", "0.25")

    config = SmartBacktrackConfig.from_env(fps=10)

    assert config.cost_config.ba_weights == {
        "release_distance": .75, "time": .25,
    }
    assert config.cost_config.bc_weights == {
        "direct_distance": .75, "time": .25,
    }
    assert config.cost_config.ac_weights == {
        "endpoint_proximity": .75, "time": .25,
    }
    assert config.cost_config.normalize_distance_time_by_gate


def test_runtime_boundary_depth_weight_applies_only_to_reverse_full(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT", "1.1")
    monkeypatch.setenv("SMART_BACKTRACK_STUDY_STAGE", "full")

    full = SmartBacktrackConfig.from_env(fps=10)

    assert full.cost_config.bc_weights["boundary_depth"] == pytest.approx(1.1)

    monkeypatch.setenv("SMART_BACKTRACK_STUDY_STAGE", "distance_time")
    distance_time = SmartBacktrackConfig.from_env(fps=10)

    assert "boundary_depth" not in distance_time.cost_config.bc_weights


def test_kalman_rts_stage_keeps_only_distance_time_costs():
    config = StudyConfig(
        name="kalman-rts-65-35",
        stage="kalman_rts",
        distance_weight=.65,
        time_weight=.35,
        kalman_process_noise_scale=.75,
        kalman_measurement_noise_scale=1.25,
        kalman_max_extrapolation_seconds=.3,
    ).resolver_config(fps=10)

    assert config.use_kalman_rts
    assert not config.confidence_aware_kalman
    assert not config.use_uncertainty_gates
    assert not config.use_reverse_trajectory
    assert config.kalman_process_noise_scale == .75
    assert config.kalman_measurement_noise_scale == 1.25
    assert config.kalman_max_extrapolation_seconds == .3
    assert config.cost_config.ba_weights == {
        "release_distance": .65, "time": .35,
    }
    assert config.cost_config.bc_weights == {
        "direct_distance": .65, "time": .35,
    }
    assert config.cost_config.ac_weights == {
        "endpoint_proximity": .65, "time": .35,
    }


def test_kalman_rts_stage_fills_missing_actor_frame_without_extra_costs():
    task = _task()
    task["actor_frames"] = [
        row for row in task["actor_frames"] if row["frame_index"] != 10
    ]
    config = StudyConfig(stage="kalman_rts").resolver_config(fps=10)
    resolver = SmartBacktrackResolver(fps=10, config=config)

    tracks = resolver._build_actor_tracks(task)
    person_at_birth = next(
        item for item in tracks[("person", 1)] if item.frame_index == 10
    )
    routes = resolver.build_routes(task)
    diagnostics = next(
        route.metadata["candidate_diagnostics"]
        for route in routes if route.route_type == "null"
    )

    assert not person_at_birth.observed
    assert person_at_birth.source == "kalman_rts"
    ba_cells = diagnostics["pair_costs"]["BA"]
    assert ba_cells
    assert set(ba_cells[0]["cell"]["weights"]) == {
        "release_distance", "time"
    }
    assert ba_cells[0]["cell"]["raw_features"]["time"] == pytest.approx(.4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kalman_process_noise_scale", 0.0),
        ("kalman_measurement_noise_scale", 0.0),
        ("kalman_max_extrapolation_seconds", -1.0),
    ],
)
def test_kalman_rts_tuning_parameters_are_validated(field, value):
    with pytest.raises(ValueError, match=field):
        StudyConfig(stage="kalman_rts", **{field: value})
