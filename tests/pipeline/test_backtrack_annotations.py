# -*- coding: utf-8 -*-
"""Tests for independent smart-backtrack annotations and metrics."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.backtrack.annotations import (
    ANNOTATION_SCHEMA,
    CANDIDATE_SCHEMA,
    evaluate_candidates,
    init_annotation_records,
    validate_records,
)


def _run_record(run_id="run-1", path="/data/litter/case.mp4", fps=10):
    return {
        "schema": CANDIDATE_SCHEMA,
        "record_type": "run",
        "run_id": run_id,
        "video": {
            "clip_id": "case",
            "input_path": path,
            "fps": fps,
            "frame_count": 100,
        },
        "summary": {},
    }


def _route(
    rank,
    route_type,
    person=None,
    vehicle=None,
    release=40,
    selected=False,
):
    return {
        "rank": rank,
        "route_id": "{}-{}".format(route_type, rank),
        "route_type": route_type,
        "person_key": person,
        "vehicle_key": vehicle,
        "cost": float(rank),
        "selected": selected,
        "metadata": {
            "release_frame": release,
            "costs": {
                "BA": {"total": 0.2, "components": {"distance": 0.1}},
                "AC": {"total": 0.3, "components": {"overlap": 0.2}},
                "BC": {"total": 0.4, "components": {"release_distance": 0.1}},
            },
        },
    }


def _event(event_id="event-1", routes=None, run_id="run-1"):
    return {
        "schema": CANDIDATE_SCHEMA,
        "record_type": "event",
        "run_id": run_id,
        "video": {"clip_id": "case", "input_path": "/data/litter/case.mp4", "fps": 10},
        "event": {"event_id": event_id, "litter_id": 1},
        "assignment": {},
        "routes": routes or [
            _route(
                1, "person_vehicle", ["person", 2], ["vehicle", 8],
                selected=True,
            ),
            _route(2, "null", None, None),
        ],
    }


def _annotation(
    event_id,
    routes,
    interval=(38, 42),
    run_id="run-1",
    ignore=False,
):
    return {
        "schema": ANNOTATION_SCHEMA,
        "record_type": "event",
        "run_id": run_id,
        "clip_id": "case",
        "event_id": event_id,
        "ignore": ignore,
        "review": {
            "event": "reviewed",
            "person": "reviewed",
            "vehicle": "reviewed",
            "release": "reviewed",
        },
        "event_label": "litter",
        "admissible_routes": routes,
        "release_interval": list(interval),
    }


def test_init_never_copies_runtime_prediction_into_gt():
    candidates = [_run_record(), _event()]

    annotations = init_annotation_records(candidates)

    assert annotations[0]["clip_weak_label"] == "litter"
    event = annotations[1]
    assert event["review"] == {
        "event": "unreviewed",
        "person": "unreviewed",
        "vehicle": "unreviewed",
        "release": "unreviewed",
    }
    assert event["event_label"] is None
    assert event["admissible_routes"] == []
    assert event["release_interval"] is None
    serialized = json.dumps(event)
    assert "person:2" not in serialized
    assert "vehicle:8" not in serialized
    assert '"release_frame"' not in serialized


def test_real_sidecar_aliases_preserve_only_explicit_clip_weak_label():
    run = {
        "schema": CANDIDATE_SCHEMA,
        "record_type": "run",
        "video": {
            "input_video": "/dataset/unclassified/case.mp4",
            "fps": 12,
        },
        "extra": {
            "clip_label": {
                "value": "litter",
                "strength": "weak_gt",
                "source": "parent_directory",
            }
        },
    }
    event = {
        "schema": CANDIDATE_SCHEMA,
        "record_type": "candidate",
        "video": {
            "input_video": "/dataset/unclassified/case.mp4",
            "fps": 12,
        },
        "event": {"litter_id": 7},
        "routes": [
            _route(1, "null", None, None, None, selected=True),
            {
                **_route(2, "person", ["person", 2], None),
                "valid": False,
                "rank": None,
            },
        ],
    }

    validation = validate_records([run, event])
    annotations = init_annotation_records([run, event])

    assert validation["valid"], validation["errors"]
    assert annotations[0]["clip_weak_label"] == "litter"
    assert annotations[0]["clip_weak_label_metadata"]["strength"] == "weak_gt"
    assert annotations[1]["event_id"] == "7"
    assert annotations[1]["event_label"] is None
    assert annotations[1]["admissible_routes"] == []


def test_validate_accepts_ambiguity_and_same_vehicle_for_multiple_people():
    annotation_run = {
        "schema": ANNOTATION_SCHEMA,
        "record_type": "run",
        "run_id": "run-1",
        "clip_id": "case",
        "clip_weak_label": "litter",
    }
    # Ambiguous people can share the same vehicle, and another event can also
    # legally use that vehicle.  No one-to-one actor capacity is imposed.
    event_1 = _annotation(
        "event-1",
        [
            {
                "route_type": "person_vehicle",
                "person_id": "person:2",
                "vehicle_id": "vehicle:8",
            },
            {
                "route_type": "person_vehicle",
                "person_id": "person:3",
                "vehicle_id": "vehicle:8",
            },
        ],
    )
    event_2 = _annotation(
        "event-2",
        [{
            "route_type": "person_vehicle",
            "person_id": "person:4",
            "vehicle_id": "vehicle:8",
        }],
    )

    report = validate_records([annotation_run, event_1, event_2])

    assert report["valid"], report["errors"]


def test_evaluate_exact_recall_coverage_release_null_and_ignore():
    candidates = [
        _run_record(),
        _event(
            "ambiguous",
            [
                _route(1, "person_vehicle", ["person", 99], ["vehicle", 8], 45, True),
                _route(2, "person_vehicle", ["person", 3], ["vehicle", 8], 41),
                _route(3, "null", None, None, None),
            ],
        ),
        _event(
            "null",
            [
                _route(1, "null", None, None, None, True),
                _route(2, "person", ["person", 4], None, 30),
            ],
        ),
        _event(
            "ignored",
            [_route(1, "null", None, None, None, True)],
        ),
    ]
    annotation_run = {
        "schema": ANNOTATION_SCHEMA,
        "record_type": "run",
        "run_id": "run-1",
        "clip_id": "case",
        "video": {"fps": 10},
        "clip_weak_label": "litter",
    }
    annotations = [
        annotation_run,
        _annotation(
            "ambiguous",
            [
                {
                    "route_type": "person_vehicle",
                    "person_id": "person:2",
                    "vehicle_id": "vehicle:8",
                },
                {
                    "route_type": "person_vehicle",
                    "person_id": "person:3",
                    "vehicle_id": "vehicle:8",
                },
            ],
            interval=(40, 42),
        ),
        _annotation(
            "null",
            [{"route_type": "null", "person_id": None, "vehicle_id": None}],
            interval=(20, 25),
        ),
        _annotation(
            "ignored",
            [{"route_type": "null", "person_id": None, "vehicle_id": None}],
            ignore=True,
        ),
    ]

    report = evaluate_candidates(candidates, annotations)

    assert report["counts"]["route_evaluable_events"] == 2
    assert report["counts"]["ignored_events"] == 1
    assert report["exact_route_top1"]["hits"] == 1  # NULL event only
    assert report["recall_at_k"]["1"]["hits"] == 1
    assert report["recall_at_k"]["3"]["hits"] == 2
    assert report["candidate_coverage"]["person"]["value"] == 1.0
    assert report["candidate_coverage"]["vehicle"]["value"] == 1.0
    assert report["candidate_coverage"]["route"]["value"] == 1.0
    # Ambiguous event top-1 release 45 is 3 frames outside [40,42].
    # The NULL route has no release prediction.
    assert report["release_interval"]["support"] == 2
    assert report["release_interval"]["hits"] == 0
    assert report["release_interval"]["mae_frames"] == 3.0
    assert report["release_interval"]["missing_prediction"] == 1
    assert report["null_classification"]["confusion"] == {
        "tp": 1, "tn": 1, "fp": 0, "fn": 0
    }


def test_mixed_null_and_non_null_truth_is_excluded_from_null_binary_metric():
    candidates = [_run_record(), _event()]
    annotations = [
        {
            "schema": ANNOTATION_SCHEMA,
            "record_type": "run",
            "run_id": "run-1",
            "clip_id": "case",
            "clip_weak_label": "litter",
        },
        _annotation(
            "event-1",
            [
                {
                    "route_type": "person_vehicle",
                    "person_id": "person:2",
                    "vehicle_id": "vehicle:8",
                },
                {"route_type": "null", "person_id": None, "vehicle_id": None},
            ],
        ),
    ]

    report = evaluate_candidates(candidates, annotations)

    assert report["null_classification"]["support"] == 0
    assert report["null_classification"][
        "ambiguous_null_and_non_null_skipped"
    ] == 1


def test_cli_init_validate_and_evaluate(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "\n".join(json.dumps(value) for value in [_run_record(), _event()]) + "\n",
        encoding="utf-8",
    )
    annotation_path = tmp_path / "annotations.jsonl"
    cli = Path(__file__).resolve().parents[2] / "tools" / "backtrack_annotations.py"
    if not cli.exists():
        pytest.skip("optional backtrack annotations CLI is not present in this checkout")

    initialized = subprocess.run(
        [
            sys.executable, str(cli), "init", str(candidate_path),
            "--output", str(annotation_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0, initialized.stderr
    payload = json.loads(initialized.stdout)
    assert payload["event_records"] == 1

    validated = subprocess.run(
        [sys.executable, str(cli), "validate", str(annotation_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True

    strict = subprocess.run(
        [
            sys.executable, str(cli), "validate", str(annotation_path),
            "--require-reviewed",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert strict.returncode == 2
