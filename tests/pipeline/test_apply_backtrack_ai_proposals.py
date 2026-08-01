import importlib.util
from pathlib import Path

import pytest

from pipeline.backtrack.annotations import AnnotationError


TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "apply_backtrack_ai_proposals.py"
)
if not TOOL_PATH.exists():
    pytest.skip(
        "optional backtrack AI proposal tool is not present in this checkout",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("apply_backtrack_ai_proposals", TOOL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _annotations(review_state="unreviewed"):
    return [
        {
            "schema": "smart-backtrack-annotations/v1",
            "record_type": "run",
            "run_id": "run-1",
            "clip_id": "case_1",
            "video": {"input_path": "/data/case_1.mp4"},
            "clip_weak_label": "litter",
        },
        {
            "schema": "smart-backtrack-annotations/v1",
            "record_type": "event",
            "run_id": "run-1",
            "clip_id": "case_1",
            "event_id": "7",
            "ignore": False,
            "review": {
                "event": review_state,
                "person": review_state,
                "vehicle": review_state,
                "release": review_state,
            },
            "event_label": None,
            "admissible_routes": [],
            "release_interval": None,
            "notes": "",
        },
    ]


def _proposal():
    return {
        "input_video": "/data/case_1.mp4",
        "litter_id": 7,
        "event_label": "litter",
        "release_interval": [35, 37],
        "admissible_routes": [
            {
                "route_type": "person_vehicle",
                "person_id": "person:12",
                "vehicle_id": "vehicle:11",
            }
        ],
        "confidence": "medium",
        "notes": "AI initial label",
        "_proposal_source": "part-a.jsonl",
        "_proposal_line": 1,
    }


def test_apply_proposal_fills_values_but_keeps_unreviewed():
    records, summary = MODULE.apply_proposals(_annotations(), [_proposal()])

    event = records[1]
    assert event["event_label"] == "litter"
    assert event["release_interval"] == {
        "start_frame": 35,
        "end_frame": 37,
    }
    assert event["admissible_routes"][0]["person_id"] == "person:12"
    assert set(event["review"].values()) == {"unreviewed"}
    assert event["ai_proposal"]["status"] == "pending_human_review"
    assert summary["human_review_states_promoted"] == 0


def test_not_litter_proposal_clears_route_and_release():
    proposal = _proposal()
    proposal["event_label"] = "not_litter"

    records, _ = MODULE.apply_proposals(_annotations(), [proposal])

    assert records[1]["release_interval"] is None
    assert records[1]["admissible_routes"] == []


def test_duplicate_or_human_reviewed_target_is_rejected():
    proposal = _proposal()
    with pytest.raises(AnnotationError, match="duplicate proposal"):
        MODULE.apply_proposals(_annotations(), [proposal, dict(proposal)])
    with pytest.raises(AnnotationError, match="human-reviewed"):
        MODULE.apply_proposals(_annotations(review_state="reviewed"), [proposal])


def test_noncanonical_actor_id_is_rejected():
    proposal = _proposal()
    proposal["admissible_routes"][0]["person_id"] = 12

    with pytest.raises(AnnotationError, match="canonical string"):
        MODULE.apply_proposals(_annotations(), [proposal])
