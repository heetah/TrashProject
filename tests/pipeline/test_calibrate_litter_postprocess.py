import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from calibrate_litter_postprocess import (
    _wilson_interval,
    _write_outputs,
    bbox_iou,
    compare_reports,
    match_event,
    summarize_clip_outputs,
)


def _event():
    return {"litter_bbox_frame": 10,
            "litter_bbox": {"x1": 100, "y1": 100, "x2": 120, "y2": 130}}


def _candidate(frame, bbox, litter_id=None, state=None, reason="birth_anchor"):
    return {"record_type": "litter_candidate", "frame_index": frame,
            "bbox": bbox, "confidence": 0.8,
            "filter_outcome": "passed" if litter_id is not None else "rejected",
            "filter_reason": reason, "tracker_litter_id": litter_id,
            "tracker_state": state}


def test_bbox_iou_and_exact_frame_assignment():
    records = [_candidate(10, [101, 100, 121, 130], 4, "pending"),
               _candidate(11, [110, 108, 130, 138], 4, "confirmed")]
    result = match_event(_event(), records)
    assert bbox_iou([100, 100, 120, 130], [102, 101, 121, 131]) > 0.75
    assert result["correct_litter_id"] == 4
    assert result["id_match_method"] == "exact_frame_tracker_assignment"
    assert result["tracker_confirmed"] is True


def test_rejected_first_box_follows_unique_continuation():
    records = [_candidate(10, [101, 100, 121, 130], reason="vehicle_contained"),
               _candidate(11, [115, 112, 136, 143], 7, "pending"),
               _candidate(12, [130, 130, 151, 161], 7, "confirmed")]
    result = match_event(_event(), records)
    assert result["first_gate_reason"] == "vehicle_contained"
    assert result["correct_litter_id"] == 7
    assert result["id_match_method"] == "forward_continuation"


def test_sentinel_does_not_invent_id():
    result = match_event({"litter_bbox_frame": None, "litter_bbox": None}, [])
    assert result["match_status"] == "detector_miss_sentinel"
    assert result["correct_litter_id"] is None


def test_wilson_and_paired_report(tmp_path):
    assert _wilson_interval(19, 63) == [0.202377, 0.423604]
    baseline = {"events": [{"gt_event_id": "a", "tracker_confirmed": False},
                            {"gt_event_id": "b", "tracker_confirmed": True}],
                "unverified_confirmed_track_count": 2}
    candidate = {"events": [{"gt_event_id": "a", "tracker_confirmed": True},
                             {"gt_event_id": "b", "tracker_confirmed": True}],
                "unverified_confirmed_track_count": 2}
    result = compare_reports(baseline, candidate, bootstrap=100)
    assert result["confirmed_correct_gain_count"] == 1
    assert result["confirmed_correct_loss_count"] == 0
    assert result["unverified_confirmed_tracks_nonincreasing"] is True
    _write_outputs({"events": [{"a": 1}, {"a": 2, "b": 3}]}, tmp_path)
    assert (tmp_path / "calibration_events.csv").exists()


def test_clip_summary_does_not_treat_candidates_as_samples(tmp_path):
    sidecar = tmp_path / "litter_case_1_annotated_litter_candidates.jsonl"
    sidecar.write_text(
        '{"record_type":"run"}\n'
        '{"record_type":"litter_candidate","tracker_litter_id":2,"tracker_state":"confirmed"}\n'
        '{"record_type":"litter_candidate","tracker_litter_id":3,"tracker_state":"confirmed"}\n',
        encoding="utf-8",
    )
    summary = summarize_clip_outputs(
        [{"video_filename": "litter_case_1_annotated.mp4", "video_usable": True}],
        tmp_path,
    )
    assert summary["clip_count"] == 1
    assert summary["confirmed_clip_count"] == 1
