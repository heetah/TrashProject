import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.confirmation_sampling import (
    LEGACY_REASON,
    confirmation_reasons,
    deterministic_sample,
    summarize_sample,
)


def test_sample_is_sorted_evenly_and_deterministic():
    rows = [{"clip_id": value} for value in ("z", "a", "m", "b", "x")]
    first = deterministic_sample(rows, 3)
    second = deterministic_sample(rows, 3)
    assert first == second
    assert [row["clip_id"] for row in first] == ["a", "m", "z"]


def test_explicit_json_reason_is_used():
    row = {"confirmation_evidence": '{"reason":"released"}'}
    assert confirmation_reasons(row) == ["released"]


def test_confirmation_rule_replaces_generic_supported_reason():
    row = {
        "confirmation_evidence": (
            '{"confirmed":true,"rule":"by_trajectory",'
            '"reason":"supported"}'
        )
    }
    assert confirmation_reasons(row) == ["by_trajectory"]


def test_legacy_artifact_is_not_inferred():
    assert confirmation_reasons({"error_stage": "tracker_confirmation_or_quarantine"}) == [
        LEGACY_REASON
    ]


def test_summary_separates_reason_and_stage_counts():
    report = summarize_sample([
        {"clip_id": "a", "error_stage": "tracker", "confirmation_reason": "released"},
        {"clip_id": "b", "error_stage": "tracker"},
    ])
    assert report["reason_source_counts"] == {
        LEGACY_REASON: 1,
        "released": 1,
    }
    assert report["error_stage_counts"] == {"tracker": 2}
