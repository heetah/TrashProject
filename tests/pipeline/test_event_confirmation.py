import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.event_confirmation import evaluate_litter_confirmation


def _evaluate(**overrides):
    values = {
        "vehicle_quarantine_active": False,
        "by_trajectory": False,
        "by_motion": False,
        "vehicle_fast_drop": False,
        "fall_then_stable": False,
    }
    values.update(overrides)
    return evaluate_litter_confirmation(**values)


def test_quarantine_is_hard_veto():
    decision = _evaluate(vehicle_quarantine_active=True, by_trajectory=True)
    assert not decision.confirmed
    assert decision.rule is None
    assert decision.reason == "vehicle_quarantine"


def test_rule_precedence_is_deterministic():
    decision = _evaluate(by_trajectory=True, by_motion=True, vehicle_fast_drop=True)
    assert decision.confirmed
    assert decision.rule == "by_trajectory"
    assert decision.reason == "supported"


def test_fast_drop_and_fall_stable_are_supported_paths():
    assert _evaluate(vehicle_fast_drop=True).rule == "vehicle_fast_drop"
    assert _evaluate(fall_then_stable=True).rule == "fall_then_stable"


def test_missing_confirmation_evidence_is_explicit():
    decision = _evaluate()
    assert not decision.confirmed
    assert decision.reason == "insufficient_confirmation_evidence"
    assert decision.as_dict()["evidence"] == {
        "by_trajectory": False,
        "by_motion": False,
        "vehicle_fast_drop": False,
        "fall_then_stable": False,
    }
