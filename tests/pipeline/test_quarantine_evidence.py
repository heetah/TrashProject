import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.quarantine_evidence import classify_quarantine_release


def _classify(**overrides):
    values = {
        "observations": 3,
        "required_observations": 3,
        "relative_displacement": 30.0,
        "required_displacement": 25.0,
        "relative_downward": 20.0,
        "required_downward": 15.0,
        "downward_steps": 2,
        "required_downward_steps": 2,
    }
    values.update(overrides)
    return classify_quarantine_release(**values)


def test_first_failure_is_insufficient_observations():
    decision = _classify(observations=2, relative_displacement=0.0)
    assert not decision.release_ready
    assert decision.reason == "insufficient_observations"


def test_release_reasons_are_unit_specific():
    assert _classify(relative_displacement=24.9).reason == (
        "relative_displacement_insufficient"
    )
    assert _classify(relative_downward=14.9).reason == (
        "relative_downward_insufficient"
    )
    assert _classify(downward_steps=1).reason == "downward_steps_insufficient"


def test_inclusive_boundaries_release():
    decision = _classify()
    assert decision.release_ready
    assert decision.reason == "released"
    assert decision.as_dict() == {"release_ready": True, "reason": "released"}
