import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.actor_gate import (
    batch_has_active_gate,
    has_fresh_observation,
    is_fresh_observation,
    is_within_ttl,
    refresh_last_observation_frame,
)


def _vehicle(**overrides):
    actor = {"cls": "vehicle", "track_id": 1}
    actor.update(overrides)
    return actor


def test_explicit_cache_is_not_fresh_even_without_observed_flag():
    assert not is_fresh_observation(_vehicle(source="cache"))
    assert not is_fresh_observation(_vehicle(observed=False, source="seg_predict"))


def test_legacy_direct_actor_without_provenance_remains_fresh():
    assert is_fresh_observation(_vehicle())
    assert has_fresh_observation([{"cls": "person"}, _vehicle()])


def test_refresh_does_not_extend_ttl_from_cached_actor():
    last = refresh_last_observation_frame(None, [_vehicle(source="seg_predict")], 10)
    assert last == 10
    last = refresh_last_observation_frame(
        last, [_vehicle(observed=False, source="cache")], 11
    )
    assert last == 10
    assert is_within_ttl(13, last, 3)
    assert not is_within_ttl(14, last, 3)


def test_batch_gate_uses_fresh_observation_and_preserves_initial_state():
    pairs = [
        ([], [_vehicle(observed=False, source="cache")]),
        ([], [_vehicle(observed=True, source="seg_predict")]),
    ]
    assert batch_has_active_gate(pairs, [20, 21], 10, 3)
    assert not batch_has_active_gate(
        [([], [_vehicle(observed=False, source="cache")])], [14], 10, 3
    )
