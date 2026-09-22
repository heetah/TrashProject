import pytest

from replay_release_policy import summarize, wilson


def test_missing_clips_and_unknown_ids_do_not_inflate_numeric_accuracy():
    rows = [dict(case=7,vehicle_id=2,route_type='direct_vehicle'),
            dict(case=7,vehicle_id=99,route_type='direct_vehicle'),
            dict(case=9,vehicle_id=None,route_type='null'),
            dict(case=174,vehicle_id=6,route_type='direct_vehicle')]
    result = summarize(rows,{7,9,12,74,174})
    assert result['numeric_correct'] == 1
    assert result['numeric_denominator'] == 3
    assert result['wrong_vehicle_events'] == 1
    assert result['null_route_events'] == 1
    assert set(result['no_confirmed_cases']) == {12,74}
    assert result['historical_adjusted_correct'] == 3
    assert result['historical_adjustment_cases'] == [74,174]


def test_wilson_all_success_still_has_uncertainty():
    low,high = wilson(40,40)
    assert low == pytest.approx(.9123784,abs=1e-6)
    assert high == pytest.approx(1)
    assert wilson(0,0) is None
