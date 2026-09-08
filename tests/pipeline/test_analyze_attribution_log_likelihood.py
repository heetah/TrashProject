import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_attribution_log_likelihood.py"
SPEC = importlib.util.spec_from_file_location("analyze_attribution_log_likelihood", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows():
    rows = []
    for event, positive_distance in (("a", 0.1), ("b", 0.2), ("c", 0.15)):
        rows.extend([
            {
                "event_key": event,
                "normalized_distance": positive_distance,
                "alpha": 0.0,
                "alpha_gt": 0.0,
                "direction_penalty": 0.0,
                "label": True,
                "is_correct_actor": True,
                "is_release_hit": True,
                "clip_id": event,
                "route_type": "direct_vehicle",
                "actor_type": "vehicle",
                "release_frame": 10,
                "actor_key": ["vehicle", 1],
            },
            {
                "event_key": event,
                "normalized_distance": 0.7,
                "alpha": 1.0,
                "alpha_gt": 0.0,
                "direction_penalty": 1.0,
                "label": False,
                "is_correct_actor": False,
                "is_release_hit": False,
                "clip_id": event,
                "route_type": "direct_vehicle",
                "actor_type": "vehicle",
                "release_frame": 9,
                "actor_key": ["vehicle", 2],
            },
        ])
    return rows


def test_logistic_probability_prefers_small_distance_and_time_gap():
    rows = _rows()
    model = MODULE.fit_logistic(
        rows, ("distance", "time"), alpha_center=0.0, l2=0.01
    )
    probability = MODULE.predict(model, rows)
    assert np.all(probability[::2] > probability[1::2])
    assert model["beta"][1] < 0.0
    assert model["beta"][2] < 0.0


def test_direction_penalty_is_learned_with_expected_sign():
    rows = _rows()
    model = MODULE.fit_logistic(
        rows, ("direction",), alpha_center=0.0, l2=0.01
    )
    probability = MODULE.predict(model, rows)
    assert np.all(probability[::2] > probability[1::2])
    assert model["beta"][1] < 0.0


def test_monotone_fit_never_rewards_a_larger_penalty():
    rows = _rows()
    for row in rows:
        row["label"] = not row["label"]
    model = MODULE.fit_logistic(
        rows,
        ("direction",),
        alpha_center=0.0,
        l2=0.01,
        monotone_penalties=True,
    )
    assert model["beta"][1] <= 0.0


def test_leave_one_event_out_never_trains_on_held_out_event():
    predictions = MODULE.leave_one_event_out(
        _rows(), ("distance", "time"), l2=0.01
    )
    assert len(predictions) == 3
    assert all(row["exact_top1"] for row in predictions)


def test_risk_coverage_orders_by_margin():
    rows = [
        {"event_key": "a", "probability_margin": 0.9, "exact_top1": True, "actor_top1": True},
        {"event_key": "b", "probability_margin": 0.1, "exact_top1": False, "actor_top1": False},
        {"event_key": "c", "probability_margin": 0.5, "exact_top1": True, "actor_top1": True},
        {"event_key": "d", "probability_margin": 0.2, "exact_top1": False, "actor_top1": True},
    ]
    result = MODULE.risk_coverage(rows)
    assert result[0]["accepted"] == 1
    assert result[0]["exact_accuracy"] == 1.0
    assert result[-1]["actual_coverage"] == 1.0


def test_uniform_choice_nll_uses_positive_mass():
    rows = _rows()
    assert MODULE.uniform_choice_nll(rows) == np.log(2.0)
