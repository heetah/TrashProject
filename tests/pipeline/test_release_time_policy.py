from dataclasses import replace
import math

import numpy as np
import pytest

from pipeline.backtrack.costs import ActorObservation, BacktrackCostConfig, compute_c_bc
from pipeline.backtrack.resolver import SmartBacktrackConfig, SmartBacktrackResolver
from pipeline.backtrack.study import StudyConfig
from pipeline.backtrack.trajectory import ReleaseHypothesis, build_release_hypotheses


@pytest.mark.parametrize('fps', [10, 12, 30, 60, 29.97])
@pytest.mark.parametrize('count', [2, 3])
def test_physical_horizon_and_quadratic_prior(fps, count):
    birth = 100
    items = build_release_hypotheses(
        [(100 + 10*i, 100 + 5*i*i) for i in range(count)],
        [birth+i for i in range(count)], birth, 200, fps,
        max_release_back_seconds=1.0,
    )
    assert min(x.frame_index for x in items) == birth - math.floor(fps)
    for item in items:
        dt = (birth-item.frame_index)/fps
        assert dt <= 1.0
        assert item.time_prior_policy == 'seconds_quadratic'
        assert not item.search_truncated
        if dt >= 0:
            expected = (max(0, dt-.25)/.75)**2
            assert item.window_prior_cost == pytest.approx(expected)
            assert item.prior_cost == pytest.approx(expected + (1 if count == 2 else 0))
        else:
            assert item.window_prior_cost == pytest.approx(-.35*dt)


def test_computational_guard_and_video_start_are_distinct():
    kwargs = dict(points_uv=[(100,100),(110,110)], frame_indices=[20,21],
                  birth_frame=20, fps=10, max_release_back_seconds=1.0)
    short = build_release_hypotheses(max_back_frames=3, **kwargs)
    assert all(x.search_truncated for x in short)
    early = build_release_hypotheses(max_back_frames=30, **{
        **kwargs, 'birth_frame':2, 'frame_indices':[2,3]})
    assert min(x.frame_index for x in early) == 0
    assert not any(x.search_truncated for x in early)


def test_single_point_fallback_cannot_become_a_free_match():
    items = build_release_hypotheses([(100,100)], [20], 20, 24, 10,
                                    fallback_prior_cost=7, max_release_back_seconds=1)
    assert len(items) == 1 and items[0].prior_cost == 7


@pytest.mark.parametrize('maximum,soft,weight', [(0,.25,1),(.25,.25,1),(1,-.1,1),
                                                (1,.25,-1),(float('nan'),.25,1),
                                                (1,float('inf'),1),(1,.25,float('nan'))])
def test_invalid_policy_is_rejected(maximum, soft, weight):
    with pytest.raises(ValueError):
        StudyConfig(max_release_back_seconds=maximum, release_soft_seconds=soft,
                    release_time_weight=weight)


def test_normalized_bc_distance_gate_and_soft_time_cost_have_separate_roles():
    config = StudyConfig(normalize_bc_distance_time_by_gate=True,
                         normalized_distance_gate_vehicle=.4,
                         distance_weight=.8, time_weight=.2).resolver_config(10)
    assert config.cost_config.ba_weights == BacktrackCostConfig().ba_weights
    assert config.cost_config.ac_weights == BacktrackCostConfig().ac_weights
    actor = ActorObservation('vehicle', 1, 20, (0,0,60,80), evidence_frame_index=18)
    release = ReleaseHypothesis(20, [100,40], np.eye(2), np.zeros(2), 'ballistic', 0)
    cell = compute_c_bc([release], [actor], 10, cost_config=config.cost_config,
                        normalized_distance_gate=.4)
    assert cell.valid
    assert cell.raw_features['direct_distance'] == pytest.approx(1)
    assert cell.components['direct_distance'] == pytest.approx(.8)
    assert cell.components['time'] == pytest.approx(.16)
    zero = replace(config.cost_config, bc_weights={'direct_distance':0, 'time':0})
    assert not compute_c_bc([replace(release, mean_uv=np.array([100.01,40]))],
                            [actor],10,cost_config=zero,normalized_distance_gate=.4).valid
    late = compute_c_bc(
        [release], [replace(actor, evidence_frame_index=17)], 10,
        cost_config=config.cost_config, normalized_distance_gate=.4,
    )
    assert late.valid
    assert late.raw_features['time'] == pytest.approx(1.2 + 4 * .2 ** 2)
    assert late.components['time'] == pytest.approx(.2 * (1.2 + 4 * .2 ** 2))

    # The former AND hard gate remains available only as an explicit replay.
    legacy = replace(config.cost_config, observation_time_cost_mode='hard')
    assert not compute_c_bc(
        [release], [replace(actor, evidence_frame_index=17)], 10,
        cost_config=legacy, normalized_distance_gate=.4,
    ).valid


def test_production_vehicle_distance_gate_includes_d04_boundary_only():
    actor = ActorObservation(
        "vehicle", 1, 20, (0.0, 0.0, 60.0, 80.0),
        evidence_frame_index=20,
    )
    # Vehicle diagonal is 100 px. The first point is exactly 40 px outside
    # the bbox (D=0.4); the second is just beyond the production boundary.
    at_boundary = ReleaseHypothesis(
        20, [100.0, 40.0], np.eye(2), np.zeros(2), "ballistic", 0,
    )
    outside = replace(at_boundary, mean_uv=np.array([100.01, 40.0]))

    assert compute_c_bc([at_boundary], [actor], fps=10).valid
    assert not compute_c_bc([outside], [actor], fps=10).valid


def test_config_roundtrip_null_and_sidecar_policy():
    from dataclasses import asdict
    study = StudyConfig(max_release_back_seconds=1, normalize_bc_distance_time_by_gate=True)
    assert StudyConfig.from_mapping(asdict(study)) == study
    resolver = SmartBacktrackResolver(10, study.resolver_config(10))
    result = resolver.resolve_task({'litter_id':1,'fps':10,'birth_frame':20,
        'history':[(100,100),(110,110)],'history_frames':[20,21], 'actor_frames':[]})
    assert result.route_type == 'null'
    assert len([r for r in result.routes if r.is_null]) == 1
    diag = next(r.metadata['candidate_diagnostics'] for r in result.routes if r.is_null)
    assert diag['resolver_config']['release_window_semantics'] == 'seconds_quadratic'
    assert all(r['time_prior_policy']=='seconds_quadratic' for r in diag['release_hypotheses'])
    assert SmartBacktrackConfig().max_release_back_seconds is None
