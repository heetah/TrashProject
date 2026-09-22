"""
TDD tests for 4c-specific GlobalLitterTracker behaviour.

The production scripts/ pipeline has a temporal-diff 4th channel that
already suppresses static objects at the detector level.  The tracker's static
suppression thresholds were calibrated for the 3c model and are currently too
aggressive for 4c, blocking legitimate throw-confirms.

Tests define expected behaviour BEFORE thresholds are re-calibrated.
Run with:
    conda run -n rtdetr python -m pytest tests/pipeline/test_litter_tracker_4c.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tracker(**kwargs):
    """Create a GlobalLitterTracker; override any init param via kwargs."""
    from litterTracker import GlobalLitterTracker
    t = GlobalLitterTracker()
    for k, v in kwargs.items():
        setattr(t, k, v)
    return t


def _person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320):
    """Person actor dict compatible with tracker internals."""
    return {'cls': 'person', 'track_id': track_id, 'box': [x1, y1, x2, y2]}


def _vehicle_actor(track_id=5, x1=300, y1=300, x2=500, y2=420):
    """Vehicle actor dict compatible with tracker internals."""
    return {'cls': 'vehicle', 'track_id': track_id, 'box': [x1, y1, x2, y2]}


def _lbox(cx, cy, half=5, conf=0.9):
    """Litter bounding box (x1, y1, x2, y2, conf) centred at (cx, cy)."""
    return (cx - half, cy - half, cx + half, cy + half, conf)


def _get_only_litter(active):
    """Return the single active litter id and data; fail if count != 1."""
    assert len(active) == 1, f"Expected 1 active litter, got {len(active)}"
    l_id = next(iter(active))
    return l_id, active[l_id]


def test_horizontal_confirmation_threshold_can_be_overridden(monkeypatch):
    """The production recovery default remains explicitly overridable."""
    from litterTracker import GlobalLitterTracker, MIN_CONFIRM_HORIZONTAL_DISPLACEMENT

    monkeypatch.setenv("LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT", "1.25")
    tracker = GlobalLitterTracker()
    try:
        assert tracker.min_confirm_horizontal_displacement == pytest.approx(1.25)
    finally:
        tracker.close()

    monkeypatch.delenv("LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT", raising=False)
    tracker = GlobalLitterTracker()
    try:
        assert tracker.min_confirm_horizontal_displacement == MIN_CONFIRM_HORIZONTAL_DISPLACEMENT
    finally:
        tracker.close()


def test_0827_confirmation_recovery_defaults(monkeypatch):
    """Guard the manually reviewed recovery profile promoted to production."""
    names = (
        "LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR",
        "LITTER_MIN_CONFIRM_AGE_VEHICLE",
        "LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE",
        "LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT",
        "LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE",
        "LITTER_MIN_VEHICLE_RELATIVE_SEPARATION",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    from litterTracker import GlobalLitterTracker

    tracker = GlobalLitterTracker()
    try:
        assert tracker.require_birth_thrower_for_confirmation is False
        assert tracker.min_confirm_age_vehicle == 2
        assert tracker.min_confirm_downward_displacement_vehicle == pytest.approx(7.0)
        assert tracker.min_confirm_horizontal_displacement == pytest.approx(1.0)
        assert tracker.max_horiz_to_down_ratio_vehicle == pytest.approx(10.0)
        assert tracker.min_vehicle_relative_separation == pytest.approx(0.0)
    finally:
        tracker.close()


def test_0827_pretracker_recovery_defaults(monkeypatch):
    """Guard the streak and camera-shake defaults in the same profile."""
    monkeypatch.delenv("LITTER_FP_STREAK_RATIO", raising=False)
    monkeypatch.delenv("LITTER_FP_STREAK_MIN_OBSERVATIONS", raising=False)
    monkeypatch.delenv("LITTER_ALLOW_SHAKE_CANDIDATES", raising=False)

    from pipeline.detect import _allow_shake_candidates
    from pipeline.geometry import (
        LITTER_FP_STREAK_MIN_OBSERVATIONS,
        LITTER_FP_STREAK_RATIO,
    )

    assert LITTER_FP_STREAK_RATIO == pytest.approx(10.0)
    assert LITTER_FP_STREAK_MIN_OBSERVATIONS == 2
    assert _allow_shake_candidates() is True

    monkeypatch.setenv("LITTER_ALLOW_SHAKE_CANDIDATES", "0")
    assert _allow_shake_candidates() is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: thrown litter should confirm
# ─────────────────────────────────────────────────────────────────────────────

class TestThrownLitterConfirm:

    def setup_method(self):
        self.tracker = _make_tracker()

    def teardown_method(self):
        self.tracker.close()

    def test_thrown_litter_confirms_by_age_3(self):
        """
        A thrown litter that moves 8px/frame (downward + horizontal) with a nearby
        person should transition pending → confirmed by the 3rd update.

        With current 4c code (min_history_span_for_confirm=18.0) the history span
        after 3 frames is ~16px < 18px, so is_static_candidate=True blocks confirm.
        After fix (min_history_span_for_confirm ≤ 14.0), 16px passes.
        """
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]

        # Frame 0: litter appears near person
        active, _ = tracker.update([_lbox(500, 300)], actors, frame_index=0)
        l_id, data = _get_only_litter(active)
        assert data['state'] == 'pending'

        # Frame 1: litter moved 8px down + 8px right
        active, _ = tracker.update([_lbox(508, 308)], actors, frame_index=1)
        assert l_id in active, "Litter should still be tracked"

        # Frame 2: litter moved another 8px down + 8px right
        active, _ = tracker.update([_lbox(516, 316)], actors, frame_index=2)
        assert l_id in active, "Litter should still be tracked"
        state = active[l_id]['state']
        assert state == 'confirmed', (
            f"Thrown litter (16px span, 3 frames, nearby actor) should be confirmed. "
            f"Got state='{state}'. "
            "Fix: lower min_history_span_for_confirm from 18.0 to ≤ 14.0."
        )

    def test_thrown_litter_has_thrower_assigned(self):
        """Confirmed throw must have a thrower_key (not None)."""
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]

        tracker.update([_lbox(500, 300)], actors, frame_index=0)
        tracker.update([_lbox(508, 308)], actors, frame_index=1)
        active, _ = tracker.update([_lbox(516, 316)], actors, frame_index=2)

        assert len(active) >= 1
        l_id = next(iter(active))
        assert active[l_id]['thrower_key'] is not None, (
            "Confirmed litter must have thrower_key assigned."
        )

    def test_larger_displacement_confirms_faster(self):
        """
        Litter with 12px/frame movement (span=24px after 3 frames) should still
        confirm even with current threshold, and is also valid with fixed threshold.
        """
        tracker = self.tracker
        actors = [_person_actor(track_id=2, x1=430, y1=200, x2=470, y2=320)]

        tracker.update([_lbox(500, 300)], actors, frame_index=0)
        tracker.update([_lbox(512, 312)], actors, frame_index=1)
        active, _ = tracker.update([_lbox(524, 324)], actors, frame_index=2)

        l_id = next(iter(active))
        # span = 24px > 18px (current threshold) AND > 14px (fixed threshold)
        # Both should confirm
        assert active[l_id]['state'] == 'confirmed', (
            "24px span throw should confirm regardless of threshold fix."
        )

    def test_three_point_strong_vehicle_descent_confirms_despite_large_bbox(self):
        """Case-167 guard: physical descent is evidence despite size scaling."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=30)
        vehicle = _vehicle_actor(
            track_id=3, x1=1180, y1=500, x2=1390, y2=720
        )
        try:
            points = [(1287, 659.5), (1291.5, 704), (1294.5, 740.5)]
            active = {}
            for frame_index, (cx, cy) in zip((139, 140, 141), points):
                active, _ = tracker.update(
                    [_lbox(cx, cy, half=20)], [vehicle], frame_index=frame_index
                )
            assert any(item['state'] == 'confirmed' for item in active.values())
        finally:
            tracker.close()

    def test_motion_consistent_shape_change_keeps_one_pending_track(self):
        """Motion blur may change aspect ratio without changing identity."""
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=850, y1=300, x2=900, y2=520)]

        active, _ = tracker.update(
            [_lbox(880, 510, half=15)], actors, frame_index=0
        )
        litter_id = next(iter(active))
        # Tall blurred box: height changes by >60%, but its centre motion is
        # bounded by the combined source/target object scale.
        active, _ = tracker.update(
            [(740, 420, 765, 481, 0.9)], actors, frame_index=1
        )

        assert litter_id in active
        assert active[litter_id]['age'] == 2

    def test_motion_prediction_rejects_shape_changed_off_path_box(self):
        """A nearby shape change away from predicted motion starts a new ID."""
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=850, y1=300, x2=900, y2=520)]
        tracker.update([_lbox(880, 510, half=15)], actors, frame_index=0)
        active, _ = tracker.update(
            [(740, 420, 765, 481, 0.9)], actors, frame_index=1
        )
        original_id = min(active)
        active, _ = tracker.update(
            [(790, 650, 852, 688, 0.9)], actors, frame_index=2
        )

        assert original_id in active
        assert active[original_id]['age'] == 2
        assert len(active) == 2

    def test_actorless_six_point_gravity_arc_confirms_with_null_route(self):
        """A physical object event must not depend on successful attribution."""
        tracker = self.tracker
        points = [
            (880, 510, 15),
            (752, 455, 30),
            (650, 435, 25),
            (550, 430, 14),
            (235, 660, 15),
            (52, 770, 16),
        ]
        active = {}
        for frame_index, (cx, cy, half) in zip((0, 1, 2, 3, 7, 12), points):
            active, _ = tracker.update(
                [_lbox(cx, cy, half=half)], [], frame_index=frame_index
            )

        assert any(item['state'] == 'confirmed' for item in active.values())
        event = tracker.get_litter_events()[0]
        assert event['thrower_key'] is None

    def test_actorless_short_linear_track_does_not_confirm(self):
        """Missing attribution is allowed only with stronger object physics."""
        tracker = self.tracker
        active = {}
        for frame_index, (cx, cy) in enumerate(((500, 300), (530, 315), (560, 330))):
            active, _ = tracker.update(
                [_lbox(cx, cy)], [], frame_index=frame_index
            )
        assert all(item['state'] == 'pending' for item in active.values())


# ─────────────────────────────────────────────────────────────────────────────
# Tests: static litter must never confirm
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticLitterNeverConfirms:

    def setup_method(self):
        self.tracker = _make_tracker()

    def teardown_method(self):
        self.tracker.close()

    def test_truly_static_litter_never_confirms(self):
        """Litter with ≤ 2px jitter over 20 frames must never reach 'confirmed'."""
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]

        # Simulate 20 frames of near-static litter (±1px noise)
        positions = [(500, 300), (501, 300), (500, 301), (499, 300), (500, 299),
                     (501, 301), (500, 300), (499, 301), (500, 300), (501, 300),
                     (500, 301), (500, 300), (499, 300), (500, 301), (501, 300),
                     (500, 300), (499, 301), (500, 300), (501, 300), (500, 301)]

        l_id = None
        for fi, (cx, cy) in enumerate(positions):
            active, _ = tracker.update([_lbox(cx, cy)], actors, frame_index=fi)
            if l_id is None and active:
                l_id = next(iter(active))
            if l_id in active:
                state = active[l_id]['state']
                assert state != 'confirmed', (
                    f"Static litter (≤2px jitter) must never confirm. "
                    f"Got confirmed at frame {fi}."
                )

    def test_static_litter_becomes_stationary_locked(self):
        """After stationary_lock_age frames with tiny span, litter gets locked."""
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]

        # Run for stationary_lock_age + 2 frames with ≤1px movement
        lock_age = tracker.stationary_lock_age
        l_id = None
        for fi in range(lock_age + 3):
            cx = 500 + (fi % 2)  # 0 or 1px oscillation
            active, _ = tracker.update([_lbox(cx, 300)], actors, frame_index=fi)
            if l_id is None and active:
                l_id = next(iter(active))

        assert l_id in active, "Litter should still be tracked"
        assert active[l_id].get('stationary_locked', False), (
            f"After {lock_age + 3} frames with ≤1px span, "
            "litter should be stationary_locked."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: stationary lock prevents confirm after lock
# ─────────────────────────────────────────────────────────────────────────────

class TestStationaryLockPreventsConfirm:

    def setup_method(self):
        self.tracker = _make_tracker()

    def teardown_method(self):
        self.tracker.close()

    def test_locked_litter_cannot_confirm_on_big_jump(self):
        """
        Litter that is stationary_locked must NOT confirm even if a later
        detection shows a large displacement (detector jitter after lock).
        """
        tracker = self.tracker
        actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]

        # Phase 1: static litter for lock_age + 1 frames → becomes locked
        lock_age = tracker.stationary_lock_age
        l_id = None
        for fi in range(lock_age + 1):
            active, _ = tracker.update([_lbox(500, 300)], actors, frame_index=fi)
            if l_id is None and active:
                l_id = next(iter(active))

        # Verify locked
        if l_id in active:
            assert active[l_id].get('stationary_locked', False), (
                "Litter should be stationary_locked after static phase."
            )

        # Phase 2: big jump — should NOT produce confirm
        fi = lock_age + 2
        active, _ = tracker.update([_lbox(540, 350)], actors, frame_index=fi)

        # The litter may be re-created as a NEW id (big dist > distance_threshold)
        # or tracked as same id — in either case no confirmed state
        for _, data in active.items():
            assert data['state'] != 'confirmed', (
                "Locked litter (or new litter near locked position) must not "
                "confirm on a single large jump."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: no thrower means no confirm
# ─────────────────────────────────────────────────────────────────────────────

class TestNoThrowerPreventsConfirm:

    def setup_method(self):
        self.tracker = _make_tracker()

    def teardown_method(self):
        self.tracker.close()

    def test_no_actors_prevents_confirm(self):
        """Moving litter without any actors must not confirm."""
        tracker = self.tracker
        actors = []   # no one around

        tracker.update([_lbox(500, 300)], actors, frame_index=0)
        tracker.update([_lbox(508, 308)], actors, frame_index=1)
        active, _ = tracker.update([_lbox(516, 316)], actors, frame_index=2)

        for _, data in active.items():
            assert data['state'] != 'confirmed', (
                "Litter without any actors must not confirm."
            )

    def test_very_far_actor_cannot_be_fallback_thrower(self):
        """
        An actor that is 800px away from the litter should not qualify as
        fallback thrower even with the fixed (slightly relaxed) threshold.
        """
        tracker = self.tracker
        # Very distant person (off to the side)
        actors = [_person_actor(track_id=99, x1=1000, y1=500, x2=1040, y2=700)]

        tracker.update([_lbox(100, 300)], actors, frame_index=0)
        tracker.update([_lbox(108, 308)], actors, frame_index=1)
        active, _ = tracker.update([_lbox(116, 316)], actors, frame_index=2)

        for _, data in active.items():
            assert data['state'] != 'confirmed', (
                "Very distant actor (>700px away) must not be assigned as thrower."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: threshold calibration for 4c
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdCalibration4c:

    def test_min_history_span_for_confirm_leq_14(self):
        """
        min_history_span_for_confirm must be ≤ 14.0 for 4c pipeline.

        The 4c model detects litter at the moment of motion (change channel
        is bright during throw).  A typical throw spans 8-12px/frame; over
        3 frames the max span is ~16px.  With threshold = 18.0 this is
        mis-classified as static, blocking confirm.
        Fix: set threshold ≤ 14.0 (e.g. 10.0).
        """
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            assert t.min_history_span_for_confirm <= 14.0, (
                f"min_history_span_for_confirm={t.min_history_span_for_confirm}, "
                "must be ≤ 14.0 for 4c pipeline.  "
                "Current value blocks legitimate throws with 15-17px span."
            )
        finally:
            t.close()

    def test_thrower_fallback_score_limit_geq_1_20(self):
        """
        thrower_fallback_score_limit must be ≥ 1.20 to allow actors
        that are slightly farther than 1× threshold distance.

        At 1.10 a person standing 10% past the score threshold cannot be
        assigned as thrower, even though they are clearly nearby.
        Fix: raise to ≥ 1.20 (e.g. 1.25).
        """
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            assert t.thrower_fallback_score_limit >= 1.20, (
                f"thrower_fallback_score_limit={t.thrower_fallback_score_limit}, "
                "must be ≥ 1.20.  "
                "Value 1.10 is too restrictive for 4c scene geometry."
            )
        finally:
            t.close()

    def test_stationary_lock_span_leq_9(self):
        """
        stationary_lock_span must be ≤ 9.0.

        A value of 12.0 means objects spanning up to 12px over lock_age frames
        are permanently locked.  A real throw spanning 10-11px over that period
        would be wrongly locked.
        Fix: lower to ≤ 9.0 (e.g. 7.0 or 8.0).
        """
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            assert t.stationary_lock_span <= 9.0, (
                f"stationary_lock_span={t.stationary_lock_span}, "
                "must be ≤ 9.0 for 4c pipeline."
            )
        finally:
            t.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: fps normalization (Phase 1)
#
# Model: pixel thresholds are fps-invariant physical facts and must NOT change
# with fps.  Only frame-window / age / per-frame velocity-cap params scale with
# fps.  Reference fps = 10, so at 10fps behaviour is byte-identical to the old
# (fps-unaware) code → zero regression on the existing 10fps-tuned clips.  At
# 30fps the same physical throw moves ~1/3 px per frame but persists ~3x frames,
# so a longer frame window accumulates the same pixel displacement WITHOUT
# lowering any threshold.
# ─────────────────────────────────────────────────────────────────────────────

class TestFpsNormalization:

    def test_default_fps_is_reference(self):
        """No-arg tracker must run at the reference fps (scale == 1.0)."""
        from litterTracker import GlobalLitterTracker, REF_FPS
        t = GlobalLitterTracker()
        try:
            assert t.fps == REF_FPS
            assert t._frame_scale == 1.0
        finally:
            t.close()

    def test_10fps_window_params_equal_base_constants(self):
        """At the reference fps every scaled param must equal its base constant."""
        from litterTracker import (
            GlobalLitterTracker, TRAJECTORY_HISTORY_LEN, MAX_MISSED_FRAMES,
            MIN_CONFIRM_AGE, MIN_CONFIRM_AGE_VEHICLE, STATIC_CANDIDATE_MIN_AGE,
            STATIONARY_LOCK_AGE, FALL_STABLE_MIN_AGE, FALL_STABLE_TAIL_WINDOW,
            FAST_DROP_MAX_FRAME_GAP, MAX_VEHICLE_THROWER_STEP_PX,
        )
        t = GlobalLitterTracker(fps=10)
        try:
            assert t.trajectory_history_len == TRAJECTORY_HISTORY_LEN
            assert t.max_missed_frames == MAX_MISSED_FRAMES
            assert t.min_confirm_age == MIN_CONFIRM_AGE
            assert t.min_confirm_age_vehicle == MIN_CONFIRM_AGE_VEHICLE
            assert t.static_candidate_min_age == STATIC_CANDIDATE_MIN_AGE
            assert t.stationary_lock_age == STATIONARY_LOCK_AGE
            assert t.fall_stable_min_age == FALL_STABLE_MIN_AGE
            assert t.fall_stable_tail_window == FALL_STABLE_TAIL_WINDOW
            assert t.fast_drop_max_frame_gap == FAST_DROP_MAX_FRAME_GAP
            assert t.max_vehicle_thrower_step_px == MAX_VEHICLE_THROWER_STEP_PX
        finally:
            t.close()

    def test_30fps_scales_persistence_up_but_not_age(self):
        """
        At 30fps only the time/persistence params scale up (longer track memory so
        sparse detections survive detection gaps).  age/maturity thresholds count
        DETECTIONS, not frames, so they must stay at base values — inflating them
        makes confirm unreachable for sparsely-detected small fast objects (case 13).
        The per-frame velocity cap shrinks with fps.
        """
        from litterTracker import (
            GlobalLitterTracker, MAX_VEHICLE_THROWER_STEP_PX,
            MIN_CONFIRM_AGE, MIN_CONFIRM_AGE_VEHICLE, STATIC_CANDIDATE_MIN_AGE,
            STATIONARY_LOCK_AGE, FALL_STABLE_MIN_AGE,
        )
        t = GlobalLitterTracker(fps=30)
        try:
            assert t._frame_scale == 3.0
            # persistence / window params scale up
            assert t.trajectory_history_len == 45       # 15 * 3
            assert t.max_missed_frames == 30            # 10 * 3
            # age / maturity thresholds stay at base (detection-count based)
            assert t.min_confirm_age == MIN_CONFIRM_AGE
            assert t.min_confirm_age_vehicle == MIN_CONFIRM_AGE_VEHICLE
            assert t.static_candidate_min_age == STATIC_CANDIDATE_MIN_AGE
            assert t.stationary_lock_age == STATIONARY_LOCK_AGE
            assert t.fall_stable_min_age == FALL_STABLE_MIN_AGE
            # per-frame velocity cap shrinks (px/frame smaller at high fps)
            assert t.max_vehicle_thrower_step_px < MAX_VEHICLE_THROWER_STEP_PX
            assert abs(t.max_vehicle_thrower_step_px - MAX_VEHICLE_THROWER_STEP_PX / 3.0) < 1e-6
        finally:
            t.close()

    def test_pixel_thresholds_are_fps_invariant(self):
        """Pixel-domain thresholds must be identical regardless of fps."""
        from litterTracker import GlobalLitterTracker
        t10 = GlobalLitterTracker(fps=10)
        t30 = GlobalLitterTracker(fps=30)
        try:
            assert t10.min_history_span_for_confirm == t30.min_history_span_for_confirm
            assert t10.stationary_lock_span == t30.stationary_lock_span
            assert t10.thrower_fallback_score_limit == t30.thrower_fallback_score_limit
        finally:
            t10.close()
            t30.close()

    def test_30fps_throw_confirms_with_small_per_frame_motion(self):
        """
        A 30fps thrown object moves only ~4px/frame but persists many frames.
        With the longer (fps-scaled) window it must still reach 'confirmed'.
        Regression guard for the 30fps window-throw FNs (cases 13/15/21).
        """
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=30)
        try:
            actors = [_person_actor(track_id=7, x1=430, y1=200, x2=470, y2=320)]
            cx, cy = 500, 300
            l_id = None
            confirmed = False
            for fi in range(14):
                active, _ = tracker.update([_lbox(cx, cy)], actors, frame_index=fi)
                if l_id is None and active:
                    l_id = next(iter(active))
                if l_id in active and active[l_id]['state'] == 'confirmed':
                    confirmed = True
                    break
                cx += 4   # ~4px/frame right  (≈120px/s, a real throw at 30fps)
                cy += 4   # ~4px/frame down
            assert confirmed, (
                "A 30fps throw (~4px/frame, near a person) must confirm with the "
                "fps-scaled window. FN recovery for cases 13/15/21."
            )
        finally:
            tracker.close()

    def test_30fps_static_litter_still_never_confirms(self):
        """Precision guard: a near-static object at 30fps must never confirm."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=30)
        try:
            actors = [_person_actor(track_id=8, x1=430, y1=200, x2=470, y2=320)]
            l_id = None
            for fi in range(40):            # > stationary_lock_age (30) at 30fps
                cx = 500 + (fi % 2)         # ≤1px oscillation
                active, _ = tracker.update([_lbox(cx, 300)], actors, frame_index=fi)
                if l_id is None and active:
                    l_id = next(iter(active))
                if l_id in active:
                    assert active[l_id]['state'] != 'confirmed', (
                        f"Static litter must never confirm, even at 30fps "
                        f"(confirmed at frame {fi})."
                    )
        finally:
            tracker.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: vehicle co-motion discrimination (Phase 2)
#
# A litter candidate that rides WITH a moving vehicle (headlight, mirror, body
# part — case 19 truck-light FP) shares the vehicle's per-frame velocity:
# parallel direction, near-equal magnitude, tiny relative motion.  A real object
# thrown from a moving vehicle decouples — gravity pulls it down and drag bleeds
# its horizontal speed, so its velocity diverges from the vehicle's.  Co-moving
# candidates must NOT confirm; decoupled throws must still confirm.
# ─────────────────────────────────────────────────────────────────────────────

class TestVehicleCoMotion:
    # NOTE: vehicle-part / co-motion / streak FP discrimination now lives in
    # detect.py preprocessing (smallFunction.litter_candidate_is_vehicle_fp) — the
    # tracker only tracks and no longer rejects candidates.  See
    # TestLitterCandidateFilter for those checks.  The tests below verify the
    # tracker still CONFIRMS real throws (its tracking job).

    def test_decoupled_throw_from_moving_vehicle_confirms(self):
        """Object thrown from a moving vehicle decouples (falls) and must confirm."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        try:
            vx, vy = 300, 300
            lx, ly = 520, 360
            confirmed = False
            for fi in range(10):
                veh = _vehicle_actor(track_id=5, x1=vx, y1=vy, x2=vx + 200, y2=vy + 120)
                active, _ = tracker.update([_lbox(lx, ly)], [veh], frame_index=fi)
                for d in active.values():
                    if d['state'] == 'confirmed':
                        confirmed = True
                vx += 8                 # vehicle drives horizontally
                lx += 4; ly += 9        # litter decouples: drag-slowed horizontal + gravity fall
            assert confirmed, (
                "A real throw decoupling from the vehicle (velocity diverges via "
                "gravity/drag) must still confirm — co-motion guard must not block it."
            )
        finally:
            tracker.close()

    def test_comotion_inert_without_vehicle(self):
        """With no vehicle nearby, the co-motion guard must not affect person throws."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        try:
            actors = [_person_actor(track_id=1, x1=430, y1=200, x2=470, y2=320)]
            cx, cy = 500, 300
            confirmed = False
            for fi in range(6):
                active, _ = tracker.update([_lbox(cx, cy)], actors, frame_index=fi)
                for d in active.values():
                    if d['state'] == 'confirmed':
                        confirmed = True
                cx += 9; cy += 9
            assert confirmed, (
                "Person throw with no vehicle nearby must confirm; co-motion guard "
                "must stay inert."
            )
        finally:
            tracker.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: sparse-detection per-frame velocity (case 13 — small fast object thrown
# from a vehicle, detected only every ~12 frames).  Each detected step spans many
# frames so its absolute displacement is large, but per-frame velocity is small
# and physical.  The step-velocity guard must judge per-frame (÷ frame gap), not
# per-step, so these throws confirm instead of being rejected as vehicle teleports.
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseDetectionPerFrameVelocity:

    def test_sparse_large_step_throw_confirms(self):
        """A vehicle throw detected sparsely (big per-step, small per-frame) must confirm."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=30)
        try:
            # Parked vehicle = thrower; object falls ~12px/frame, detected every 12 frames.
            veh = _vehicle_actor(track_id=5, x1=300, y1=100, x2=500, y2=220)
            frames = [150, 162, 174, 186]
            xs = [510, 522, 534, 546]               # mild horizontal drift
            ys = [230, 230 + 144, 230 + 288, 230 + 432]   # 144px per detection = 12px/frame
            confirmed = False
            for i, fi in enumerate(frames):
                active, _ = tracker.update([_lbox(xs[i], ys[i])], [veh], frame_index=fi)
                for d in active.values():
                    if d['state'] == 'confirmed':
                        confirmed = True
            assert confirmed, (
                "A sparsely-detected throw (big per-step displacement but ~12px/frame) "
                "must confirm; the step-velocity guard must normalise by frame gap, not "
                "reject it as a vehicle teleport (case 13 white-tissue FN)."
            )
        finally:
            tracker.close()

    def test_per_frame_teleport_still_rejected(self):
        """A genuine per-frame teleport (huge displacement in one frame) must NOT confirm."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=30)
        try:
            veh = _vehicle_actor(track_id=5, x1=300, y1=100, x2=500, y2=220)
            # consecutive frames, ~150px jump per single frame → vehicle-body artifact
            frames = [150, 151, 152, 153]
            xs = [510, 560, 610, 660]
            ys = [230, 380, 530, 680]
            confirmed = False
            for i, fi in enumerate(frames):
                active, _ = tracker.update([_lbox(xs[i], ys[i])], [veh], frame_index=fi)
                for d in active.values():
                    if d['state'] == 'confirmed':
                        confirmed = True
            assert not confirmed, (
                "A per-frame teleport (~150px in a single frame) is a vehicle artifact "
                "and must still be rejected by the per-frame velocity cap."
            )
        finally:
            tracker.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: vehicle-part containment + horizontal-streak FP suppression (Phase 2b)
#
# co-motion / separation compare against the vehicle CENTRE displacement, which
# fails for large perspective-scaling vehicles (a truck headlight rides the body
# but the body's centre moves differently than its edge — case 19).  Two robust,
# physically-grounded guards complement it:
#   - containment: at the confirm frame a real dropped object has EXITED the
#     vehicle (overlap≈0); a rigid part stays almost entirely inside (overlap≈1).
#   - horizontal streak: a real fall is gravity-driven (meaningful downward); a
#     vehicle/scooter smeared horizontally across the frame is not.
# Real vertical throws (overlap≈0, downward-dominant) must still confirm.
# ─────────────────────────────────────────────────────────────────────────────

class TestVehiclePartAndStreakFP:
    # The FP *rejection* (vehicle-part / streak) now lives in the detect-stage
    # filter; see TestLitterCandidateFilter.  This test keeps the tracker's
    # positive behaviour: a real drop that EXITS the vehicle still confirms.

    def test_object_exiting_vehicle_box_confirms(self):
        """A dropped object that EXITS the vehicle box (overlap→0) must confirm."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        try:
            # Static vehicle; object starts at its bottom edge and falls clear of it.
            veh = _vehicle_actor(track_id=5, x1=400, y1=200, x2=600, y2=360)
            confirmed = False
            lx, ly = 500, 350
            for i in range(8):
                active, _ = tracker.update([_lbox(lx, ly)], [veh], frame_index=i)
                for d in active.values():
                    if d['state'] == 'confirmed':
                        confirmed = True
                lx += 5; ly += 16      # falls down out of the box, mild horizontal
            assert confirmed, (
                "An object that falls clear of the vehicle box (overlap→0, downward-"
                "dominant) is a real drop and must confirm."
            )
        finally:
            tracker.close()


class TestTemporalVisualBridgeAndVehicleQuarantine:

    def test_two_detector_anchors_allow_one_temporal_component_bridge(self):
        """A third visible moving frame may satisfy, but not bypass, quarantine."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            active = {}
            for frame_index, (cx, cy) in enumerate(((500, 300), (504, 340))):
                frame = np.zeros((720, 960, 3), dtype=np.uint8)
                frame[cy - 6:cy + 6, cx - 6:cx + 6] = 255
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter], [vehicle], frame_index=frame_index, frame=frame,
                    quarantined_litters=[litter],
                )
            litter_id = next(iter(active))

            frame = np.zeros((720, 960, 3), dtype=np.uint8)
            frame[374:386, 502:514] = 255
            active, _ = tracker.update(
                [], [vehicle], frame_index=2, frame=frame,
            )

            assert list(active) == [litter_id]
            assert active[litter_id]['detector_observation_count'] == 2
            assert active[litter_id]['history_sources'][-1] == 'visual_bridge'
            assert active[litter_id]['state'] == 'confirmed'
            task = tracker._smart_tasks[litter_id]
            assert task['history_sources'] == [
                'accepted_tracker', 'accepted_tracker', 'accepted_tracker'
            ]
            assert task['history_provenance'] == [
                'detector', 'detector', 'visual_bridge'
            ]
            assert task['history_lineage'][-1]['independent_measurement'] is False
        finally:
            tracker.close()


    @staticmethod
    def _moving_frame(frame_index):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        cx = 320 + 25 * frame_index
        cy = 220 + 25 * frame_index
        value = 255 if frame_index % 2 == 0 else 128
        frame[cy - 20:cy + 20, cx - 20:cx + 20] = value
        return frame

    def test_large_seed_requires_four_visual_observations_before_confirmation(self):
        """A large high-confidence seed may bridge, but cannot confirm early."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        person = _person_actor(
            track_id=1, x1=250, y1=80, x2=350, y2=280,
        )
        try:
            active, _ = tracker.update(
                [(300, 200, 340, 240, 0.9)],
                [person],
                frame_index=0,
                frame=self._moving_frame(0),
            )
            litter_id = next(iter(active))
            for frame_index in range(1, 4):
                active, _ = tracker.update(
                    [], [person], frame_index=frame_index,
                    frame=self._moving_frame(frame_index),
                )
                assert active[litter_id]['state'] == 'pending'

            active, _ = tracker.update(
                [], [person], frame_index=4,
                frame=self._moving_frame(4),
            )
            assert active[litter_id]['state'] == 'confirmed'
            assert active[litter_id]['detector_observation_count'] == 1
            assert active[litter_id]['history_sources'].count('visual_bridge') == 4
        finally:
            tracker.close()

    def test_small_single_seed_cannot_start_visual_chain(self):
        """Small objects retain the stricter two-detector single-bridge rule."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        person = _person_actor(track_id=1)
        try:
            first = np.zeros((600, 800, 3), dtype=np.uint8)
            first[295:305, 495:505] = 255
            active, _ = tracker.update(
                [_lbox(500, 300)], [person], frame_index=0, frame=first,
            )
            litter_id = next(iter(active))
            second = np.zeros_like(first)
            second[305:315, 505:515] = 128
            active, _ = tracker.update(
                [], [person], frame_index=1, frame=second,
            )
            assert active[litter_id]['age'] == 1
            assert active[litter_id]['history_sources'] == ['detector']
        finally:
            tracker.close()

    def test_visual_bridge_cannot_authorize_later_containment_exit_handoff(self):
        """Synthetic evidence must not merge a later ordinary release segment."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(track_id=5, x1=300, y1=100, x2=700, y2=600)
        try:
            for frame_index, (cx, cy) in enumerate(((500, 300), (504, 305))):
                frame = np.zeros((720, 960, 3), dtype=np.uint8)
                frame[cy - 6:cy + 6, cx - 6:cx + 6] = 255
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter], [vehicle], frame_index=frame_index, frame=frame,
                    quarantined_litters=[litter],
                )
            original_id = next(iter(active))
            bridge_frame = np.zeros((720, 960, 3), dtype=np.uint8)
            bridge_frame[304:316, 502:514] = 255
            active, _ = tracker.update(
                [], [vehicle], frame_index=2, frame=bridge_frame,
            )
            assert active[original_id]['history_sources'][-1] == 'visual_bridge'

            ordinary = _lbox(540, 430, half=6)
            active, _ = tracker.update(
                [ordinary], [vehicle], frame_index=3,
                frame=np.zeros((720, 960, 3), dtype=np.uint8),
            )
            assert len(active) == 2
            assert active[original_id]['bbox'] != ordinary
        finally:
            tracker.close()

    def test_stale_quarantine_track_cannot_absorb_new_release(self):
        """A long detector gap starts a new identity even at the same location."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=30)
        vehicle = _vehicle_actor(track_id=5, x1=300, y1=100, x2=700, y2=600)
        try:
            old = _lbox(500, 300, half=6)
            active, _ = tracker.update(
                [old], [vehicle], frame_index=0,
                quarantined_litters=[old],
            )
            old_id = next(iter(active))
            new = _lbox(502, 302, half=6)
            active, _ = tracker.update(
                [new], [vehicle], frame_index=30,
                quarantined_litters=[new],
            )
            assert len(active) == 2
            assert active[old_id]['age'] == 1
        finally:
            tracker.close()

    def test_same_carrier_comotion_remains_pending(self):
        """A vehicle part can accumulate history but can never self-confirm."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        try:
            for frame_index in range(5):
                offset = 12 * frame_index
                vehicle = _vehicle_actor(
                    track_id=5,
                    x1=400 + offset,
                    y1=200,
                    x2=600 + offset,
                    y2=420,
                )
                litter = _lbox(500 + offset, 300, half=6)
                active, _ = tracker.update(
                    [litter],
                    [vehicle],
                    frame_index=frame_index,
                    quarantined_litters=[litter],
                )
                _, data = _get_only_litter(active)
                assert data['state'] == 'pending'
                assert data['vehicle_quarantine_active'] is True
                assert data.get('vehicle_quarantine_released') is False
        finally:
            tracker.close()

    def test_three_point_relative_descent_releases_then_confirms(self):
        """A sustained in-box drop is released to the normal confirmation gates."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=400, y1=180, x2=620, y2=460,
        )
        try:
            released_data = None
            for frame_index, (cx, cy) in enumerate(
                ((500, 250), (504, 280), (509, 320))
            ):
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter],
                    [vehicle],
                    frame_index=frame_index,
                    quarantined_litters=[litter],
                )
                _, released_data = _get_only_litter(active)
            assert released_data is not None
            assert released_data['vehicle_quarantine_active'] is False
            assert released_data['vehicle_quarantine_released'] is True
            assert released_data['vehicle_quarantine_release_frame'] == 2
            # Quarantine release is not confirmation. A subsequent ordinary
            # observation must still satisfy actor and trajectory gates.
            assert released_data['state'] == 'pending'

            released_litter = _lbox(515, 480, half=6)
            active, _ = tracker.update(
                [released_litter],
                [vehicle],
                frame_index=3,
            )
            _, final_data = _get_only_litter(active)
            assert final_data['state'] == 'confirmed'
        finally:
            tracker.close()

    def test_two_observations_cannot_release_even_after_large_jump(self):
        """One detector jump is insufficient physical evidence for release."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            final_data = None
            for frame_index, (cx, cy) in enumerate(((450, 180), (500, 420))):
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter],
                    [vehicle],
                    frame_index=frame_index,
                    quarantined_litters=[litter],
                )
                _, final_data = _get_only_litter(active)
            assert final_data is not None
            assert final_data['state'] == 'pending'
            assert final_data['vehicle_quarantine_active'] is True
        finally:
            tracker.close()

    def test_late_box_restarts_quarantine_evidence_segment(self):
        """A long detector gap cannot join a vehicle part to another box."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            for frame_index, (cx, cy) in (
                (0, (450, 180)),
                (1, (455, 190)),
                (6, (500, 420)),
            ):
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter],
                    [vehicle],
                    frame_index=frame_index,
                    quarantined_litters=[litter],
                )
            _, data = _get_only_litter(active)
            assert data['state'] == 'pending'
            assert data['vehicle_quarantine_active'] is True
            assert data['vehicle_quarantine_frames'] == [6]
        finally:
            tracker.close()

    def test_ordinary_exit_starts_independent_pending_track(self):
        """Contained evidence cannot seed an ordinary confirmation history."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            contained = _lbox(500, 560, half=6)
            active, _ = tracker.update(
                [contained], [vehicle], frame_index=0,
                quarantined_litters=[contained],
            )
            quarantined_id, quarantined = _get_only_litter(active)
            assert quarantined['vehicle_quarantine_active'] is True

            ordinary = _lbox(520, 610, half=6)
            active, _ = tracker.update(
                [ordinary], [vehicle], frame_index=1,
            )

            ordinary_ids = set(active) - {quarantined_id}
            assert len(ordinary_ids) == 1
            ordinary_data = active[ordinary_ids.pop()]
            assert ordinary_data['state'] == 'pending'
            assert ordinary_data['age'] == 1
            assert ordinary_data['history'] == [(520.0, 610.0)]
            assert ordinary_data.get('vehicle_quarantine_active', False) is False
        finally:
            tracker.close()

    def test_three_contained_observations_can_use_ordinary_exit_for_release(self):
        """The first box outside the carrier completes, rather than resets, evidence."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            active = {}
            for frame_index, (cx, cy) in enumerate(
                ((500, 480), (502, 484), (504, 488))
            ):
                litter = _lbox(cx, cy, half=6)
                active, _ = tracker.update(
                    [litter], [vehicle], frame_index=frame_index,
                    quarantined_litters=[litter],
                )
            litter_id, data = _get_only_litter(active)
            assert data['vehicle_quarantine_active'] is True

            # Same physical track exits below the carrier.  It supplies the
            # fourth relative point and passes the existing displacement gates.
            exited = _lbox(510, 640, half=6)
            active, _ = tracker.update(
                [exited], [vehicle], frame_index=3,
            )

            assert list(active) == [litter_id]
            assert active[litter_id]['vehicle_quarantine_active'] is False
            assert active[litter_id]['vehicle_quarantine_released'] is True
        finally:
            tracker.close()

    def test_contained_observation_cannot_quarantine_ordinary_track(self):
        """A containment-classification jitter starts a separate track."""
        from litterTracker import GlobalLitterTracker

        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=100, x2=700, y2=600,
        )
        try:
            ordinary = _lbox(500, 610, half=6)
            active, _ = tracker.update(
                [ordinary], [vehicle], frame_index=0,
            )
            ordinary_id, ordinary_data = _get_only_litter(active)
            assert ordinary_data.get('vehicle_quarantine_active', False) is False

            contained = _lbox(500, 590, half=6)
            active, _ = tracker.update(
                [contained], [vehicle], frame_index=1,
                quarantined_litters=[contained],
            )

            contained_ids = set(active) - {ordinary_id}
            assert len(contained_ids) == 1
            assert active[contained_ids.pop()]['vehicle_quarantine_active'] is True
            assert active[ordinary_id].get('vehicle_quarantine_active', False) is False
        finally:
            tracker.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: detect-stage litter candidate FP filter (Phase 2 relocated)
#
# Per the architecture decision, ALL litter FP filtering is centralized in
# detect.py preprocessing via smallFunction.litter_candidate_is_vehicle_fp; the
# tracker only tracks.  These tests pin that filter directly (pure function).
# ─────────────────────────────────────────────────────────────────────────────

class TestLitterCandidateFilter:

    def _veh(self, track_id=5, x1=400, y1=200, x2=600, y2=420):
        return {'cls': 'vehicle', 'track_id': track_id, 'box': [x1, y1, x2, y2]}

    def test_drops_contained_in_vehicle(self):
        """Candidate almost entirely inside a vehicle box → drop (vehicle part)."""
        from smallFunction import litter_candidate_is_vehicle_fp
        veh = self._veh()
        litter = _lbox(500, 320, half=6)          # well inside [400,200,600,420]
        drop, reason = litter_candidate_is_vehicle_fp(litter, [veh])
        assert drop and reason == 'vehicle_contained'

    def test_keeps_object_clear_of_vehicle(self):
        """Candidate outside any vehicle box, downward history → keep."""
        from smallFunction import litter_candidate_is_vehicle_fp
        veh = self._veh()
        litter = _lbox(900, 700, half=6)          # clear of the vehicle
        hist = [(892, 540), (896, 620)]           # falling, mild horizontal
        drop, reason = litter_candidate_is_vehicle_fp(litter, [veh], prev_litter_history=hist)
        assert not drop, f"clear downward object must be kept (got reason={reason})"

    def test_drops_horizontal_streak(self):
        """Near-pure-horizontal trajectory → drop (not gravity-driven)."""
        from smallFunction import litter_candidate_is_vehicle_fp
        litter = _lbox(800, 305, half=5)
        hist = [(500, 300), (650, 303)]           # huge horiz, tiny down
        drop, reason = litter_candidate_is_vehicle_fp(litter, [], prev_litter_history=hist)
        assert drop and reason == 'horizontal_streak'

    def test_can_defer_horizontal_streak_until_enough_observations(self, monkeypatch):
        """Research override defers rejection without declaring an event."""
        from smallFunction import litter_candidate_is_vehicle_fp
        monkeypatch.setenv("LITTER_FP_STREAK_MIN_OBSERVATIONS", "6")
        litter = _lbox(800, 305, half=10)
        hist = [(770, 300), (785, 303)]

        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [], prev_litter_history=hist
        )

        assert not drop
        assert reason is None

    def test_deferred_streak_rejects_implausible_identity_jump(self, monkeypatch):
        """Deferral cannot bridge a large normalized jump to another object."""
        from smallFunction import litter_candidate_is_vehicle_fp
        monkeypatch.setenv("LITTER_FP_STREAK_MIN_OBSERVATIONS", "6")
        monkeypatch.setenv(
            "LITTER_FP_STREAK_DEFER_MAX_STEP_DIAGONALS_PER_FRAME", "1.1"
        )
        litter = _lbox(800, 305, half=10)
        hist = [(500, 300)]

        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [], prev_litter_history=hist, prev_litter_missed=0
        )

        assert drop
        assert reason == 'horizontal_streak'

    def test_deferred_streak_normalizes_for_source_frame_gap(self, monkeypatch):
        """Sparse inference uses elapsed source frames, not processed calls."""
        from smallFunction import litter_candidate_is_vehicle_fp
        monkeypatch.setenv("LITTER_FP_STREAK_MIN_OBSERVATIONS", "6")
        monkeypatch.setenv(
            "LITTER_FP_STREAK_DEFER_MAX_STEP_DIAGONALS_PER_FRAME", "1.1"
        )
        litter = _lbox(800, 305, half=10)
        hist = [(740, 300)]

        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [], prev_litter_history=hist, prev_litter_missed=2
        )

        assert not drop
        assert reason is None

    def test_deferred_streak_still_drops_at_minimum_observations(self, monkeypatch):
        """The override delays the gate; it does not disable it."""
        from smallFunction import litter_candidate_is_vehicle_fp
        monkeypatch.setenv("LITTER_FP_STREAK_MIN_OBSERVATIONS", "6")
        litter = _lbox(800, 305, half=5)
        hist = [
            (500, 300),
            (560, 301),
            (620, 302),
            (680, 303),
            (740, 304),
        ]

        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [], prev_litter_history=hist
        )

        assert drop
        assert reason == 'horizontal_streak'

    def test_keeps_recent_horizontal_release_from_vehicle_edge(self):
        """A bottle that starts inside a vehicle box may initially fly almost horizontally."""
        from smallFunction import litter_candidate_is_vehicle_fp
        veh = {'cls': 'vehicle', 'track_id': 2, 'box': [1050, 0, 1678, 554]}
        litter = _lbox(1774, 60, half=5)          # 96px outside the right vehicle edge
        hist = [(1674, 61)]                        # prior observation was inside vehicle 2
        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [veh], prev_litter_history=hist)
        assert not drop, f"recent vehicle-edge release must reach tracker (got {reason})"

    def test_keeps_vertical_throw(self):
        """Downward-dominant trajectory clear of vehicles → keep."""
        from smallFunction import litter_candidate_is_vehicle_fp
        litter = _lbox(520, 360, half=5)
        hist = [(500, 300), (510, 330)]           # down 60, horiz 20
        drop, reason = litter_candidate_is_vehicle_fp(litter, [], prev_litter_history=hist)
        assert not drop, f"vertical throw must be kept (got reason={reason})"

    def test_keeps_descent_after_internal_apex_even_above_birth_height(self):
        """Rise-then-fall arc must not be mistaken for a horizontal streak."""
        from smallFunction import litter_candidate_is_vehicle_fp
        litter = _lbox(445, 245, half=5)
        hist = [(500, 300), (475, 230), (460, 235)]

        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [], prev_litter_history=hist
        )

        assert not drop, f"ballistic descent must survive streak gate ({reason=})"


    def test_drops_comoving_with_vehicle(self):
        """Candidate near a moving vehicle, moving at the vehicle's velocity → drop."""
        from smallFunction import litter_candidate_is_vehicle_fp
        # vehicle to the LEFT, its right edge near the litter; vehicle moving right+down.
        veh = {'cls': 'vehicle', 'track_id': 5, 'box': [300, 300, 495, 460]}
        vehicle_history = {5: {'centroids': [(390, 375), (398, 380)]}}  # step (8,5)
        litter = _lbox(520, 400, half=5)          # ~25px right of the vehicle edge
        hist = [(512, 395)]                        # litter step to (520,400) = (8,5) → matches vehicle
        drop, reason = litter_candidate_is_vehicle_fp(
            litter, [veh], vehicle_history=vehicle_history, prev_litter_history=hist)
        assert drop and reason == 'vehicle_comotion'

    def test_new_candidate_without_history_not_streak_or_comotion(self):
        """A brand-new candidate (no history) clear of vehicles must be kept (birth)."""
        from smallFunction import litter_candidate_is_vehicle_fp
        litter = _lbox(900, 700, half=5)
        drop, reason = litter_candidate_is_vehicle_fp(litter, [self._veh()], prev_litter_history=None)
        assert not drop, f"new clear candidate must birth (got reason={reason})"

    def test_vehicle_edge_release_fallback_beats_perspective_score(self):
        """Exact box-origin plus a clear release must retain the source vehicle."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
        vehicle = {'cls': 'vehicle', 'track_id': 2, 'box': [1050, 0, 1678, 554]}
        history = [(1674.5, 61.0), (1774.5, 60.5), (1945.5, 156.0)]
        thrower_key, _ = tracker._find_thrower_for_litter(
            _lbox(1945.5, 156.0, half=5), [vehicle], history=history)
        assert thrower_key == ('vehicle', 2)


class TestReleaseTimeActorCausality:

    def test_late_passerby_cannot_confirm_actorless_birth(self):
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        try:
            tracker.update([_lbox(500, 300)], [], frame_index=0)
            tracker.update([_lbox(510, 315)], [], frame_index=1)
            passerby = _vehicle_actor(
                track_id=7, x1=300, y1=250, x2=500, y2=420
            )
            active, _ = tracker.update(
                [_lbox(520, 335)], [passerby], frame_index=2
            )
            assert all(
                item['state'] != 'confirmed' for item in active.values()
            )
        finally:
            tracker.close()

    def test_arc_with_consistent_release_actor_confirms(self):
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        actor = _person_actor(
            track_id=2, x1=390, y1=220, x2=430, y2=340
        )
        try:
            points = [
                (440, 300), (430, 250), (420, 220),
                (410, 235), (400, 270), (390, 305),
            ]
            confirmed = False
            for frame_index, (cx, cy) in enumerate(points):
                active, _ = tracker.update(
                    [_lbox(cx, cy)], [actor], frame_index=frame_index
                )
                confirmed = confirmed or any(
                    item['state'] == 'confirmed' for item in active.values()
                )
            assert confirmed
        finally:
            tracker.close()

    def test_vehicle_to_person_handoff_keeps_release_causality(self):
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=1, x1=390, y1=220, x2=490, y2=340
        )
        person = _person_actor(
            track_id=2, x1=360, y1=180, x2=420, y2=360
        )
        try:
            tracker.update([_lbox(440, 300)], [vehicle], frame_index=0)
            tracker.update([_lbox(430, 250)], [vehicle], frame_index=1)
            points = [(420, 220), (410, 235), (400, 270), (390, 305)]
            confirmed = False
            for offset, (cx, cy) in enumerate(points, start=2):
                active, _ = tracker.update(
                    [_lbox(cx, cy)], [person], frame_index=offset
                )
                confirmed = confirmed or any(
                    item['state'] == 'confirmed' for item in active.values()
                )
            assert confirmed
        finally:
            tracker.close()

    def test_actorless_birth_allows_downward_dominant_two_point_fast_drop(self):
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=250, x2=500, y2=420
        )
        try:
            tracker.update([_lbox(520, 300)], [], frame_index=0)
            active, _ = tracker.update(
                [_lbox(550, 350)], [vehicle], frame_index=1
            )
            assert any(item['state'] == 'confirmed' for item in active.values())
        finally:
            tracker.close()

    def test_actorless_birth_allows_moderate_horizontal_drop_in_0827_profile(self):
        """The 8/27 profile permits ratio 85/60, below its extreme-streak gate."""
        from litterTracker import GlobalLitterTracker
        tracker = GlobalLitterTracker(fps=10)
        vehicle = _vehicle_actor(
            track_id=5, x1=300, y1=250, x2=500, y2=420
        )
        try:
            tracker.update([_lbox(520, 300)], [], frame_index=0)
            active, _ = tracker.update(
                [_lbox(605, 360)], [vehicle], frame_index=1
            )
            assert any(item['state'] == 'confirmed' for item in active.values())
        finally:
            tracker.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: camera-shake detection (slight monitor shake → global shift → litter FP)
#
# estimate_global_shift measures the dominant whole-frame translation.  Stable
# scenes (even with moving vehicles) report <1px; a camera jolt reports several
# px.  detect.py uses this to drop litter candidates during shake intervals.
# ─────────────────────────────────────────────────────────────────────────────

class TestCameraShakeDetection:

    def _texture(self):
        import numpy as np
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (360, 640, 3), dtype="uint8")

    def test_no_shift_reports_near_zero(self):
        from smallFunction import estimate_global_shift
        img = self._texture()
        mag, _ = estimate_global_shift(img, img.copy())
        assert mag < 1.0, f"identical frames must report ~0 global shift, got {mag}"

    def test_global_shift_detected(self):
        import numpy as np
        from smallFunction import estimate_global_shift
        img = self._texture()
        shifted = np.roll(img, 12, axis=1)      # whole frame shifts 12px → camera jolt
        mag, _ = estimate_global_shift(img, shifted)
        assert mag > 6.0, f"a 12px whole-frame shift must be detected, got {mag}"

    def test_none_inputs_safe(self):
        from smallFunction import estimate_global_shift
        assert estimate_global_shift(None, None) == (0.0, 0.0)
