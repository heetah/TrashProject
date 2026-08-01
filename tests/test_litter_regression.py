# -*- coding: utf-8 -*-
"""
Regression tests for litter tracker false-positive / false-negative cases.

Cases tested (all from /mnt/8tb_hdd/under115a/output/litter-all-test/):
  Case 10  – FP: old litter revealed by passing vehicle (0-1s) → must NOT confirm
  Case 100 – FN: real short arc throw (3-4s) → must confirm
  Case 102 – FP: object on top of vehicle moves with vehicle (16-17s) → holding, must NOT confirm
  Case 115 – FP: small noise on left side (6-8s) → must NOT confirm
  Case 116 – FP: 2-frame spurious detection (2-3s) → must NOT confirm

Run:
  conda run -n rtdetr pytest tests/test_litter_regression.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from litterTracker import GlobalLitterTracker
from smallFunction import litter_holding, validate_trajectory


# ── helpers ──────────────────────────────────────────────────────────────────

def mk_actor(cls, track_id, box, mask_poly=None):
    """Minimal actor dict compatible with litterTracker / litter_holding."""
    d = {'cls': cls, 'track_id': track_id, 'box': list(map(float, box))}
    if mask_poly is not None:
        d['mask_poly'] = mask_poly
    return d


def mk_litter(x1, y1, x2, y2, score=0.9):
    return (float(x1), float(y1), float(x2), float(y2), float(score))


def run_tracker_sequence(frames, tracker=None):
    """
    frames: list of (litters, actors, frame_index) tuples.
    Returns (final active_litters, final violators, tracker).
    """
    if tracker is None:
        tracker = GlobalLitterTracker()
    active, violators = {}, set()
    for litters, actors, fi in frames:
        active, violators = tracker.update(litters, actors, frame_index=fi)
    return active, violators, tracker


# ── Case 10 ──────────────────────────────────────────────────────────────────

class TestCase10OldLitterAfterOcclusion:
    """
    A vehicle passes in front of an old (stationary) litter.
    As the vehicle reveals the litter, the detection shifts ~15-18 px
    from the vehicle's shadow position to the true ground position.
    That short shift must NOT trigger a confirmation.
    """

    def _make_tracker(self):
        return GlobalLitterTracker()

    def test_two_frame_shift_at_vehicle_edge_not_confirmed(self):
        """
        Age-2 detection where litter appears to shift 18 px (vehicle shadow moving)
        must NOT confirm, even when a vehicle is nearby as a thrower candidate.
        """
        tracker = self._make_tracker()

        vehicle = mk_actor('vehicle', 1, (300, 440, 620, 570))

        # Frame 0: litter first seen at vehicle's right edge (shadow position)
        litter0 = mk_litter(610, 478, 632, 500, 0.88)
        tracker.update([litter0], [vehicle], frame_index=0)

        # Frame 1: vehicle has moved left ~15 px; litter detected at real ground position
        # centroid shift: ~12 px right, ~16 px down  (= 20 px displacement)
        vehicle2 = mk_actor('vehicle', 1, (285, 440, 605, 570))
        litter1 = mk_litter(622, 494, 644, 516, 0.88)
        active, violators, _ = run_tracker_sequence(
            [([litter1], [vehicle2], 1)], tracker=tracker
        )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert not confirmed, (
            f"Case 10 FP: 2-frame detection with 20px shift falsely confirmed. "
            f"ids={confirmed}, "
            f"history={[ld.get('history') for ld in active.values()]}"
        )
        tracker.close()

    def test_occluded_litter_reappears_at_same_spot_not_confirmed(self):
        """
        Litter tracked for 3 frames, disappears for 6 frames (occlusion),
        reappears 8 px shifted. Must NOT confirm.
        """
        tracker = self._make_tracker()
        vehicle = mk_actor('vehicle', 1, (250, 420, 600, 560))

        # Build small history (stationary with minor jitter)
        for fi in range(3):
            lit = mk_litter(500 + fi, 480, 522 + fi, 502, 0.87)
            tracker.update([lit], [vehicle], frame_index=fi)

        # Occlusion: 6 missed frames
        for fi in range(3, 9):
            tracker.update([], [vehicle], frame_index=fi)

        # Reappear at slightly shifted position (old litter, no real movement)
        litter_back = mk_litter(508, 488, 530, 510, 0.87)
        active, violators, _ = run_tracker_sequence(
            [([litter_back], [vehicle], 9)], tracker=tracker
        )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert not confirmed, (
            f"Case 10 FP: occluded litter reappear falsely confirmed. "
            f"ids={confirmed}"
        )
        tracker.close()


# ── Case 100 ─────────────────────────────────────────────────────────────────

class TestCase100RealThrowNotConfirmed:
    """
    A real littering event where the litter is thrown with a moderate arc.
    The upward phase of the arc must not block confirmation.
    """

    def test_arc_throw_trajectory_valid(self):
        """validate_trajectory should accept an arc throw that ends below start."""
        # Arc: 3 frames up (-8px/frame) then 4 frames down (+14px/frame)
        history = [
            (400.0, 380.0),
            (415.0, 372.0),
            (430.0, 364.0),
            (445.0, 356.0),  # peak
            (460.0, 370.0),
            (475.0, 390.0),
            (488.0, 414.0),
        ]
        is_valid, straightness = validate_trajectory(history)
        assert is_valid, (
            f"Case 100 FN: arc throw rejected by validate_trajectory. "
            f"straightness={straightness:.3f}, "
            f"start_y={history[0][1]}, end_y={history[-1][1]}"
        )

    def test_short_arc_throw_confirmed_by_tracker(self):
        """
        Short throw with arc: person at (350,300-480), litter goes up-then-down
        ending ~60px horizontal from start. Must confirm by age 6.
        """
        tracker = GlobalLitterTracker()
        person = mk_actor('person', 1, (340, 300, 415, 480))

        # Arc trajectory (30fps; 6 frames ≈ 0.2s)
        arc_frames = [
            (390, 370, 412, 392),   # f0  thrown from mid-air
            (407, 362, 429, 384),   # f1  going up
            (424, 355, 446, 377),   # f2  peak
            (441, 362, 463, 384),   # f3  descending
            (456, 376, 478, 398),   # f4
            (470, 396, 492, 418),   # f5  landed
        ]
        active = {}
        for fi, (x1, y1, x2, y2) in enumerate(arc_frames):
            active, violators = tracker.update(
                [mk_litter(x1, y1, x2, y2, 0.9)],
                [person],
                frame_index=fi,
            )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert confirmed, (
            f"Case 100 FN: real arc throw not confirmed by age {len(arc_frames)}. "
            f"states={[(lid, ld.get('state'), ld.get('age')) for lid, ld in active.items()]}"
        )
        tracker.close()


# ── Case 102 ─────────────────────────────────────────────────────────────────

class TestCase102ObjectOnVehicleTop:
    """
    An object sitting on the roof/top of a vehicle moves with the vehicle.
    litter_holding must detect this as 'held' (no release).
    """

    def test_litter_inside_vehicle_bbox_is_holding_no_mask(self):
        """
        Litter bbox fully inside vehicle bbox → holding must return True
        even when the anchor point is far from the vehicle bottom center.
        """
        vehicle = mk_actor('vehicle', 1, (200, 280, 520, 510))
        # Object on roof: inside vehicle bbox, near top edge
        litter_box = mk_litter(310, 285, 365, 315, 0.9)

        # Object moves WITH vehicle (relative motion ≈ 0)
        prev_center = (337.5, 300.0)
        litter_center_now = ((310 + 365) / 2, (285 + 315) / 2)  # (337.5, 300)
        # Simulate vehicle moving right 10 px → same relative motion
        veh_history = {1: {'centroids': [(355.0, 390.0), (365.0, 390.0)]}}

        is_holding, actor_key = litter_holding(
            litter_box,
            [vehicle],
            prev_litter_center=prev_center,
            vehicle_history=veh_history,
        )
        assert is_holding, (
            "Case 102 FP: object inside vehicle bbox not caught as holding. "
            f"actor_key={actor_key}"
        )

    def test_object_on_vehicle_top_not_confirmed_by_tracker(self):
        """
        Object on vehicle roof that moves with vehicle for 20 frames must NOT confirm.
        (Tests tracker path assuming litter passes holding filter somehow.)
        """
        tracker = GlobalLitterTracker()
        confirmed_at = None

        for fi in range(20):
            x_offset = fi * 12  # vehicle moves right
            vehicle = mk_actor('vehicle', 1, (200 + x_offset, 290, 510 + x_offset, 510))
            # Object on roof, same x_offset → moves WITH vehicle
            ox = 315 + x_offset
            litter = mk_litter(ox, 288, ox + 40, 312, 0.9)

            active, _ = tracker.update([litter], [vehicle], frame_index=fi)
            if any(ld['state'] == 'confirmed' for ld in active.values()):
                confirmed_at = fi
                break

        assert confirmed_at is None, (
            f"Case 102 FP: object on vehicle roof confirmed at frame {confirmed_at}"
        )
        tracker.close()


# ── Case 115 / 116 ───────────────────────────────────────────────────────────

class TestCase115116NoiseFalsePositive:
    """
    Brief noise detections (small bbox, short-lived) must not confirm,
    even when an actor (vehicle) is nearby.
    """

    def test_tiny_2frame_noise_not_confirmed(self):
        """
        Tiny bbox (10x8 px) seen for 2 frames with 16px shift and vehicle nearby.
        Must NOT confirm (Case 115 / 116 pattern).
        """
        tracker = GlobalLitterTracker()
        vehicle = mk_actor('vehicle', 1, (100, 380, 400, 510))

        # Frame 0: noise appears
        noise0 = mk_litter(155, 400, 165, 408, 0.7)
        tracker.update([noise0], [vehicle], frame_index=0)

        # Frame 1: noise shifted 10px right, 12px down
        noise1 = mk_litter(165, 412, 175, 420, 0.7)
        active, violators, _ = run_tracker_sequence(
            [([noise1], [vehicle], 1)], tracker=tracker
        )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert not confirmed, (
            f"Case 115/116 FP: 2-frame tiny noise falsely confirmed. ids={confirmed}"
        )
        tracker.close()

    def test_3frame_noise_small_bbox_not_confirmed(self):
        """
        Small bbox (12x10 px) seen for 3 frames, straight movement 18px total.
        Must NOT confirm without significant absolute displacement.
        """
        tracker = GlobalLitterTracker()
        vehicle = mk_actor('vehicle', 1, (80, 370, 380, 500))

        for fi in range(3):
            ox = 200 + fi * 9
            oy = 415 + fi * 5
            noise = mk_litter(ox, oy, ox + 12, oy + 10, 0.65)
            active, violators = tracker.update([noise], [vehicle], frame_index=fi)

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert not confirmed, (
            f"Case 115/116 FP: 3-frame small noise falsely confirmed. ids={confirmed}"
        )
        tracker.close()

    def test_noise_without_thrower_not_confirmed(self):
        """
        Noise detection with no actors nearby must never confirm.
        """
        tracker = GlobalLitterTracker()
        for fi in range(5):
            ox = 300 + fi * 8
            oy = 400 + fi * 6
            noise = mk_litter(ox, oy, ox + 10, oy + 8, 0.6)
            active, _ = tracker.update([noise], [], frame_index=fi)

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert not confirmed, f"Noise without thrower should never confirm. ids={confirmed}"
        tracker.close()


# ── Positive regression (resize.mp4-style real throw) ───────────────────────

class TestPositiveRegressionRealThrow:
    """
    Positive regression: a real street-side littering event must still confirm
    after all FP fixes. Simulates the kind of trajectory produced by resize.mp4:
    person standing beside a scooter drops/tosses an object that falls
    diagonally down from waist height.
    """

    def test_side_drop_from_scooter_rider_confirms(self):
        """
        Rider on scooter drops litter at mid-frame, object falls ~80px down
        and ~60px sideways over 5 frames. Person and scooter both in scene.
        Must confirm by frame 4 (age=5).
        """
        tracker = GlobalLitterTracker()
        # Person standing beside scooter
        person = mk_actor('person', 2, (380, 240, 440, 460))
        scooter = mk_actor('scooter', 5, (340, 350, 500, 490))

        # Litter falls from about waist-height of person, diagonal down-right
        drop_frames = [
            (418, 340, 442, 364),   # f0  released from hand area
            (430, 358, 454, 382),   # f1  falling
            (443, 381, 467, 405),   # f2
            (454, 407, 478, 431),   # f3
            (462, 418, 486, 442),   # f4  near ground
        ]
        active, confirmed_at = {}, None
        for fi, (x1, y1, x2, y2) in enumerate(drop_frames):
            active, _ = tracker.update(
                [mk_litter(x1, y1, x2, y2, 0.88)],
                [person, scooter],
                frame_index=fi,
            )
            if any(ld['state'] == 'confirmed' for ld in active.values()):
                confirmed_at = fi
                break

        assert confirmed_at is not None, (
            f"Positive regression FAIL: resize.mp4-style drop not confirmed. "
            f"Final states={[(lid, ld.get('state'), ld.get('age')) for lid, ld in active.items()]}"
        )
        tracker.close()

    def test_fast_throw_large_first_step_confirms(self):
        """
        resize.mp4 regression: first tracker step is ~147px (fast throw from vehicle).
        max_vehicle_thrower_step_px must not block this (threshold must be > 147px).
        """
        tracker = GlobalLitterTracker()
        vehicle = mk_actor('vehicle', 1, (300, 350, 600, 510))
        person = mk_actor('person', 2, (390, 250, 450, 460))

        # Litter thrown fast: first step 147px downward from vehicle/person area
        throw_frames = [
            # f0: birth (near person's hand, inside vehicle bbox area)
            (420, 290, 445, 315),
            # f1: +25 right, +147 down (fast throw, large first step)
            (445, 437, 470, 462),
            # f2: continues downward
            (452, 470, 477, 495),
        ]
        active = {}
        for fi, (x1, y1, x2, y2) in enumerate(throw_frames):
            active, _ = tracker.update(
                [mk_litter(x1, y1, x2, y2, 0.88)],
                [vehicle, person],
                frame_index=fi + 75,  # matches resize.mp4 timing (fi=75-77)
            )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert confirmed, (
            f"resize.mp4 regression: fast throw with 147px first step not confirmed. "
            f"Check max_vehicle_thrower_step_px threshold. "
            f"states={[(lid, ld.get('state'), ld.get('age')) for lid, ld in active.items()]}"
        )
        tracker.close()

    def test_fast_horizontal_throw_from_person_confirms(self):
        """
        Person throws litter fast (large horizontal + downward displacement).
        Must confirm via motion path (can_confirm_by_motion) by age 3.
        """
        tracker = GlobalLitterTracker()
        person = mk_actor('person', 1, (300, 200, 360, 420))

        throw_frames = [
            (345, 290, 375, 318),   # f0  release
            (390, 314, 420, 342),   # f1  +45 x, +24 y
            (432, 346, 462, 374),   # f2  +87 x, +56 y from start
        ]
        active = {}
        for fi, (x1, y1, x2, y2) in enumerate(throw_frames):
            active, _ = tracker.update(
                [mk_litter(x1, y1, x2, y2, 0.91)],
                [person],
                frame_index=fi,
            )

        confirmed = [lid for lid, ld in active.items() if ld['state'] == 'confirmed']
        assert confirmed, (
            f"Positive regression FAIL: fast throw not confirmed. "
            f"states={[(lid, ld.get('state'), ld.get('age')) for lid, ld in active.items()]}"
        )
        tracker.close()
