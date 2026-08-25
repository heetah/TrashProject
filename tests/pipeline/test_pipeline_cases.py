"""
TDD integration tests for 9 specific litter detection video cases.

Expected outcomes:
  FP cases (should NOT confirm litter): 10, 102, 115, 116, 137, 138
  FN cases (should confirm litter):     100, 126, 140

Test strategy:
  1. Run the production scripts/main.py with environment-based configuration
  2. Parse the "confirmed_ids=N" from the summary line
  3. Assert FP cases have 0 confirms
  4. Assert FN cases have >= 1 confirm

Run with:
    cd /home/se_copilot/trashProject
    conda run -n rtdetr python -m pytest \
        tests/pipeline/test_pipeline_cases.py -v --timeout=300
"""
import os
import re
import subprocess
import sys
import pytest

# Absolute paths
PROJECT_ROOT = "/home/se_copilot/trashProject"
SCRIPT_PATH  = os.path.join(PROJECT_ROOT, "scripts", "main.py")
PYTHON       = sys.executable
VIDEO_BASE   = "/mnt/8tb_hdd/under115a/litter_vidshort/litter"
OUTPUT_BASE  = "/tmp/litter_case_tests"

os.makedirs(OUTPUT_BASE, exist_ok=True)

# The video-driven tests below run the full pipeline (model load + video decode)
# per case and are slow.  They are opt-in so the fast unit-test loop stays fast;
# enable the heavy classes with:
#     RUN_PIPELINE_TESTS=1 conda run -n rtdetr python -m pytest .../test_pipeline_cases.py
# (TestTrackerThresholds below is a fast unit check and always runs.)
_heavy = pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_TESTS") != "1",
    reason="Heavy pipeline integration tests; set RUN_PIPELINE_TESTS=1 to run.",
)

# Case definitions: (label, fps, description)
CASES = {
    10:  ("FP", 30,  "0-1s: old litter behind passing car"),
    100: ("FN", 12,  "3-4s: litter not confirmed"),
    102: ("FP", 10,  "16-17s: vehicle object"),
    115: ("FP", 10,  "6-8s: left noise"),
    116: ("FP", 10,  "2-3s: false confirm"),
    126: ("FN", 10,  "whole: garbage bag not confirmed"),
    137: ("FP", 10,  "2-3s: under-vehicle object"),
    138: ("FP", 10,  "15-16s: car sweep"),
    140: ("FN", 10,  "6-7s: litter not confirmed"),
}

CONF_THRESHOLD = "0.15"   # Low enough to detect real litter throws


def _run_case(case_id: int):
    """Run main.py on a case video and return (confirmed_ids, raw_candidates)."""
    video = os.path.join(VIDEO_BASE, f"litter_case_{case_id}.mp4")
    cmd = [PYTHON, SCRIPT_PATH, video]
    env = os.environ.copy()
    env.update({
        "PIPELINE_BATCH": "8",
        "TRASH_CONF": CONF_THRESHOLD,
        "OUTPUT_ROOT": OUTPUT_BASE,
        "SMART_BACKTRACK_SIDECAR": "0",
    })
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=PROJECT_ROOT,
        env=env,
    )
    output_text = result.stdout + result.stderr
    # Parse summary line
    m_conf = re.search(r"confirmed_ids=(\d+)", output_text)
    m_raw  = re.search(r"raw_candidates=(\d+)", output_text)
    confirmed = int(m_conf.group(1)) if m_conf else -1
    raw       = int(m_raw.group(1)) if m_raw else -1
    return confirmed, raw


# ─────────────────────────────────────────────────────────────────────────────
# FP cases: must NOT confirm litter
# ─────────────────────────────────────────────────────────────────────────────

@_heavy
class TestFalsePositiveSuppression:
    """Vehicle artifacts, old litter re-detection, and noise must not confirm."""

    def test_case_10_old_litter_behind_car(self):
        """Case 10 (0-1s): static old litter behind passing car — must NOT confirm."""
        confirmed, raw = _run_case(10)
        assert confirmed == 0, (
            f"Case 10 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Old litter revealed when car passes should NOT be confirmed as a new throw."
        )

    def test_case_102_vehicle_object(self):
        """Case 102 (16-17s): object on vehicle — must NOT confirm."""
        confirmed, raw = _run_case(102)
        assert confirmed == 0, (
            f"Case 102 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Object on/near vehicle bbox must not be confirmed as thrown litter."
        )

    def test_case_115_left_noise(self):
        """Case 115 (6-8s): noise on left side — must NOT confirm."""
        confirmed, raw = _run_case(115)
        assert confirmed == 0, (
            f"Case 115 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Detector noise detections near vehicles must not confirm as litter."
        )

    def test_case_116_false_confirm(self):
        """Case 116 (2-3s): false confirm — must NOT confirm."""
        confirmed, raw = _run_case(116)
        assert confirmed == 0, (
            f"Case 116 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Vehicle-associated object with horizontal-only motion must not confirm."
        )

    def test_case_137_under_vehicle_object(self):
        """Case 137 (2-3s): object under vehicle — must NOT confirm."""
        confirmed, raw = _run_case(137)
        assert confirmed == 0, (
            f"Case 137 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Object appearing/disappearing under vehicle must not confirm as thrown litter."
        )

    def test_case_138_car_sweep(self):
        """Case 138 (15-16s): car sweeping past — must NOT confirm."""
        confirmed, raw = _run_case(138)
        assert confirmed == 0, (
            f"Case 138 FP: got confirmed_ids={confirmed} (expected 0). "
            f"raw_candidates={raw}. "
            "Motion artifact from car sweeping past must not confirm as litter."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FN cases: MUST confirm at least one litter event
# ─────────────────────────────────────────────────────────────────────────────

@_heavy
class TestFalseNegativeRecovery:
    """Real litter throw events must be confirmed."""

    def test_case_100_litter_at_3_4s(self):
        """Case 100: litter thrown at ~3-4s — must confirm."""
        confirmed, raw = _run_case(100)
        assert confirmed >= 1, (
            f"Case 100 FN: got confirmed_ids={confirmed} (expected >= 1). "
            f"raw_candidates={raw}. "
            "Real litter throw event must be confirmed by the tracker."
        )

    def test_case_126_garbage_bag(self):
        """Case 126: garbage bag visible throughout — must confirm."""
        confirmed, raw = _run_case(126)
        assert confirmed >= 1, (
            f"Case 126 FN: got confirmed_ids={confirmed} (expected >= 1). "
            f"raw_candidates={raw}. "
            "Garbage bag litter event must be detected and confirmed."
        )

    def test_case_140_litter_at_6_7s(self):
        """Case 140: litter thrown at ~6-7s — must confirm."""
        confirmed, raw = _run_case(140)
        assert confirmed >= 1, (
            f"Case 140 FN: got confirmed_ids={confirmed} (expected >= 1). "
            f"raw_candidates={raw}. "
            "Real litter throw event must be confirmed by the tracker."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration guard: tracker threshold values
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackerThresholds:
    """Guard the tracker parameters that were changed for 4c pipeline."""

    def test_vehicle_thrower_min_confirm_age(self):
        """Vehicle-thrower confirmations require age >= 3 (not default 2)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            assert hasattr(t, 'min_confirm_age_vehicle'), (
                "GlobalLitterTracker must have 'min_confirm_age_vehicle' attribute. "
                "Add it to suppress vehicle-artifact FP confirmations."
            )
            assert t.min_confirm_age_vehicle >= 3, (
                f"min_confirm_age_vehicle={t.min_confirm_age_vehicle}, must be >= 3. "
                "Vehicle-thrower litters confirmed at age=2 are mostly vehicle artifacts."
            )
        finally:
            t.close()

    def test_vehicle_thrower_downward_displacement(self):
        """Vehicle-thrower confirmations require stronger downward displacement."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            assert hasattr(t, 'min_confirm_downward_displacement_vehicle'), (
                "GlobalLitterTracker must have 'min_confirm_downward_displacement_vehicle' attribute."
            )
            assert t.min_confirm_downward_displacement_vehicle >= 12.0, (
                f"min_confirm_downward_displacement_vehicle={t.min_confirm_downward_displacement_vehicle}, "
                "must be >= 12.0. Small downward displacement with vehicle thrower is a FP pattern."
            )
        finally:
            t.close()

    def test_stationary_locked_does_not_absorb_new_detections(self):
        """Stationary-locked pending litters must not absorb new detections."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
        from litterTracker import GlobalLitterTracker
        t = GlobalLitterTracker()
        try:
            # Create a stationary locked litter by holding it static for lock_age frames
            actors = [{'cls': 'vehicle', 'track_id': 1, 'box': [300, 200, 500, 400]}]

            # Hold litter static for lock_age+1 frames → becomes stationary_locked
            lock_age = t.stationary_lock_age
            litter_box = (450, 350, 470, 370, 0.8)  # Near vehicle
            l_id = None
            for fi in range(lock_age + 2):
                active, _ = t.update([litter_box], actors, frame_index=fi)
                if l_id is None and active:
                    l_id = next(iter(active))

            # Verify litter is stationary_locked
            assert l_id in t.active_litters, "Litter should still be tracked"
            assert t.active_litters[l_id].get('stationary_locked', False), (
                "Litter should be stationary_locked after static period"
            )

            # Now send a new detection NEAR the locked litter (within distance_threshold)
            # It should NOT match the locked litter → creates a new track
            new_box = (460, 355, 480, 375, 0.8)  # 15px away from locked litter center
            fi_new = lock_age + 3
            active, _ = t.update([new_box], actors, frame_index=fi_new)

            # There should be a NEW litter id (not the locked one)
            new_ids = [lid for lid in active if lid != l_id]
            assert len(new_ids) >= 1, (
                "New detection near stationary_locked litter must create a new track, "
                "not be absorbed into the locked one. "
                "This enables real throws near old static litter to be tracked freshly."
            )
        finally:
            t.close()
