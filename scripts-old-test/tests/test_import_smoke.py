# -*- coding: utf-8 -*-
"""套件翻新的 import 冒煙測試。

驗證舊模組路徑(相容 shim)與新的 pipeline.* 路徑都能解析,且兩邊指向同一個物件,
確保 re-export 搬移沒有破壞既有 import。不需要 GPU/模型載入即可執行,作為每次搬移的
回歸安全網:

    conda run -n rtdetr python -m pytest scripts-old-test/tests/test_import_smoke.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_profiling_shim_identity():
    import timeUtils
    from pipeline import profiling

    assert timeUtils.PipelineProfiler is profiling.PipelineProfiler
    assert timeUtils.profile_block is profiling.profile_block
    assert timeUtils.time_it is profiling.time_it


def test_geometry_shim_identity():
    import smallFunction
    from pipeline import geometry

    for name in (
        "litter_holding",
        "motion_evidence",
        "calculate_iom_matrix",
        "calculate_iou_matrix",
        "estimate_global_shift",
        "litter_candidate_is_vehicle_fp",
        "validate_trajectory",
        "calculate_mask_overlap_ratio",
        "SHAKE_COOLDOWN_SEC",
        "SHAKE_SHIFT_FLOOR_PX",
        "SHAKE_SHIFT_FRAC",
    ):
        assert getattr(smallFunction, name) is getattr(geometry, name), name


def test_paths_repo_root_resolves():
    from pipeline import paths

    # pipeline.paths 取代 action.py/litterTracker.py 的 __file__ 相對推算,
    # 必須指到真正的 repo root(含 mmaction2),搬移後路徑深度才不會漂掉。
    assert os.path.isdir(paths.REPO_ROOT), paths.REPO_ROOT
    assert os.path.basename(paths.SCRIPTS_DIR) == "scripts-old-test"
    assert paths.MMACTION_REPO == os.path.join(paths.REPO_ROOT, "mmaction2")
    assert os.path.isdir(paths.MMACTION_REPO), paths.MMACTION_REPO


def test_plate_and_action_shims_identity():
    import licensePlate
    from pipeline import plate

    assert licensePlate.get_plate_number is plate.get_plate_number
    assert licensePlate.detect_license_plates is plate.detect_license_plates

    import action
    from pipeline import action as pipeline_action

    assert action.STGCNActionModule is pipeline_action.STGCNActionModule
    assert action.ACTION_CLASSES == {0: "normal", 1: "urinate"}


def test_detect_shim_and_device_source():
    import detect
    from pipeline import detect as pipeline_detect
    from pipeline import devices

    assert detect.detect_batch is pipeline_detect.detect_batch
    assert detect.compute_pixel_change_map is pipeline_detect.compute_pixel_change_map
    # detect 的 device 常數現在來自 pipeline.devices(不再自己定義)。
    assert detect.BBOX_DEVICE is devices.BBOX_DEVICE
    assert detect.TRASH_HALF is devices.TRASH_HALF


def test_infra_no_longer_imports_detector():
    # infra 不得再相依偵測器;device 常數改由中性的 pipeline.devices 取得。
    from pipeline import infra
    from pipeline import devices

    assert infra.BBOX_DEVICE is devices.BBOX_DEVICE
    assert infra.TRASH_DEVICE is devices.TRASH_DEVICE
    src = open(infra.__file__, encoding="utf-8").read()
    assert "from detect import" not in src

    # examine 舊路徑仍是可用的相容 shim,指向同一組物件。
    import examine
    assert examine.AsyncFFmpegVideoWriter is infra.AsyncFFmpegVideoWriter


def test_main_uses_canonical_pipeline_imports():
    # 生產進入點應直接 import pipeline.*,不再依賴舊扁平 shim。
    main_src = open(
        os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8"
    ).read()
    for legacy in (
        "from detect import",
        "from litterTracker import",
        "from action import",
        "from licensePlate import",
        "from timeUtils import",
        "from examine import",
    ):
        assert legacy not in main_src, legacy


def test_litter_tracker_shim_identity():
    import litterTracker
    from pipeline import litter_tracker

    assert litterTracker.GlobalLitterTracker is litter_tracker.GlobalLitterTracker
    assert litterTracker.REF_FPS == litter_tracker.REF_FPS


def test_consumers_resolve_through_shims():
    # 這些模組透過 shim 鏈間接 import pipeline 子模組;import 成功即代表 shim 正常。
    # (import 階段不觸發模型載入,故不需 GPU。)
    import detect
    import litterTracker
    import licensePlate

    assert hasattr(detect, "detect_batch")
    assert hasattr(litterTracker, "GlobalLitterTracker")
    assert hasattr(licensePlate, "get_plate_number")
