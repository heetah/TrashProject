# -*- coding: utf-8 -*-
"""事件正規化與每影片精簡 analysis.json 輸出。

事件建構為純函式，不依賴 GPU/模型，可獨立單元測試。

``analysis.json`` 是前端唯一資料檔，保留影片長度、RT-DETR/後處理逐幀診斷、
confirmed 事件數、通行車輛估計、confidence、車牌與精簡事件。沒有人工 reviewed
ground truth 時，accuracy 固定為 ``null``，不可把 confidence 當成 accuracy。

事件 schema(每行一個 JSON 物件):
  litter:  {type:"litter", frame_index, time_sec, litter_id, bbox:[x1,y1,x2,y2],
            thrower:{cls,track_id}|null, license_plate:str|null, escalated:bool}
  urinate: {type:"urinate", frame_index, time_sec, track_id,
            vehicle:{cls,track_id}|null, license_plate:str|null}
"""
import json
import math
import os
from pathlib import Path

VEHICLE_LIKE = ("vehicle", "scooter")
ANALYSIS_SCHEMA_VERSION = "2.1.0"


def _time_sec(frame_index, fps):
    fps = float(fps) if fps else 30.0
    return round(int(frame_index) / fps, 2)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value, digits=4):
    number = _finite_float(value)
    return round(number, digits) if number is not None else None


def _plate_for_thrower(thrower_key, vehicle_history):
    # thrower 綁定 vehicle/scooter 時，回傳 (車牌字串, OCR confidence, 狀態)。
    if not thrower_key:
        return None, None, "not_applicable"
    cls, track_id = thrower_key[0], thrower_key[1]
    if cls not in VEHICLE_LIKE:
        return None, None, "not_applicable"
    info = (vehicle_history or {}).get(track_id)
    if not info:
        return None, None, "not_requested"
    plate = info.get("license_plate")
    if isinstance(plate, dict):
        number = plate.get("number")
        confidence = _rounded(plate.get("conf"))
    else:
        number = plate
        confidence = None
    number = str(number) if number not in (None, "") else None
    if number:
        status = "recognized"
    elif info.get("plate_search_until_found"):
        status = "pending"
    elif int(info.get("plate_ocr_misses", 0) or 0) > 0:
        status = "attempted_no_result"
    else:
        status = "not_requested"
    return number, confidence, status


def _litter_time_segment(event, fps):
    """回傳可 seek 的垃圾事件片段；來源層級一併輸出，避免冒充人工標註。"""
    confirm_frame = int(event.get("confirm_frame", event.get("frame_index", 0)))
    backtrack = event.get("backtrack") if isinstance(event.get("backtrack"), dict) else {}
    if backtrack.get("confirm_frame") is not None:
        confirm_frame = int(backtrack["confirm_frame"])

    release_frame = backtrack.get("release_frame")
    birth_frame = event.get("birth_frame", backtrack.get("birth_frame"))
    if release_frame is not None:
        start_frame = int(release_frame)
        basis = "estimated_release_to_confirmation"
    elif birth_frame is not None:
        start_frame = int(birth_frame)
        basis = "candidate_birth_to_confirmation"
    else:
        start_frame = int(event.get("frame_index", 0))
        basis = "confirmation_frame_only"

    end_frame = max(start_frame, confirm_frame)
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": _time_sec(start_frame, fps),
        "end_sec": _time_sec(end_frame, fps),
        "basis": basis,
        "human_reviewed": False,
    }


def _urinate_time_segment(event, fps):
    """以確認時間與已累積 evidence 建立可剪輯區間。"""
    frame_index = event.get("frame_index")
    if frame_index is None:
        return {
            "start_sec": None,
            "end_sec": None,
            "basis": "aggregate_without_timestamp",
            "human_reviewed": False,
        }
    end_sec = _time_sec(frame_index, fps)
    evidence_sec = max(_finite_float(event.get("evidence_sec")) or 0.0, 0.0)
    return {
        "start_sec": round(max(0.0, end_sec - evidence_sec), 2),
        "end_sec": end_sec,
        "basis": "stgcn_evidence_to_confirmation",
        "human_reviewed": False,
    }


def _vehicle_for_person(track_id, person_vehicle_map, event_frame_index=None):
    if track_id is None or not person_vehicle_map:
        return None
    try:
        person_id = int(track_id)
    except (TypeError, ValueError):
        return None
    vehicle_key = None
    if event_frame_index is not None:
        try:
            vehicle_key = person_vehicle_map.get((person_id, int(event_frame_index)))
        except (TypeError, ValueError):
            vehicle_key = None
    if vehicle_key is None:
        vehicle_key = person_vehicle_map.get(person_id)
    if vehicle_key is None:
        vehicle_key = person_vehicle_map.get(str(person_id))
    if not isinstance(vehicle_key, (tuple, list)) or len(vehicle_key) != 2:
        return None
    cls_name = str(vehicle_key[0]).lower()
    if cls_name not in VEHICLE_LIKE:
        return None
    try:
        vehicle_id = int(vehicle_key[1])
    except (TypeError, ValueError):
        return None
    return (cls_name, vehicle_id)


def build_litter_events(litter_events, vehicle_history, fps):
    """把 tracker 的 confirmed litter 事件(get_litter_events())轉成扁平 event 記錄。"""
    out = []
    for ev in litter_events or []:
        thrower_key = ev.get("thrower_key")
        vehicle_key = ev.get("vehicle_key")
        thrower = None
        if thrower_key:
            thrower = {"cls": str(thrower_key[0]), "track_id": int(thrower_key[1])}
        vehicle = None
        if vehicle_key:
            vehicle = {"cls": str(vehicle_key[0]), "track_id": int(vehicle_key[1])}
        plate_number, plate_confidence, plate_status = _plate_for_thrower(
            vehicle_key or thrower_key, vehicle_history
        )
        item = {
            "type": "litter",
            "frame_index": int(ev.get("frame_index", 0)),
            "time_sec": _time_sec(ev.get("frame_index", 0), fps),
            "time_segment": _litter_time_segment(ev, fps),
            "litter_id": int(ev.get("litter_id", -1)),
            "bbox": [int(v) for v in ev.get("bbox", [])],
            "thrower": thrower,
            "vehicle": vehicle,
            "license_plate": plate_number,
            "license_plate_confidence": plate_confidence,
            "license_plate_status": plate_status,
            "detector_confidence": _rounded(ev.get("detector_confidence")),
            "escalated": bool(ev.get("escalated", False)),
            "backtrack_status": ev.get("backtrack_status"),
            # Explicit runtime gate provenance; never interpreted as reviewed
            # ground truth or accuracy.
            "confirmation_evidence": ev.get("confirmation_evidence"),
            "vehicle_quarantine_evidence": ev.get(
                "vehicle_quarantine_evidence"
            ),
        }
        if ev.get("backtrack") is not None:
            item["backtrack"] = ev["backtrack"]
        out.append(item)
    return out


def build_urinate_events(
    urinate_events,
    run_summary,
    fps,
    person_vehicle_map=None,
    vehicle_history=None,
):
    """優先用 per-track 確認明細(action_module.get_urinate_events());沒有時退回 summary
    聚合(相容舊行為:一筆 run-level urinate 事件)。"""
    if urinate_events:
        out = []
        for ev in urinate_events:
            fi = ev.get("frame_index")
            vehicle_key = _vehicle_for_person(
                ev.get("track_id"), person_vehicle_map, ev.get("frame_index")
            )
            vehicle = (
                {"cls": vehicle_key[0], "track_id": vehicle_key[1]}
                if vehicle_key is not None
                else None
            )
            plate_number, plate_confidence, plate_status = _plate_for_thrower(
                vehicle_key, vehicle_history
            )
            out.append({
                "type": "urinate",
                "track_id": int(ev.get("track_id", -1)),
                "frame_index": int(fi) if fi is not None else None,
                "time_sec": _time_sec(fi, fps) if fi is not None else None,
                "time_segment": _urinate_time_segment(ev, fps),
                "conf": round(float(ev.get("conf", 0.0)), 3),
                "evidence_sec": round(float(ev.get("evidence_sec", 0.0)), 2),
                "vehicle": vehicle,
                "license_plate": plate_number,
                "license_plate_confidence": plate_confidence,
                "license_plate_status": plate_status,
                "attribution_status": "resolved" if vehicle else "dustbin",
            })
        out.sort(key=lambda e: e["frame_index"] if e["frame_index"] is not None else float("inf"))
        return out

    confirmed = int((run_summary or {}).get("stgcn_urinate_confirmed", 0))
    if confirmed <= 0:
        return []
    return [{
        "type": "urinate",
        "track_id": None,
        "frame_index": None,
        "time_sec": None,
        "confirmed_count": confirmed,
    }]


def build_run_events(
    litter_events,
    urinate_events,
    vehicle_history,
    run_summary,
    fps,
    action_vehicle_associations=None,
):
    """組合一次 run 的所有事件:litter 依 frame 排序,urinate(per-track 或聚合)接在後面。"""
    events = build_litter_events(litter_events, vehicle_history, fps)
    events.sort(key=lambda e: e.get("frame_index") or 0)
    events.extend(build_urinate_events(
        urinate_events,
        run_summary,
        fps,
        person_vehicle_map=action_vehicle_associations,
        vehicle_history=vehicle_history,
    ))
    return events


def write_events_jsonl(events, path):
    """一行一事件寫出 JSONL(UTF-8)。回傳寫出的事件數。"""
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return len(events)


def _mean(values):
    numbers = [number for value in values if (number := _finite_float(value)) is not None]
    return _rounded(sum(numbers) / len(numbers)) if numbers else None


def _candidate_frame_records(value):
    """正規化逐幀 candidate 計數；重複幀會合併，輸出依 0-based 幀號排序。"""
    totals = {}
    for item in value if isinstance(value, (list, tuple)) else []:
        if not isinstance(item, dict):
            continue
        try:
            frame_index = int(item.get("frame_index"))
            candidate_count = int(item.get("candidate_count"))
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or candidate_count <= 0:
            continue
        totals[frame_index] = totals.get(frame_index, 0) + candidate_count
    return [
        {"frame_index": frame_index, "candidate_count": totals[frame_index]}
        for frame_index in sorted(totals)
    ]


def _candidate_stage(run_summary, count_key, frames_key):
    frames = _candidate_frame_records(run_summary.get(frames_key))
    try:
        candidate_count = max(0, int(run_summary.get(count_key, 0) or 0))
    except (TypeError, ValueError):
        candidate_count = sum(item["candidate_count"] for item in frames)
    return {
        "candidate_count": candidate_count,
        "detected_frame_count": len(frames),
        "detected_frames": frames,
    }


def _compact_analysis_event(event):
    if event.get("type") == "litter":
        segment = event.get("time_segment") or {}
        vehicle = event.get("vehicle")
        thrower = event.get("thrower")
        if vehicle is None and thrower and thrower.get("cls") in VEHICLE_LIKE:
            vehicle = thrower
        vehicle_id = (
            f"{vehicle['cls']}:{int(vehicle['track_id'])}" if vehicle else None
        )
        compact = {
            "type": "litter",
            "id": int(event.get("litter_id", -1)),
            "start_sec": segment.get("start_sec", event.get("time_sec")),
            "end_sec": segment.get("end_sec", event.get("time_sec")),
            "confidence": event.get("detector_confidence"),
            "vehicle": vehicle_id,
            "plate": event.get("license_plate"),
            "plate_confidence": event.get("license_plate_confidence"),
            "plate_status": event.get("license_plate_status"),
            "attribution_status": event.get("backtrack_status"),
            "review_required": True,
        }
        for key in ("confirmation_evidence", "vehicle_quarantine_evidence"):
            value = event.get(key)
            if value is not None:
                compact[key] = value
        return compact

    segment = event.get("time_segment") or {}
    vehicle = event.get("vehicle")
    vehicle_id = (
        f"{vehicle['cls']}:{int(vehicle['track_id'])}" if vehicle else None
    )
    return {
        "type": "urinate",
        "track_id": event.get("track_id"),
        "time_sec": event.get("time_sec"),
        "start_sec": segment.get("start_sec", event.get("time_sec")),
        "end_sec": segment.get("end_sec", event.get("time_sec")),
        "confidence": event.get("conf"),
        "vehicle": vehicle_id,
        "plate": event.get("license_plate"),
        "plate_confidence": event.get("license_plate_confidence"),
        "plate_status": event.get("license_plate_status"),
        "attribution_status": event.get("attribution_status"),
        "review_required": True,
    }


def build_analysis_report(
    run_summary,
    events,
    vehicle_history,
    *,
    fps,
):
    """建立前端精簡 JSON；只報 confidence，不虛構 accuracy。"""
    run_summary = dict(run_summary or {})
    events = list(events or [])
    fps_value = float(fps) if fps and float(fps) > 0 else 30.0
    processed_frames = int(run_summary.get("processed_frames", 0) or 0)
    litter_events = [event for event in events if event.get("type") == "litter"]
    urinate_events = [event for event in events if event.get("type") == "urinate"]
    littering_plates = sorted({
        str(event["license_plate"])
        for event in litter_events
        if event.get("escalated") and event.get("license_plate")
    })
    output_video = run_summary.get("output_video")
    duration_sec = run_summary.get("duration_sec")
    if duration_sec is None:
        duration_sec = round(processed_frames / fps_value, 3)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "video": {
            "file": Path(output_video).name if output_video else None,
            "duration_sec": _rounded(duration_sec, 3),
        },
        "litter_detection": {
            "rtdetr_4channel": {
                "enabled": bool(run_summary.get("rtdetr_enabled", False)),
                "confidence_threshold": _rounded(
                    run_summary.get("rtdetr_confidence_threshold")
                ),
                "evaluated_frame_count": max(
                    0, int(run_summary.get("rtdetr_evaluated_frames", 0) or 0)
                ),
                "vehicle_gate_skipped_frame_count": max(
                    0, int(run_summary.get("vehicle_gate_skipped_frames", 0) or 0)
                ),
                **_candidate_stage(
                    run_summary,
                    "rtdetr_litter_candidates",
                    "rtdetr_litter_candidate_frames",
                ),
            },
            "geometry_passed": _candidate_stage(
                run_summary,
                "raw_litter_candidates",
                "geometry_litter_candidate_frames",
            ),
            "motion_holding_passed": _candidate_stage(
                run_summary,
                "filtered_litter_candidates",
                "filtered_litter_candidate_frames",
            ),
            "confirmed_event_count": len(litter_events),
        },
        "summary": {
            "litter_event_count": len(litter_events),
            "urinate_event_count": len(urinate_events),
            "passed_vehicle_count": len(vehicle_history or {}),
            "average_litter_confidence": _mean(
                event.get("detector_confidence") for event in litter_events
            ),
            "detection_accuracy": None,
            "accuracy_status": "not_evaluated",
            "littering_plates": littering_plates,
            "review_required": bool(events),
        },
        "events": [_compact_analysis_event(event) for event in events],
    }


def write_analysis_json(report, path):
    """Atomic UTF-8 JSON write；前端不會讀到半份檔案。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(target)
