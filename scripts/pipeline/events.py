# -*- coding: utf-8 -*-
"""事件輸出:把一次 run 的確認事件整理成扁平 JSON 記錄,寫成 events.jsonl。

前端監測台的資料來源。與既有 summary.json 分離(不改變 summary 行為),一行一事件,
方便串流/增量讀取。事件建構為純函式,不依賴 GPU/模型,可獨立單元測試。

事件 schema(每行一個 JSON 物件):
  litter:  {type:"litter", frame_index, time_sec, litter_id, bbox:[x1,y1,x2,y2],
            thrower:{cls,track_id}|null, license_plate:str|null, escalated:bool}
  urinate: {type:"urinate", frame_index:null, time_sec:null, confirmed_count:int}
"""
import json

VEHICLE_LIKE = ("vehicle", "scooter")


def _time_sec(frame_index, fps):
    fps = float(fps) if fps else 30.0
    return round(int(frame_index) / fps, 2)


def _plate_for_thrower(thrower_key, vehicle_history):
    # thrower 綁定 vehicle/scooter 時,回傳其車牌(若已辨識)。
    if not thrower_key:
        return None
    cls, track_id = thrower_key[0], thrower_key[1]
    if cls not in VEHICLE_LIKE:
        return None
    info = (vehicle_history or {}).get(track_id)
    if not info:
        return None
    return info.get("license_plate")


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
        item = {
            "type": "litter",
            "frame_index": int(ev.get("frame_index", 0)),
            "time_sec": _time_sec(ev.get("frame_index", 0), fps),
            "litter_id": int(ev.get("litter_id", -1)),
            "bbox": [int(v) for v in ev.get("bbox", [])],
            "thrower": thrower,
            "vehicle": vehicle,
            "license_plate": _plate_for_thrower(
                vehicle_key or thrower_key, vehicle_history
            ),
            "escalated": bool(ev.get("escalated", False)),
            "backtrack_status": ev.get("backtrack_status"),
        }
        if ev.get("backtrack") is not None:
            item["backtrack"] = ev["backtrack"]
        out.append(item)
    return out


def build_urinate_events(urinate_events, run_summary, fps):
    """優先用 per-track 確認明細(action_module.get_urinate_events());沒有時退回 summary
    聚合(相容舊行為:一筆 run-level urinate 事件)。"""
    if urinate_events:
        out = []
        for ev in urinate_events:
            fi = ev.get("frame_index")
            out.append({
                "type": "urinate",
                "track_id": int(ev.get("track_id", -1)),
                "frame_index": int(fi) if fi is not None else None,
                "time_sec": _time_sec(fi, fps) if fi is not None else None,
                "conf": round(float(ev.get("conf", 0.0)), 3),
                "evidence_sec": round(float(ev.get("evidence_sec", 0.0)), 2),
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


def build_run_events(litter_events, urinate_events, vehicle_history, run_summary, fps):
    """組合一次 run 的所有事件:litter 依 frame 排序,urinate(per-track 或聚合)接在後面。"""
    events = build_litter_events(litter_events, vehicle_history, fps)
    events.sort(key=lambda e: e.get("frame_index") or 0)
    events.extend(build_urinate_events(urinate_events, run_summary, fps))
    return events


def write_events_jsonl(events, path):
    """一行一事件寫出 JSONL(UTF-8)。回傳寫出的事件數。"""
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return len(events)
