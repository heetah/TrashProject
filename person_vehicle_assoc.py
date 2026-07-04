"""Event-Anchored Person↔Vehicle Association (offline, rule-based).

獨立的 offline 分析模組:從 tracklet 歷史推斷「每個 person 屬於哪一台 vehicle/scooter」,
即使人車不同時可見(車先到、人下車才被偵測;人下車走遠便溺再走回上車)。

核心思想(見專案討論 2026-06-26):
  關聯證據不在「狀態(同幀共現)」,而在「事件(下車/上車)」。
  - person tracklet 的 birth(下車)/death(上車) 與 vehicle 的 motion-state 轉態(stop/move) 對齊。
  - per-frame 距離在便溺階段是誤導信號,完全不用。
  - 4c-RTDETR litter event 作為硬錨定監督。
  - 1對1 全域最優指派 = 匈牙利演算法(scipy linear_sum_assignment)。
  - dummy/dustbin 欄容許「行人無車」不被強配(SuperGlue, Sarlin et al. CVPR 2020 的規則型版)。
  - 確認車輛類別(過濾欄) 必須在建分數矩陣「之前」,否則 dustbin 會被假車欄搶走。

整合契約(之後接入 scripts-old-test/ 用):
  唯一輸入 `frame_history` 的格式 == litterTracker.actor_frame_history 的元素:
      [{'frame_index': int,
        'actors': [{'cls': 'person'|'vehicle'|'scooter',
                    'track_id': int,
                    'box': [x1,y1,x2,y2],
                    'center': (cx,cy)   # 可省略,缺則由 box 計算
                    'plate_roi': ... }, ...]}, ...]
  注意:pipeline 的 actor_frame_history 是 120 幀 ring buffer;offline 全片關聯需要「完整」歷史,
  整合時請在 run 期間另存一份不截斷的 list,或在收尾時對 dump 出的完整歷史呼叫本模組。

  輸出 AssociationResult 提供 `bound_vehicle_at(person_key, frame)`,簽章刻意對齊
  litterTracker._bound_vehicle_for_person,接入時可直接替換 dismount-edge 查詢。

相依:numpy, scipy(皆為 rtdetr env 既有相依)。零 pipeline 相依,可單獨執行自測:
    python person_vehicle_assoc.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


VEHICLE_CLASSES = ("vehicle", "scooter")
ActorKey = Tuple[str, int]  # (cls, track_id)


# --------------------------------------------------------------------------- #
# 設定(集中所有門檻,接入時可由 env 包一層;此處用 dataclass 保持獨立)
# --------------------------------------------------------------------------- #
@dataclass
class AssocConfig:
    fps: float = 30.0

    # 車輛確認閘門(建欄之前)
    confirm_min_frames: int = 8
    confirm_min_class_frac: float = 0.6
    confirm_min_median_area: float = 900.0

    # motion-state(車輛 stop/move 偵測)
    stationary_speed_thr: float = 0.02   # 位移 / (車高 * 幀);scale-invariant
    motion_smooth_win: int = 5

    # 事件分數權重
    w_birth: float = 1.0
    w_death: float = 1.0
    w_nesting: float = 0.5
    w_litter: float = 2.0

    # 空間/時間核
    prox_sigma: float = 1.0              # 中心距離衰減尺度(以車高為單位)
    birth_death_k: int = 5               # 頭/尾聚合幀數
    moving_at_event_factor: float = 0.3  # 事件當下車在動 → 折扣(仍可能但較弱)
    box_match_max_gap: int = 8           # 找「某幀附近的 box」允許的幀距

    # litter 錨定
    litter_assoc_radius_h: float = 3.0   # litter 與 actor 中心距離 < radius*車高 視為相鄰

    # 指派
    tau: float = 0.35                    # 最低綁定門檻(落 dustbin 的臨界分數)

    # TTL 傳播(person death 後綁定保留)
    bind_ttl_frames: int = 180


# --------------------------------------------------------------------------- #
# 資料契約
# --------------------------------------------------------------------------- #
@dataclass
class Tracklet:
    key: ActorKey
    cls: str                                  # 觀測類別(person/vehicle/scooter)
    boxes: Dict[int, np.ndarray] = field(default_factory=dict)   # frame -> xyxy
    centers: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    cls_history: List[str] = field(default_factory=list)
    plate_frames: List[int] = field(default_factory=list)        # 有 plate_roi 的幀
    locked_cls: Optional[str] = None          # 確認後鎖定的 subtype

    @property
    def frames(self) -> List[int]:
        return sorted(self.boxes.keys())

    @property
    def birth_frame(self) -> int:
        return self.frames[0]

    @property
    def death_frame(self) -> int:
        return self.frames[-1]

    @property
    def median_area(self) -> float:
        if not self.boxes:
            return 0.0
        return float(np.median([_box_area(b) for b in self.boxes.values()]))

    def box_at(self, frame: int, max_gap: int) -> Optional[np.ndarray]:
        """回傳最接近 frame 的觀測 box(幀距 <= max_gap),否則 None。"""
        if frame in self.boxes:
            return self.boxes[frame]
        best, best_gap = None, max_gap + 1
        for f, b in self.boxes.items():
            g = abs(f - frame)
            if g < best_gap:
                best, best_gap = b, g
        return best


@dataclass
class LitterEvent:
    frame_index: int
    center: Tuple[float, float]


@dataclass
class Binding:
    person_key: ActorKey
    vehicle_key: Optional[ActorKey]           # None = 行人/無車
    score: float
    birth_frame: int
    death_frame: int
    ttl_until_frame: int                      # death_frame + bind_ttl_frames


@dataclass
class AssociationResult:
    bindings: Dict[ActorKey, Binding]
    person_keys: List[ActorKey]
    vehicle_keys: List[ActorKey]              # 已確認的車輛欄
    cost_matrix_score: np.ndarray             # 除錯用:C[P x M](未含 dustbin)

    def bound_vehicle_at(self, person_key: ActorKey, frame: int) -> Optional[ActorKey]:
        """簽章對齊 litterTracker._bound_vehicle_for_person:
        在 [birth, death+TTL] 內回傳綁定車輛 key,否則 None。"""
        b = self.bindings.get(person_key)
        if b is None or b.vehicle_key is None:
            return None
        if b.birth_frame <= int(frame) <= b.ttl_until_frame:
            return b.vehicle_key
        return None


# --------------------------------------------------------------------------- #
# 幾何 helper
# --------------------------------------------------------------------------- #
def _to_xyxy(box) -> np.ndarray:
    return np.asarray(box, dtype=np.float64)[:4]


def _box_area(box) -> float:
    x1, y1, x2, y2 = _to_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_height(box) -> float:
    x1, y1, x2, y2 = _to_xyxy(box)
    return max(1.0, y2 - y1)


def _box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = _to_xyxy(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _inter(a, b) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def _iom(a, b) -> float:
    """Intersection over Min-area:部件/重疊偵測,對大小懸殊的人車比 IoU 更穩。"""
    inter = _inter(a, b)
    m = min(_box_area(a), _box_area(b))
    return inter / m if m > 0 else 0.0


def _spatial_score(person_box, vehicle_box, sigma: float) -> float:
    """重疊或鄰近都算:max(IoM, 中心距離以車高正規化的指數衰減)。"""
    iom = _iom(person_box, vehicle_box)
    pc, vc = _box_center(person_box), _box_center(vehicle_box)
    d = math.hypot(pc[0] - vc[0], pc[1] - vc[1]) / _box_height(vehicle_box)
    prox = math.exp(-d / max(sigma, 1e-6))
    return max(iom, prox)


# --------------------------------------------------------------------------- #
# 1) tracklet 重建(吃 actor_frame_history 格式)
# --------------------------------------------------------------------------- #
def tracklets_from_frame_history(frame_history: Sequence[dict]) -> Dict[ActorKey, Tracklet]:
    tracks: Dict[ActorKey, Tracklet] = {}
    for frame in frame_history:
        fi = int(frame["frame_index"])
        for a in frame.get("actors", []):
            cls = str(a.get("cls", "")).lower()
            try:
                tid = int(a["track_id"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (cls, tid)
            box = _to_xyxy(a["box"])
            t = tracks.get(key)
            if t is None:
                t = Tracklet(key=key, cls=cls)
                tracks[key] = t
            t.boxes[fi] = box
            t.centers[fi] = tuple(a.get("center") or _box_center(box))
            t.cls_history.append(cls)
            if a.get("plate_roi") is not None:
                t.plate_frames.append(fi)
    return tracks


# --------------------------------------------------------------------------- #
# 2) 車輛確認閘門(必須在建分數矩陣之前)
# --------------------------------------------------------------------------- #
def confirm_vehicle_tracklets(tracks: Dict[ActorKey, Tracklet],
                              cfg: AssocConfig) -> List[Tracklet]:
    """濾掉閃爍/誤分類/過小的車輛候選,並以多數決鎖定 subtype(scooter/vehicle)。
    機車與汽車都保留(都要開罰),locked_cls 只是固定欄身份避免漂移。"""
    confirmed: List[Tracklet] = []
    for t in tracks.values():
        if t.cls not in VEHICLE_CLASSES:
            continue
        seq = t.cls_history
        if len(seq) < cfg.confirm_min_frames:
            continue
        veh = [c for c in seq if c in VEHICLE_CLASSES]
        if len(veh) / max(len(seq), 1) < cfg.confirm_min_class_frac:
            continue
        if t.median_area < cfg.confirm_min_median_area:
            continue
        n_sc, n_ve = veh.count("scooter"), veh.count("vehicle")
        t.locked_cls = "scooter" if n_sc >= n_ve else "vehicle"
        confirmed.append(t)
    return confirmed


# --------------------------------------------------------------------------- #
# 3) motion-state:車輛 stationary 區間
# --------------------------------------------------------------------------- #
def stationary_intervals(vehicle: Tracklet, cfg: AssocConfig) -> List[Tuple[int, int]]:
    """回傳車輛靜止的 [start_frame, end_frame] 區間清單(frame-index 空間)。"""
    frames = vehicle.frames
    if len(frames) < 2:
        return [(frames[0], frames[0])] if frames else []

    med_h = max(np.median([_box_height(b) for b in vehicle.boxes.values()]), 1.0)
    flags: List[Tuple[int, bool]] = []
    prev_f = frames[0]
    prev_c = vehicle.centers[prev_f]
    flags.append((prev_f, True))  # 起始視為靜止,首位移再修正
    for f in frames[1:]:
        c = vehicle.centers[f]
        dt = max(f - prev_f, 1)
        speed = math.hypot(c[0] - prev_c[0], c[1] - prev_c[1]) / (med_h * dt)
        flags.append((f, speed < cfg.stationary_speed_thr))
        prev_f, prev_c = f, c

    # 多數決平滑,去抖
    win = max(1, int(cfg.motion_smooth_win))
    smoothed: List[Tuple[int, bool]] = []
    for i, (f, _) in enumerate(flags):
        lo, hi = max(0, i - win // 2), min(len(flags), i + win // 2 + 1)
        votes = [flags[j][1] for j in range(lo, hi)]
        smoothed.append((f, sum(votes) >= len(votes) / 2.0))

    intervals: List[Tuple[int, int]] = []
    run_start = None
    for f, st in smoothed:
        if st and run_start is None:
            run_start = f
        elif not st and run_start is not None:
            intervals.append((run_start, prev_stat_f))
            run_start = None
        if st:
            prev_stat_f = f
    if run_start is not None:
        intervals.append((run_start, smoothed[-1][0]))
    return intervals


def _stationary_at(intervals: List[Tuple[int, int]], frame: int) -> bool:
    return any(s <= frame <= e for s, e in intervals)


def _interval_covering(intervals: List[Tuple[int, int]],
                       lo: int, hi: int) -> float:
    """[lo,hi] 被某單一 stationary 區間覆蓋的最大比例(nesting 用)。"""
    span = max(hi - lo, 1)
    best = 0.0
    for s, e in intervals:
        ov = max(0, min(hi, e) - max(lo, s))
        best = max(best, ov / span)
    return best


# --------------------------------------------------------------------------- #
# 4) 事件分數
# --------------------------------------------------------------------------- #
def _event_box(person: Tracklet, at_birth: bool, k: int) -> np.ndarray:
    """person 頭(birth)或尾(death) k 幀的代表 box(中位數),抗單幀抖動。"""
    frames = person.frames
    sel = frames[:k] if at_birth else frames[-k:]
    arr = np.stack([person.boxes[f] for f in sel], axis=0)
    return np.median(arr, axis=0)


def _endpoint_score(person: Tracklet, vehicle: Tracklet,
                    intervals: List[Tuple[int, int]],
                    at_birth: bool, cfg: AssocConfig) -> float:
    event_f = person.birth_frame if at_birth else person.death_frame
    vbox = vehicle.box_at(event_f, cfg.box_match_max_gap)
    if vbox is None:
        return 0.0
    pbox = _event_box(person, at_birth, cfg.birth_death_k)
    s_box = _spatial_score(pbox, vbox, cfg.prox_sigma)
    factor = 1.0 if _stationary_at(intervals, event_f) else cfg.moving_at_event_factor
    return s_box * factor


def _litter_anchor_score(person: Tracklet, vehicle: Tracklet,
                         litter_events: Sequence[LitterEvent],
                         cfg: AssocConfig) -> float:
    """litter event 當幀同時鄰近 person 與 vehicle → 硬錨定。回傳最佳事件分數。"""
    best = 0.0
    for ev in litter_events:
        pbox = person.box_at(ev.frame_index, cfg.box_match_max_gap)
        vbox = vehicle.box_at(ev.frame_index, cfg.box_match_max_gap)
        if pbox is None or vbox is None:
            continue
        vh = _box_height(vbox)
        dp = math.hypot(ev.center[0] - _box_center(pbox)[0],
                        ev.center[1] - _box_center(pbox)[1]) / vh
        dv = math.hypot(ev.center[0] - _box_center(vbox)[0],
                        ev.center[1] - _box_center(vbox)[1]) / vh
        if dp <= cfg.litter_assoc_radius_h and dv <= cfg.litter_assoc_radius_h:
            best = max(best, math.exp(-(dp + dv) / 2.0))
    return best


def build_event_cooccurrence(persons: List[Tracklet],
                             vehicles: List[Tracklet],
                             litter_events: Sequence[LitterEvent],
                             cfg: AssocConfig) -> np.ndarray:
    """事件錨定共現矩陣 C[P x M](越高越綁)。per-frame 距離完全不用。"""
    P, M = len(persons), len(vehicles)
    C = np.zeros((P, M), dtype=np.float64)
    veh_intervals = [stationary_intervals(v, cfg) for v in vehicles]
    for i, p in enumerate(persons):
        for j, v in enumerate(vehicles):
            iv = veh_intervals[j]
            birth = _endpoint_score(p, v, iv, at_birth=True, cfg=cfg)
            death = _endpoint_score(p, v, iv, at_birth=False, cfg=cfg)
            # nesting 必須由端點空間證據 gate:沒在下車/上車靠近此車,
            # 「時間巢狀」對任何背景停車都成立 → 無意義,須歸零(否則行人誤綁)。
            endpoint_evidence = max(birth, death)
            nest = _interval_covering(iv, p.birth_frame, p.death_frame) * endpoint_evidence
            litter = _litter_anchor_score(p, v, litter_events, cfg)
            C[i, j] = (cfg.w_birth * birth + cfg.w_death * death
                       + cfg.w_nesting * nest + cfg.w_litter * litter)
    return C


# --------------------------------------------------------------------------- #
# 5) 匈牙利 + dustbin(全域 1對1,容許未綁定)
# --------------------------------------------------------------------------- #
def assign_with_dustbin(C: np.ndarray, tau: float) -> Dict[int, Optional[int]]:
    """C:(P, M) 綁定分數(越高越好);tau:最低綁定門檻。
    回傳 {person_idx: vehicle_idx 或 None}。None = 落 dustbin(行人/無車)。"""
    P, M = C.shape
    if P == 0:
        return {}
    if M == 0:
        return {r: None for r in range(P)}

    BIG = 1e6
    cost_real = -C.astype(np.float64)                 # 分數 → 成本(求最小)
    cost_dustbin = np.full((P, P), BIG, dtype=np.float64)
    np.fill_diagonal(cost_dustbin, -float(tau))       # person i 只能落自己的 dustbin
    cost_aug = np.hstack([cost_real, cost_dustbin])   # (P, M+P)

    row, col = linear_sum_assignment(cost_aug)
    out: Dict[int, Optional[int]] = {}
    for r, c in zip(row, col):
        out[int(r)] = int(c) if c < M else None
    return out


# --------------------------------------------------------------------------- #
# 6) 頂層編排
# --------------------------------------------------------------------------- #
def associate(frame_history: Sequence[dict],
              litter_events: Optional[Sequence[LitterEvent]] = None,
              cfg: Optional[AssocConfig] = None) -> AssociationResult:
    """單一入口。順序: 確認車輛 → 建事件分數 → 匈牙利+dustbin → TTL 綁定。"""
    cfg = cfg or AssocConfig()
    litter_events = list(litter_events or [])

    tracks = tracklets_from_frame_history(frame_history)
    persons = [t for t in tracks.values() if t.cls == "person"]
    persons.sort(key=lambda t: (t.birth_frame, t.key[1]))

    vehicles = confirm_vehicle_tracklets(tracks, cfg)     # ① 先確認車(過濾欄)
    vehicles.sort(key=lambda t: (t.birth_frame, t.key[1]))

    C = build_event_cooccurrence(persons, vehicles, litter_events, cfg)  # ②
    assign = assign_with_dustbin(C, cfg.tau)              # ③ Hungarian + dustbin

    bindings: Dict[ActorKey, Binding] = {}
    for i, p in enumerate(persons):
        j = assign.get(i)
        vkey = vehicles[j].key if j is not None else None
        # 鎖定 subtype 反映到輸出 key(便於下游 OCR/開罰一致)
        if j is not None and vehicles[j].locked_cls:
            vkey = (vehicles[j].locked_cls, vehicles[j].key[1])
        score = float(C[i, j]) if j is not None else 0.0
        bindings[p.key] = Binding(
            person_key=p.key,
            vehicle_key=vkey,
            score=score,
            birth_frame=p.birth_frame,
            death_frame=p.death_frame,
            ttl_until_frame=p.death_frame + cfg.bind_ttl_frames,
        )

    return AssociationResult(
        bindings=bindings,
        person_keys=[p.key for p in persons],
        vehicle_keys=[v.key for v in vehicles],
        cost_matrix_score=C,
    )


# --------------------------------------------------------------------------- #
# 自測:合成「下車→走遠便溺→走回上車」場景 + 行人干擾 + 遠處停車干擾
# --------------------------------------------------------------------------- #
def _synthetic_frame_history():
    """V1 停在中央,P1 從 V1 下車走遠再走回上車;P2 行人路過(無車);V2 遠處長停(干擾)。"""
    frames = []
    v1_box = [900, 500, 1040, 640]      # 中央車,靜止
    v2_box = [100, 200, 180, 280]       # 遠處小車,靜止(干擾欄)
    for fi in range(0, 420):
        actors = []
        # V1:全程靜止可見
        actors.append({"cls": "vehicle", "track_id": 1, "box": list(v1_box),
                       "center": _box_center(v1_box), "plate_roi": object()})
        # V2:遠處靜止
        actors.append({"cls": "vehicle", "track_id": 2, "box": list(v2_box),
                       "center": _box_center(v2_box), "plate_roi": object()})
        # P1:f60 下車(在 V1 上),走遠到 (300,800),f300 折返,f360 回到 V1 上車
        if 60 <= fi <= 360:
            if fi <= 200:
                t = (fi - 60) / 140.0
                cx = 970 + (300 - 970) * t
                cy = 570 + (800 - 570) * t
            else:
                t = (fi - 200) / 160.0
                cx = 300 + (970 - 300) * t
                cy = 800 + (570 - 800) * t
            pb = [cx - 30, cy - 80, cx + 30, cy + 80]
            actors.append({"cls": "person", "track_id": 7, "box": pb,
                           "center": (cx, cy)})
        # P2:f100~180 行人從左邊路過,從不靠近任何車
        if 100 <= fi <= 180:
            cx = 400 + (cx2 := (fi - 100) * 2)
            pb = [cx - 25, 300 - 70, cx + 25, 300 + 70]
            actors.append({"cls": "person", "track_id": 9, "box": pb,
                           "center": (cx, 300)})
        frames.append({"frame_index": fi, "actors": actors})
    return frames


def _self_test():
    cfg = AssocConfig()
    fh = _synthetic_frame_history()
    # litter event:P1 下車後不久(f80)在 V1 旁丟垃圾
    litter = [LitterEvent(frame_index=80, center=(980, 660))]
    res = associate(fh, litter, cfg)

    print("=== confirmed vehicles ===", res.vehicle_keys)
    print("=== persons ===", res.person_keys)
    print("=== C (person x vehicle) ===")
    print(np.round(res.cost_matrix_score, 3))
    for pk, b in res.bindings.items():
        print(f"  {pk} -> {b.vehicle_key}  score={b.score:.3f}  "
              f"life=[{b.birth_frame},{b.death_frame}]  ttl_until={b.ttl_until_frame}")

    # 斷言:P1(7) 綁 V1(1);P2(9) 行人落 dustbin(None)
    b7 = res.bindings[("person", 7)]
    b9 = res.bindings[("person", 9)]
    assert b7.vehicle_key is not None and b7.vehicle_key[1] == 1, f"P1 應綁 V1,得 {b7.vehicle_key}"
    assert b9.vehicle_key is None, f"P2 行人應落 dustbin,得 {b9.vehicle_key}"

    # TTL:P1 死於 360,180 幀內(<=540) 仍綁定,之後解除
    assert res.bound_vehicle_at(("person", 7), 360) is not None
    assert res.bound_vehicle_at(("person", 7), 539) is not None
    assert res.bound_vehicle_at(("person", 7), 541) is None
    # 便溺階段(走遠,f200)仍綁定(事件錨定不看中間距離)
    assert res.bound_vehicle_at(("person", 7), 200) is not None
    print("\nALL ASSERTIONS PASSED ✓")


if __name__ == "__main__":
    _self_test()
