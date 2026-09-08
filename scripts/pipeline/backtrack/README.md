# Smart Backtrack

> Optional tooling notice: this checkout currently does not include the root
> `validate_old_test_videos.py` or `tools/` annotation/preview CLIs referenced
> by some research commands below. Core backtrack modules and tests remain
> available; CLI-only tests explicitly skip when their optional tool is absent.
> Do not report those workflows as runnable until the files are restored.

Production attribution flow:

```text
actor detections
  -> Kalman prediction + Hungarian (same-object ID only)
  -> confidence-aware Kalman filtering
  -> RTS smoothing

litter trajectory
  -> airborne-prefix extraction
  -> x(t) linear / y(t) quadratic fit in seconds
  -> reverse release hypotheses with covariance

release + actor tracklets
  -> C_BA(litter, person)
  -> C_AC(person, vehicle)
  -> C_BC(litter, vehicle)
  -> event-expanded route candidates
  -> residual-graph successive-shortest-path Min-Cost Flow
  -> authoritative person / vehicle / NULL assignment
```

Semantics:

- Hungarian never performs person↔vehicle or litter attribution.
- Kalman transition/process noise uses seconds and the video's real FPS.
- `C_BA` uses a predicted person upper-body release zone plus an
  uncertainty-independent physical-distance gate. Larger covariance can only
  weaken a match; it cannot make a distant person valid.
- `C_AC` uses person/vehicle endpoint, time, quality and uncertainty evidence.
  Bbox overlap is still recorded for ablation but has zero production weight:
  a wrong-depth vehicle bbox can cover the person. It requires sustained
  support or a strong enter/exit endpoint, and does not reuse the litter anchor.
- `C_BC` is a hard gate only for direct-vehicle routes. On person routes it is
  optional support at the same release frame as `C_BA`, so a person may leave
  a vehicle, walk away, then throw.
- Person pruning ranks completed `B->A->C` routes, not `C_BA` alone.
- A one-point litter history receives a dustbin-level prior. Two distinct
  observations use a constant-velocity reverse hypothesis with anisotropic
  covariance growth and an explicit short-track prior; three or more points
  use the ballistic model. The legacy 0.4 s option is replay-compatible only
  and no longer truncates the physical candidate set.
- Let `B0/B1/B2` be the first three distinct accepted litter detections.
  `H0=T_B1-T_B0` defines the zero-cost interval
  `I0=[T_B0-H0,T_B0]`. Before that interval, the time prior is
  `lambda_w*(T_B0-H0-T_r)/H0`; post-birth points on the observed airborne
  path retain `lambda_f*(T_r-T_B0)/FPS`. Candidates remain available. Early source
  direction is opposite `B0->B1`, while `B2` supplies a velocity-cosine
  consistency diagnostic. `confirm_frame` participates in neither calculation.
- The 8/27 production recovery profile permits actor support to appear after
  the first litter observation, accommodating detector cadence and occlusion.
  Confirmation still requires an actor plus motion, trajectory/displacement
  and non-stationary evidence; attribution remains a separate later stage.
- Rise-then-fall tracks use descent from their internal apex, so a real arc may
  return near its release height without being rejected as a horizontal streak.
- Full-stage `C_BC` sidecars also expose `reverse_direction`, `exit_deficit`
  and `relative_motion_deficit`. Their production weights are zero: the reviewed
  seven-case ablation showed inconsistent directions under two-point occlusion,
  so they remain diagnostics rather than unvalidated rewards.
- Person and vehicle capacities are unbounded: one person may explain several
  litter events and one vehicle may link to several people/events.
- Every event has a NULL route. Invalid or highly uncertain candidates are not
  forced into a match.

Production entrypoint 會從 repository root `.env` 載入以下控制項；shell 明確 export 的值優先，
完整集中範本見 root `.env.example`。Environment controls (defaults shown):

```text
SMART_BACKTRACK=1
SMART_BACKTRACK_MAX_BACK_FRAMES=<2.4 seconds worth of frames; 24 at 10 FPS; computational guard>
SMART_BACKTRACK_CONTEXT_SEC=10.0
SMART_BACKTRACK_RAW_PREFIX=1
SMART_BACKTRACK_TOPK_PERSON=5
SMART_BACKTRACK_TOPK_VEHICLE=5
SMART_BACKTRACK_DUSTBIN_COST=7.0
SMART_BACKTRACK_NULL_VEHICLE_COST=1.4
SMART_BACKTRACK_DIRECT_VEHICLE_COST=0.9
SMART_BACKTRACK_AC_WEIGHT=0.75
SMART_BACKTRACK_BC_SUPPORT_BONUS=0.25
SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT=0.0
SMART_BACKTRACK_SIGMA_FLOOR=2.0
SMART_BACKTRACK_TWO_POINT_MAX_BACK_SEC=0.4
SMART_BACKTRACK_TWO_POINT_PRIOR_COST=1.0
SMART_BACKTRACK_MAX_FORWARD_RELEASE_SEC=0.5
SMART_BACKTRACK_RELEASE_WINDOW_WEIGHT=0.35
SMART_BACKTRACK_SIDECAR=0
SMART_BACKTRACK_STUDY_STAGE=full  # full | distance_time | kalman_rts | confidence | uncertainty | reverse
SMART_BACKTRACK_DT_DISTANCE_WEIGHT=1.0
SMART_BACKTRACK_DT_TIME_WEIGHT=1.0
SMART_BACKTRACK_TIME_COST_MODE=soft  # soft | hard (legacy replay)
SMART_BACKTRACK_TIME_SOFT_KAPPA=4.0
SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=0
SMART_BACKTRACK_HOMOGRAPHY_MIN_CONFIDENCE=0.65
```

The Homography switch is an opt-in production code path, not a promotion of the
current calibration constants. Each confirmed event records one snapshot
decision. Only a numerically valid `LOCKED` runtime H at or above the confidence
threshold is used for `C_BA`, `C_BC` and `C_AC` spatial features. Missing,
unlocked, low-confidence or invalid snapshots keep the complete image-space
cost path and NULL route, with the fallback reason in candidate diagnostics.
Kalman/RTS state estimation and temporal evidence remain in image coordinates.

Confirmed litter 送入 resolver 前，tracker 可從尚未經 geometry/motion/
holding 後處理的 RT-DETR bbox，向前恢復連續的 raw trajectory prefix。
Raw bbox 只補 release trajectory；不能建立、延長或確認 litter event。
`C_BC` 的 `boundary_depth` 是 release 點到原始 vehicle bbox 最近邊界的
正規化深度。它用來降低巨大遮擋 bbox 把鄰車 release 點吃進車體深處的
優勢，仍是 soft cost，不會放寬任何 physical hard gate。

Actor geometry is already retained in bounded queues before a litter event is
confirmed. The OCR-capable queue keeps 120 frames; the lightweight Smart
Backtrack queue keeps 10 seconds by default plus the post-birth allowance and
omits image crops. Exact detector boxes, frame indices, confidence, observed
flags and tracklet epochs are copied into the immutable resolver task. Kalman/
RTS therefore does not replace history storage: it fills missing frames,
smooths detector jitter and supplies covariance. Production currently uses the
smoothed state even at observed frames while retaining the nearest real
detector frame as evidence for the freshness gate.

`SMART_BACKTRACK=0` keeps the legacy resolver available as a rollback path.
When smart mode is enabled, the legacy heuristic is used only if the smart
resolver raises an exception.

The flow graph is event-expanded, so an event-dependent `C_BC(B,C)` is
represented exactly. With all actor capacities intentionally unbounded, events
currently decompose mathematically into independent shortest routes. The graph
form is retained for explicit NULL handling and later cross-event consistency
constraints; it should not be described as adding cross-event coupling today.

## Reproducible cost study

`scripts/backtrack_study.py` replays `resolver_input` stored in a candidate
sidecar.  It never invokes detector inference or promotes a candidate into a
confirmed litter event.  A trial uses an immutable JSON config and a grouped
60/20/20 manifest. `kalman_rts` is a clean branch from D+T; the later
`uncertainty` stage combines confidence and Kalman/covariance evidence:

```text
distance_time -> confidence ---------> uncertainty -> reverse -> full
            \-> kalman_rts (pure) ----/
```

`distance_time` uses only distance/time cost components, original observed
actor boxes, and the litter birth anchor. Distance is divided by its hard gate
before weighting, so its accepted feature is a dimensionless fraction in the
range 0--1. Time is also dimensionless, but in production soft mode it may
exceed 1 and receives the continuous excess penalty defined below. The DT
environment weights affect all three matrices; remaining hard gates stay
independent and cannot be relaxed by a weight. `kalman_rts` enables actor
position smoothing/interpolation while
keeping uniform Kalman measurement confidence and disabling covariance gates,
covariance/Mahalanobis costs, and reverse release fitting. `uncertainty` adds
confidence and covariance evidence; `reverse` enables ballistic release hypotheses with uniform litter
weights; `full` additionally restores confidence-weighted trajectory fitting.
All route types and the full NULL route remain present in every stage.
`SMART_BACKTRACK_MAX_BACK_FRAMES` bounds enumeration/runtime only. Every emitted
hypothesis records `search_truncated` and its reason, so research reports do not
misstate this engineering guard as a physical release-time gate.

Production direct litter→vehicle association uses the original, unexpanded
vehicle bbox. For release point `p` and vehicle bbox `B` with width `w` and
height `h`, the physical distance is
`D=dist(p,B)/sqrt(w^2+h^2)` and the hard gate is `D<=0.40`. Actor observation
freshness is a hybrid soft penalty rather than a physical cutoff. Let
`z_s=(gap_frames/FPS)/0.25`, `z_f=gap_frames/3`, and `z=max(z_s,z_f)`; production
uses `rho(z)=z+kappa*max(0,z-1)^2` with `kappa=4`. The former `0.25 s` and
`3 frames` thresholds are therefore dimensionless bend points. `max` preserves
the stricter side of the former AND rule without adding two perfectly dependent
measurements. Bounded actor history/Kalman extrapolation, spatial/uncertainty
hard gates and the full NULL route remain the outer safety mechanisms.

`SMART_BACKTRACK_TIME_COST_MODE=hard` exactly restores the former temporal AND
gate for replay: internally the seconds limit is converted with
`floor(FPS*0.25)` and then minimized with the 3-frame cap. The soft curvature
is exposed as `SMART_BACKTRACK_TIME_SOFT_KAPPA`; `4.0` is the declared production
baseline requested for this iteration, not a statistically proven universal
constant. Tune it only with event-grouped validation.

Research replay can override these fields through `StudyConfig`, including
`vehicle_bbox_expand_x_ratio` and `vehicle_bbox_expand_y_ratio`. The legacy
control is reproducible with expansion `0.18/0.15`, vehicle distance gate
`0.8`, and `max_observation_gap_frames=None`; these are no longer production
defaults. Seconds preserve physical meaning across FPS, while the frame cap
limits detector-miss tolerance.

`kalman_rts` trial configs may additionally set
`kalman_process_noise_scale`, `kalman_measurement_noise_scale`, and
`kalman_max_extrapolation_seconds`. These affect actor smoothing/interpolation
only; they do not add confidence, covariance, or Mahalanobis cost components.
Kalman states supply geometry only. Their D+T time feature remains the gap to
the nearest real detector frame, so interpolation cannot silently rewrite
missing actor evidence as a zero synchronization gap.

For the narrower queued-bbox ablation, set
`preserve_observed_actor_boxes=true`. An exact detector bbox from the actor
history ring is then used unchanged at observed frames; Kalman/RTS geometry is
used only at missing frames. This switch is constructor/StudyConfig-only and
does not alter production defaults.

New candidate rows contain `resolver_input`, the JSON-safe immutable worker
task required for replay. Rows without it are diagnostic-only and are rejected
by the replay CLI rather than silently re-running a different pipeline.
`resolver_input` intentionally excludes legacy `plate_actor_frames` and every
`plate_roi` image buffer: Smart resolver never reads those OCR-only pixels, and
expanding NumPy crops into JSON would make sidecars several GB. Actor geometry,
track IDs, confidence, observation flags and litter history remain replayable.


### Release time / vehicle cost proposal (2026-09-05)

An explicit research configuration supports `normalize_bc_distance_time_by_gate=true`
with `normalized_distance_gate_vehicle=0.4`. In the full stage this normalizes
the BC distance term to `w_D * D/0.4`; its time term follows the selected
observation-time mode. Production soft mode uses `w_T*rho(max(z_s,z_f))`, while
`observation_time_cost_mode=hard` reproduces the former `T_E/0.25` feature and
AND gate. Other BC terms and BA/AC weights remain unchanged.

Set `max_release_back_seconds=1.0`, `release_soft_seconds=0.25` and
`release_time_weight=1.0` to replace the backward B0/B1 window prior with:

```text
T_RB = max(0, (resolver_birth_frame - release_frame) / FPS)
R = 0                                 for T_RB <= 0.25
R = ((T_RB - 0.25) / 0.75)^2           for 0.25 < T_RB <= 1.0
reject hypothesis                     for T_RB > 1.0
```

The reference is the frozen resolver's birth frame, not GT release or
confirm frame. The physical limit is applied before hypothesis enumeration,
using `floor(FPS * max_release_back_seconds)`. Video-start bounds prevent
negative release frames. A tighter computational limit still sets
`search_truncated`; reaching the physical boundary alone does not. One-point
fallback and two-point model priors remain intact. Post-birth hypotheses are
still bounded by the observed airborne path and retain their existing forward
penalty. Every event retains its full NULL route.

These fields are constructor/StudyConfig options, not new production environment
switches. Production defaults use D=0.4 and the observation-gap prior. Run
the four controlled proposals against the same frozen confirmed detections:

```bash
OPENBLAS_NUM_THREADS=1 conda run -n rtdetr python scripts/replay_release_policy.py \
  --candidates output/rtdetr_recovery_horiz1_full_20260826 \
  --output artifacts/release_policy_new_run
```

The output directory must be new. It contains exact trial configurations,
resolved per-FPS configurations, input hashes and event selections. Its numeric
tracker-ID score is a legacy same-run diagnostic only: IDs cannot be compared
across inference runs and the score is not product-readiness eligible. The
legacy manual +2 adjustment for cases 74/174 is separately labeled and is not
new validation of those cases. Missing confirmed clips stay in the denominator.
Use `scripts/evaluate_reviewed_readiness.py` for same-run bbox mapping. This is
exploratory development replay, not a held-out evaluation.

## Candidate/component sidecar

Research runs with `SMART_BACKTRACK_SIDECAR=1` additionally include:

```text
<annotated-video-stem>_backtrack_candidates.jsonl
```

Production keeps it disabled so the frontend receives only `analysis.json`.
The research sidecar stores labeling, gate-analysis and cost-calibration data;
it is not part of the frontend schema.

Schema `smart-backtrack-candidates/v1` contains:

- one `run` record per clip, including FPS, processed frame count, solver
  summary and an optional clip-level weak label;
- one `candidate` record per confirmed litter event;
- complete litter history and every reverse-release hypothesis;
- every actor tracklet observation, including `tracklet_uid`;
- `C_BA`, `C_AC`, `C_BC`, per-release cells, valid/rejected state and hard-gate
  reason;
- for valid cells, `raw_features`, `weights`, weighted `components`, and
  `feature_details` showing all three together;
- routes before top-K pruning, pruning reason, final ranked routes and exactly
  one full NULL route;
- selected assignment, microscale Min-Cost Flow cost, route margin, distinct
  person/vehicle margins and NULL-vs-non-NULL margin.

JSON is strict: non-finite values become JSON `null`, never `NaN` or
`Infinity`. Cost scaling uses `1e6` microscale precision and the same
half-away-from-zero rule as the flow solver; this prevents sub-0.001 actor
differences from becoming route-ID ties.

To create sidecars without writing annotated videos:

```bash
conda run -n rtdetr python validate_old_test_videos.py \
  /path/to/video1.mp4 /path/to/video2.mp4 \
  --pipeline-dir scripts \
  --sidecar-dir artifacts/backtrack_candidates
```

## Annotation workflow

Candidate output and ground truth use different schemas. Initialization may
copy an explicit clip-level weak label such as a parent directory named
`litter`; it never copies predicted event validity, release frame, person,
vehicle, selected route, cost or rank into ground truth.

Create the independent annotation queue:

```bash
conda run -n rtdetr python tools/backtrack_annotations.py init \
  artifacts/backtrack_candidates \
  -o artifacts/backtrack_annotations.jsonl
```

Create model-blind contact sheets for review:

```bash
conda run -n rtdetr python tools/render_backtrack_annotation_previews.py \
  artifacts/backtrack_candidates \
  artifacts/backtrack_annotation_previews \
  --strict
```

Each sheet shows event timing, litter history and actor `tracklet_uid`, but
hides release hypotheses, selected route, rank, cost and margin by default.
Use `--show-release-hypotheses` and `--show-model-decisions` only during later
error analysis.

AI-assisted first-pass proposals can be overlaid into a new file without
promoting any review state:

```bash
conda run -n rtdetr python tools/apply_backtrack_ai_proposals.py \
  --annotations artifacts/backtrack_annotations.jsonl \
  --proposals artifacts/ai_label_proposals_*.jsonl \
  -o artifacts/backtrack_annotations_ai_proposed.jsonl
```

The result keeps every field `unreviewed` and records
`ai_proposal.status=pending_human_review`. It is an editable queue, not final
ground truth.

For every event, a reviewer must label:

- `event_label`: `litter` or `not_litter`;
- `release_interval`: inclusive start/end frame;
- one or more `admissible_routes`;
- `ignore=true` only when no reliable truth can be established.

Route semantics:

```text
person_vehicle : person != NULL, vehicle != NULL
person         : person != NULL, vehicle == NULL
direct_vehicle : person == NULL, vehicle != NULL
null           : person == NULL, vehicle == NULL
```

`NULL` means no safely attributable actor. If multiple answers are genuinely
possible, list each legal tuple in `admissible_routes`; do not turn uncertainty
into a forced NULL. Different people/events may reference the same vehicle.

Validate during labeling and require all fields before final evaluation:

```bash
conda run -n rtdetr python tools/backtrack_annotations.py validate \
  artifacts/backtrack_annotations.jsonl

conda run -n rtdetr python tools/backtrack_annotations.py validate \
  artifacts/backtrack_annotations.jsonl --require-reviewed
```

Evaluate reviewed data:

```bash
conda run -n rtdetr python tools/backtrack_annotations.py evaluate \
  --candidates artifacts/backtrack_candidates \
  --annotations artifacts/backtrack_annotations.jsonl \
  -o artifacts/backtrack_metrics.json
```

When `--candidates` points to a pipeline output directory, the evaluator reads
only `smart-backtrack-candidates/v1` records and ignores adjacent frontend
analysis JSON files.

The report separates candidate coverage from ranking quality and includes
Exact Route Top-1, Recall@1/3/5, person/vehicle/route coverage, release interval
hit/MAE and NULL classification. Unreviewed and ignored events do not enter
headline accuracy.
