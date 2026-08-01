# Smart Backtrack

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
- `C_AC` uses only person/vehicle endpoint, overlap, time, quality and
  uncertainty evidence. It requires sustained support or a strong
  enter/exit endpoint, and does not reuse the litter anchor.
- `C_BC` is a hard gate only for direct-vehicle routes. On person routes it is
  optional support at the same release frame as `C_BA`, so a person may leave
  a vehicle, walk away, then throw.
- Person pruning ranks completed `B->A->C` routes, not `C_BA` alone.
- A 1-2 point litter history receives a dustbin-level prior instead of
  pretending that a precise ballistic fit exists.
- Person and vehicle capacities are unbounded: one person may explain several
  litter events and one vehicle may link to several people/events.
- Every event has a NULL route. Invalid or highly uncertain candidates are not
  forced into a match.

Environment controls (defaults shown):

```text
SMART_BACKTRACK=1
SMART_BACKTRACK_MAX_BACK_FRAMES=<2.4 seconds worth of frames; 24 at 10 FPS>
SMART_BACKTRACK_CONTEXT_SEC=10.0
SMART_BACKTRACK_TOPK_PERSON=5
SMART_BACKTRACK_TOPK_VEHICLE=5
SMART_BACKTRACK_DUSTBIN_COST=7.0
SMART_BACKTRACK_NULL_VEHICLE_COST=1.4
SMART_BACKTRACK_DIRECT_VEHICLE_COST=0.9
SMART_BACKTRACK_AC_WEIGHT=0.75
SMART_BACKTRACK_BC_SUPPORT_BONUS=0.25
SMART_BACKTRACK_SIGMA_FLOOR=2.0
SMART_BACKTRACK_SIDECAR=1
```

`SMART_BACKTRACK=0` keeps the legacy resolver available as a rollback path.
When smart mode is enabled, the legacy heuristic is used only if the smart
resolver raises an exception.

The flow graph is event-expanded, so an event-dependent `C_BC(B,C)` is
represented exactly. With all actor capacities intentionally unbounded, events
currently decompose mathematically into independent shortest routes. The graph
form is retained for explicit NULL handling and later cross-event consistency
constraints; it should not be described as adding cross-event coupling today.

## Candidate/component sidecar

Normal pipeline output now includes:

```text
<annotated-video-stem>_backtrack_candidates.jsonl
```

Set `SMART_BACKTRACK_SIDECAR=0` to disable it. The sidecar is separate from
frontend `events.jsonl`: events stay compact, while the research sidecar keeps
the data needed for labeling, gate analysis and cost calibration.

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
- selected assignment, scaled Min-Cost Flow cost and margin.

JSON is strict: non-finite values become JSON `null`, never `NaN` or
`Infinity`. Cost scaling uses the same half-away-from-zero rule as the flow
solver.

To create sidecars without writing annotated videos:

```bash
conda run -n rtdetr python validate_old_test_videos.py \
  /path/to/video1.mp4 /path/to/video2.mp4 \
  --pipeline-dir scripts-old-test \
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

The report separates candidate coverage from ranking quality and includes
Exact Route Top-1, Recall@1/3/5, person/vehicle/route coverage, release interval
hit/MAE and NULL classification. Unreviewed and ignored events do not enter
headline accuracy.
