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

### Ordinal occlusion feasibility boundary

`SMART_BACKTRACK_MASK_DIAGNOSTICS` only exports bounded mask/litter scalars for
research. The resolver does not read them and their production weight is zero.
Visible segmentation-mask containment is not metric depth and does not identify
the litter source; a foreground vehicle may contain the litter pixel while being
the wrong thrower.

The historical `build_ordinal_occlusion_package.py` created a separate,
attribution-blinded queue from observed vehicle/scooter overlaps. Its one-off
source is now in `artifacts/research_python_archive_20260915.tar.gz`; the
existing research result and limitations remain documented. It never reads route,
assignment, release, litter, or ground-truth actor fields. Public rows use opaque
A/B aliases; raw paths, model tracklet IDs, and display boxes stay in a private
mapping. The current builder is still event-conditioned and uses model box
proposals, so its output is feasibility material rather than independent ordinal
truth. It does not modify `runs/grounding_truth`, the production UI, resolver
tasks, costs, or NULL behavior.

One reviewer may label `a_front_b`, `b_front_a`, `tie`, or `unknown`; uncertainty
must not be forced into a direction. A delayed blinded repeat round with A/B
reversal measures intra-rater consistency. Camera IDs and source-video groups
must be verified before camera-separated evaluation. Correlated frames cannot be
counted as independent accuracy samples.

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

### Litter observation provenance contract

Each immutable resolver task keeps the legacy `history_sources` values
(`accepted_tracker` or `raw_rtdetr_recovered`) for sidecar/replay compatibility.
Two additive diagnostic arrays have the same length and ordering as `history`,
`history_frames`, `history_boxes` and `history_confidences`:

- `history_provenance`: exact producer `detector`, `visual_bridge`,
  `raw_rtdetr_recovered`, or conservative `legacy_unknown` for old callers;
- `history_lineage`: deterministic observation ID, optional parent observation,
  independence-group ID, `derived`, and `independent_measurement`.

Detector observations are independent measurement roots. A `visual_bridge`
inherits the preceding observation's independence group. A recovered raw prefix
is retrospectively selected from the first accepted anchor and is also marked
non-independent. A duplicate observation ID remains in the legacy trajectory
when necessary for behavior compatibility, but its later lineage row is not
independent. This prevents future research from treating one observation and
its derived copies as separate statistical evidence.

Phase 0 is serialization-only: `SmartBacktrackResolver` does not read either
new array. Let `P` remove the two new fields from enriched task `T'`; then the
production contract is `resolve(T') = resolve(P(T'))`. The regression test
checks selected route, person, vehicle and every route cost exactly. No release,
cost, confirmation, route or NULL rule changed. Multi-hypothesis smoothing is a
later research phase and must not consume these fields without a separately
validated policy.

### Release-state marginalization primitives (Phase 1A, research only)

`release_marginalization.py` has no production caller. It provides three strict
building blocks for later experiments:

1. Gaussian NLL
   `0.5 * (d*log(2*pi) + log(det(Sigma)) + r^T*Sigma^-1*r)` computed by
   Cholesky factorization and solve. Residual and covariance axes must have
   matching dimensions and units; covariance must be finite, exactly symmetric
   and positive definite. The function never repairs covariance or adds jitter.
2. Caller-defined release-time prior density discretized in seconds. For sorted
   unique points `t_i`, interior cell boundaries are
   `b_i=(t_(i-1)+t_i)/2`; the caller supplies both outer domain boundaries.
   Mass is `density_i*(b_(i+1)-b_i)` followed by global normalization.
   Duplicate time points merge before integration and cannot gain prior mass.
3. Stable log-sum-exp marginalization of auditable hypotheses within routes and
   then across routes. Every hypothesis carries an ID, route ID, release time,
   log likelihood and log prior mass. Conditional hypothesis priors and route
   priors must already sum to one; a caller-specified NULL route and hypothesis
   are mandatory.

`-inf` log likelihood/prior denotes exactly zero likelihood/mass. A zero-
evidence route receives posterior zero when another route has finite evidence;
all-zero route evidence fails explicitly. NaN and `+inf` remain invalid. Prior
normalization tolerance uses the standard floating-point forward-error bound
`gamma_(N+1) = ((N+1)u)/(1-(N+1)u)`, where `u=ulp(1.0)/2`; it accounts for N
exponentiations and the final accurately rounded sum, rather than introducing a
model threshold. This bound assumes the standard floating-point model for the
platform `exp`/`fsum` implementations; it is not a cross-libm formal guarantee.
A singleton prior must be exactly `log(1)=0`.

The module does not read `history_lineage`, multiply observation likelihoods,
construct identity/visibility states, choose top-1, or write resolver output.
Its posterior is conditional on explicitly supplied priors and likelihoods; it
is not calibrated attribution accuracy. Release windows, bandwidths, gates and
weights remain entirely caller-supplied. Production `trajectory.py`,
`costs.py`, `resolver.py`, flow routes and schemas are unchanged.

### Target-48 guarded route composition (research only)

The frozen 58-clip development sidecar was also evaluated with a research-only
composition of five guarded comparisons: a ballistic boundary-ambiguity check,
same-vehicle person disambiguation from `C_AC` endpoint distance, a high/low
`C_BC` quality consistency check, a weak person-to-vehicle association check,
and a same-model motion-consistency check. Each comparison uses only existing
dimensionless sidecar features and an explicit route-cost-difference bound; it
does not add a production weight, alter event confirmation, or remove the
complete `NULL` route. `boundary_depth` remains a normalized image-box edge
diagnostic, not metric depth or a pseudo-homography reconstruction.

On the reviewed positive development cohort this candidate changed 9 of 93
confirmed-record assignments and moved the official fail-closed result from
43/58 to 48/58 route-correct clips (51/58 assignment-conditioned event
matches), with five gains and zero losses. The exact paired sign-test
diagnostic is `p=0.0625`; the point estimate reaches the interim 48/58 target,
but the 85% gate requires 50/58 and the Wilson lower bound is only 0.711.
Nine existing confidence/scale/compression sidecars reproduce 48/58 with no
losses. This is decision-stability evidence on the same development cohort,
not calibrated accuracy or cross-camera generalization. The candidate has no
production caller; thresholds must be preregistered and re-evaluated on a
reviewed camera/source-recording-disjoint holdout and a reviewed negative set
before any promotion.

The complete rule equations, hashes, case-level gains, temporal-offset
sensitivity and rollback boundary are recorded in
`versions/2026-09-15_codex_target48_guarded_route_research.md`.

### Single-reviewer camera/site holdout worksheet (research only)

`camera_holdout_review.py` creates a deterministic `camera-holdout-review/v1`
worksheet from the 18-case camera-safe index. It copies only immutable source
filename/path/SHA-256 fields and one blank `review` object per case; model
summaries, assignments, route scores, release hypotheses and annotated output
paths are stripped. `verify_source_files` rehashes each source before review,
and `validate_completed_review` requires one reviewer, explicit positive /
negative / ambiguous / unusable status, source SHA confirmation,
camera/site/session/source-recording IDs and reviewed timestamps. It reports
camera-disjoint readiness only when the caller supplies non-empty reviewed
camera/source groups and no exact provenance overlap; it never invents a
camera ID or treats missing labels as negatives. This worksheet has no
production caller and does not compute attribution accuracy.

Build the worksheet from the existing camera-safe index and verify the source
bytes before handing it to the reviewer:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_camera_holdout_review.py \
  --index /tmp/target41_camera_safe_review_index.json \
  --output /tmp/target41_camera_safe_blinded_review.json \
  --verify-sources
```

The command refuses to overwrite an existing file and keeps all review fields
blank. After the reviewer fills the copy, run `validate_completed_review` with
the reviewed camera/source-recording sets before any accuracy evaluator.

The standalone gate reports a blank worksheet as not ready and exits non-zero;
it likewise rejects a partially completed or malformed worksheet. It never
infers missing provenance or starts an evaluator:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/validate_camera_holdout_review.py \
  --review /tmp/target41_camera_safe_blinded_review.json \
  --verify-sources
```

After human review, repeat the command with the independently reviewed
reference `--reviewed-camera-id` and `--reviewed-source-recording-id` values
(and, when available, `--reviewed-source-hash`). Exit code `0` means the
provenance/coverage gate is ready for a separately frozen evaluation; it does
not certify attribution accuracy.

The current blind runtime sample under `/tmp/target41_camera_safe_run` has
18/18 analysis JSON files, backtrack sidecars, and decodable H.264 MP4 output.
This is a container/output smoke check only; the worksheet remains
`unreviewed`, so no accuracy or attribution number may be derived from it.

The frozen Target-48 composition was also replayed in memory against these
blind sidecars as a behaviour preflight. It read no truth labels and wrote no
production or evaluation sidecar: 29 confirmed-record rows from 13 clips were
parsed, one `direct_person` guard changed one assignment (`litter_case_202`),
and all 29 rows retained a NULL alternative, one selected-route alignment, and
the original input path. The 18 source SHA-256 values are unique and the
source-case audit reports no overlap with the reviewed source cases, but
camera/site identity is still unverified; this is source-recording/output
stability evidence, not a camera-independent accuracy result. The worksheet
remains 18/18 `unreviewed` and the camera-disjoint gate remains blocked.

For spreadsheet-friendly handoff, the worksheet can be exported and imported
through a strict, model-blind CSV round trip. The JSON template remains the
source of immutable case identity; the table contains exactly the columns
`case_id`, `video_filename`, `source_video`, `source_sha256`, the explicit
review/provenance fields, `events_json`, and `notes`. Extra prediction columns,
changed paths or hashes, duplicate/missing case IDs, malformed booleans, and a
changed header fail closed. Importing a partial table is allowed so a reviewer
can save progress, but it never makes the worksheet evaluation-ready:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/export_camera_holdout_review.py \
  --review /tmp/target41_camera_safe_blinded_review.json \
  --output /tmp/target41_camera_safe_blinded_review.csv \
  --verify-sources

PYTHONPATH=scripts conda run -n rtdetr python scripts/import_camera_holdout_review.py \
  --template /tmp/target41_camera_safe_blinded_review.json \
  --table /tmp/target41_camera_safe_blinded_review.csv \
  --output /tmp/target41_camera_safe_blinded_review_roundtrip.json \
  --verify-sources
```

Both commands refuse to overwrite an existing output. The CSV contains no
model summaries, route IDs, scores, release hypotheses, or annotated output
paths. After the reviewer finishes the imported JSON, run the validation gate
above and supply independently reviewed camera/source-recording groups.

### Replay decision consensus (research only)

`replay_consensus.py` is a pure, no-caller primitive for checking whether a
single event keeps the same route under a caller-supplied set of replay
conditions (for example confidence, image scale or codec perturbations). It
canonicalizes `(route_type, person_key, vehicle_key)` and deliberately ignores
local `route_id` values when comparing conditions; those IDs are retained only
for audit output. A route is returned as `consensus_route` only when every
condition agrees. Any disagreement, including disagreement with an explicit
NULL route, returns `safe_route = ("null", None, None)` and sets
`manual_review_required`; no plurality or top-1 choice is made. Unanimous NULL
is preserved as a valid consensus, not treated as missing data.

The check validates unique non-empty condition IDs, strict actor-key shapes and
route semantics, and sorts audit rows by condition ID for deterministic replay.
It does not combine scores, multiply observations, infer camera/depth
relationships, or change resolver costs, release logic, route schemas or NULL
behavior. The nine-condition development replay currently has 58/58 identical
route tuples, but this primitive does not convert that in-sample invariance
into accuracy or camera-independent robustness evidence.

The retained `{0,+1}` release-mask window also passed a fixed single-offset
sensitivity screen: offsets `-1`, `0`, and `+1` produced 41/58, 43/58, and
42/58 respectively, with no route or accepted-event losses; offsets `-2` and
`+2` produced 40/58. This is a development missing-observation check only and
does not justify a production weight or a camera-generalization claim.

The frozen mask replay is also isolated from detector/tracker evidence: its
93 baseline/candidate records have identical keys and route payloads, with
zero non-decision evidence differences; only five documented assignment
selection mirrors changed. See
`/tmp/target41_mask_candidate_lineage_audit.json`.

Exact source-file lineage also supports a paired source-case sensitivity check:
on 54 clips in 50 SHA-linked source groups, the candidate-minus-baseline rate
difference was +9.26 percentage points with a grouped bootstrap 95% interval
of +1.92 to +17.54 points. The groups are source recordings, not verified
cameras, and the replay was selected on the same development set; see
`/tmp/target41_source_case_paired_bootstrap.json` and do not treat this as
confirmatory significance.

`SMART_BACKTRACK=0` keeps the legacy resolver available as a rollback path.
When smart mode is enabled, the legacy heuristic is used only if the smart
resolver raises an exception.

The flow graph is event-expanded, so an event-dependent `C_BC(B,C)` is
represented exactly. With all actor capacities intentionally unbounded, events
currently decompose mathematically into independent shortest routes. The graph
form is retained for explicit NULL handling and later cross-event consistency
constraints; it should not be described as adding cross-event coupling today.

### Target-41 evidence ledger (research only)

Maintenance note: this target-41 implementation and its focused tests were
superseded by the frozen target-48 study and moved to
`artifacts/research_python_archive_20260915.tar.gz`. The section below is a
historical evidence record; restore the archive into an isolated directory
before attempting its old commands. It is not an active production workflow.

`target41_evidence.py` and `scripts/build_target41_evidence_ledger.py` verify
that baseline/candidate case tables and their official readiness reports have
the same fixed denominator, reviewed labels, and exact case IDs. The ledger
records each case's outcome transition and an explicit explanation code, then
checks the separately supplied replay-consensus report. It does not rerun
matching, infer truth, select a route, or call production code.

Example for the frozen `{0,+1}` replay:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_target41_evidence_ledger.py \
  --baseline-cases /tmp/target41_mask_temporal_aggregate/+0_+1/baseline_eval/case_outcomes.csv \
  --candidate-cases /tmp/target41_mask_temporal_aggregate/+0_+1/candidate_eval/case_outcomes.csv \
  --baseline-readiness /tmp/target41_mask_temporal_aggregate/+0_+1/baseline_eval/readiness.json \
  --candidate-readiness /tmp/target41_mask_temporal_aggregate/+0_+1/candidate_eval/readiness.json \
  --consensus /tmp/target41_mask_decision_consensus.json \
  --expected-denominator 58 --target-count 41 \
  --output /tmp/target41_evidence_ledger.json
```

The resulting ledger reports 43/58 candidate correctness, five explicit
`wrong_route -> correct_route` gains, zero losses, and nine replay conditions
with no unstable cases. These are development-set evidence and decision
consistency, not calibrated accuracy or camera-independent robustness.

The 18-case camera-safe runtime audit currently reports 29 event snapshots,
`applied_events=0`, and 29 `feature_disabled` fallbacks for dynamic
pseudo-homography. Thus the 43/58 mask replay contains no implicit 3D-depth
effect; a future H experiment requires a separately reviewed continuous-camera
run that reaches `LOCKED`.

### Target-41 robustness certificate (research only)

Maintenance note: the certificate builders/modules and their tests are stored
in the same research archive. Existing hashes and conclusions are retained for
audit, while the active tree keeps the target-48, Phase 0/1A, replay-consensus,
camera-holdout and release-validation paths.

`target41_robustness.py` joins the audited target-41 evidence ledger with its
replay-consensus report and emits a strict, hash-linked status certificate. It
reports the development point estimate (currently 43/58), complete per-case
explanation coverage, and the nine supplied confidence/scale/compression replay
conditions (58/58 route tuples consistent). The certificate is an accounting
and reproducibility primitive: it does not rerun inference, multiply correlated
observations, choose a route, infer camera identity, or modify resolver costs,
release logic, schemas, or NULL behavior. It must not be described as calibrated
accuracy, a confidence interval, or cross-camera generalization.

The explanation vocabulary is closed and validated against the six outcome
reasons emitted by `target41_evidence.py`; an unknown code fails closed. The
candidate outcome is also checked against its one-to-one explanation reason,
so a known code cannot be attached to the wrong result semantics. The
certificate also emits `case_ids_by_explanation`, a deterministic mapping from
each explanation code to the exact clip IDs, so aggregate counts cannot hide a
case-level audit gap.
It also emits `case_explanations`: one deterministic row per clip containing
the baseline/candidate outcomes, expected route type, and selected route
tuples, plus the case-table observables (`match_tier`, event-match flag,
mapped actor keys, and provisional correctness) on each side. These are audit
summaries only; they do not become resolver input or ground truth.

For stronger replay provenance, `replay_evidence.py` and
`scripts/build_target41_replay_evidence.py` consume the official
`case_outcomes.csv` from each replay condition. They rehash every table,
recompute the complete per-case route/outcome rows, and record disagreements as
`unstable_cases`; no voting or fallback route is introduced. Supplying
`--replay-evidence` to the certificate makes it verify this artifact against
the consensus condition IDs and source CSV bytes. The current artifact covers
all nine conditions and all 58 cases with zero unstable routes:
`/tmp/target41_replay_evidence_full_paired_v3.json` (SHA-256
`620ccc246710237b4ae26b62afaece81035023401ed189ad515f6501793edf8d`).
When a baseline table is supplied, the artifact also recomputes each
condition's baseline-to-candidate gains/losses and event-match changes; the
current nine conditions all reproduce 38→43, five gains, zero losses.
The certificate also recomputes the exact two-sided paired sign-test diagnostic
for the discordant cases: `p=0.0625` for five gains and zero losses. This is a
development-cohort diagnostic only; it is not calibrated accuracy, proof of
causality, or a production promotion gate.
For this frozen artifact, the machine-readable provenance fields report
`condition_table_sha256_unique_count=1` and
`all_condition_tables_byte_identical=true`: all nine final
`case_outcomes.csv` files are byte-identical (SHA-256
`e4033acf3e6ba49b46e33566e25edb1c603e8370ccf0e93a6633f0fb22d50a80`). This
therefore proves final-decision-table invariance only. The fact that the
upstream mask perturbations were actually applied is supported separately by
their raw audits (confidence measurements 441/437/437; image-size
measurements 431/437/439; JPEG measurements 440/440); the certificate does
not infer that fact from identical decision tables.
The paired certificate carrying this diagnostic is
`/tmp/target41_robustness_certificate_replay_paired_v7.json` (SHA-256
`0a39af1a6511fd14b80f1a21cb750c8d119c0044e7a562f9d40274bca15b7ce7`).
Build it by repeating caller-supplied condition/table pairs; the command
requires at least two conditions and refuses to overwrite its output:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_target41_replay_evidence.py \
  --condition base=/path/to/base/case_outcomes.csv \
  --condition scale640=/path/to/scale640/case_outcomes.csv \
  --baseline /path/to/baseline/case_outcomes.csv \
  --denominator 58 \
  --output /tmp/target41_replay_evidence.json
```

An optional `--source-case-bootstrap` input adds the already-computed grouped
sensitivity artifact. The current artifact covers 54 mapped clips in 50 exact
source-lineage groups: candidate-minus-baseline is +9.26 percentage points,
with a paired percentile-bootstrap interval of +1.92 to +17.54 points. The
certificate validates its count/rate arithmetic, finite interval bounds,
replicate metadata and SHA, but labels these groups as source recordings—not
verified cameras—and does not interpret the bootstrap probability as
confirmatory significance. Without this option, the source-case section remains
explicitly unavailable.

An optional `--temporal-offset-summary` input separately records the fixed
single-observation offset screen. The current artifact reports 40/58, 41/58,
43/58, 42/58 and 40/58 for offsets `-2,-1,0,+1,+2`, respectively, with no
correctness losses in that positive development cohort. Only `-1, 0, +1`
reach the 41/58 point target. This is missing-observation sensitivity, not
evidence that the chosen `{0,+1}` window is camera-invariant or production-safe.

Build it from the immutable ledger and consensus artifacts:

```bash
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_target41_robustness_certificate.py \
  --ledger /tmp/target41_evidence_ledger.json \
  --consensus /tmp/target41_mask_decision_consensus.json \
  --replay-evidence /tmp/target41_replay_evidence_full_paired_v3.json \
  --source-case-bootstrap /tmp/target41_source_case_paired_bootstrap.json \
  --temporal-offset-summary /tmp/target41_mask_single_offsets__it1z8re/summary.json \
  --target-count 41 \
  --output /tmp/target41_robustness_certificate.json
```

The command refuses to overwrite an existing output and fails closed on a
changed SHA-256, denominator/case mismatch, duplicate replay or case ID,
inconsistent gain/loss arithmetic, or unstable replay. `target_point_estimate_reached`
may be true while `camera_independence.verified` and
`promotion.ready_for_production` remain false. A human-reviewed, camera/site /
source-recording-disjoint holdout and a reviewed negative set are still required
before any release claim.

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

`SMART_BACKTRACK_MASK_DIAGNOSTICS=1` may additionally place compact
`mask_litter_diagnostics` on resolver-input frame rows that contain raw litter
candidates. The producer accepts only real observed `seg_track`/`seg_predict`
vehicle or scooter masks. It serializes bounded scalar overlap, containment,
mask-fill and signed-distance summaries, never polygons or dense masks. The
resolver ignores the field, so it cannot alter event confirmation, costs,
routes or NULL. This describes visible-silhouette evidence only; it is neither
ordinal depth nor a 3D coordinate.

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
