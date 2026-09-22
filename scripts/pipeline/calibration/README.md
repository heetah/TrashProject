# Online pseudo-homography calibration

This subsystem treats moving vehicles as long-term traffic-geometry
observations, never as static point correspondences between adjacent frames.
It is intentionally phased so incomplete calibration cannot silently change
attribution evidence.

## Implemented: Phase 1 through Phase 8

```text
YOLO-Seg vehicle/scooter polygon
  -> robust bottom-band ground point
  -> confidence/area/border/continuity/teleport filtering
  -> rolling per-camera lightweight track buffer
  -> local motion segments and tangent vectors
  -> robust grid motion field
  -> spatial + direction + track-continuity flow clustering
  -> deterministic relative-scale H0
  -> safe image_to_ground and complete-image validation
  -> robust scale-resistant motion/speed/curvature/direction losses
  -> cross-vehicle local lane/flow consistency loss
  -> bounded candidate search, confidence and validated safe-update proposal
  -> persistent state, locking, drift monitoring and rollback
  -> optional static-background stabilization and complete debug view
```

`extract_vehicle_ground_point(mask)` accepts dense masks or Ultralytics polygon
coordinates. It takes the median of pixels at or below the configurable
bottom percentile instead of trusting one lowest segmentation pixel.

`VehicleTrajectoryCollector` accepts only real detector observations. Cached
skip-frame boxes are recorded as rejected evidence and do not create zero-speed
updates. The rolling `CalibrationBuffer` stores only frame/time, track identity,
image point, velocity and quality metadata. It never stores a video frame or
full mask. Local directions are calculated segment by segment; no whole-track
straight-line assumption is made, so legitimate curved roads remain curved.

Enable collection with `DYNAMIC_HOMOGRAPHY=1`. The analysis JSON then contains
`run.homography_calibration` with accepted/rejected counts and local-motion
coverage. These are observation diagnostics, not calibration accuracy.

`TrafficMotionField` uses a circular medoid and trims the largest angular
residuals before calculating its dominant vector. Cell speed is a
quality-weighted median. A cell's confidence combines support, observation
quality and circular concentration; it is not an accuracy probability.

`cluster_traffic_flows` is a deterministic, grid-indexed DBSCAN variant. Two
segments from different tracks must be close in normalized image position and
direction. Consecutive segments of the same track may connect a longer curved
path under a separate continuity radius/angle. A cluster also requires multiple
independent track IDs, so one long noisy vehicle cannot establish a traffic
flow alone. This remains lightweight and adds no clustering dependency.

`render_motion_debug(frame, observations, field, clusters)` returns an
annotated copy containing local segments, per-cell arrows and cluster colors.
It never mutates the input or runs in the normal renderer.

Phase 3 starts with the non-random baseline
`H0=diag(1/width,1/height,1)`. It preserves image orientation and maps the
image rectangle to relative `[0,1]x[0,1]` coordinates. It is deliberately not
described as metric or perspective-correct calibration. Candidate optimization
in later phases must improve it using traffic evidence.

`image_to_ground(points, H)` accepts `(N,2)` coordinates. Rows with NaN/Inf,
near-zero homogeneous denominator or excessive projected magnitude become NaN
instead of poisoning downstream costs. `validate_homography` additionally
checks projective-scale normalization, determinant, condition number, a full
image-grid denominator range/sign, projected corner orientation, area and
bounds. `initial_homography_snapshot` records matrix source, version 0,
confidence 0, relative-scale-only status and the complete validation report.

Phase 4 evaluates a safe candidate matrix without activating it. For projected
points `q_i` at time `t_i`, segment velocity is
`v_i=(q_i-q_(i-1))/(t_i-t_(i-1))`. Motion residual is
`||v_i-v_(i-1)||/median_track_speed`; speed residual is
`|log((||v_i||+eps)/(||v_(i-1)||+eps))|`. Their scale-free forms prevent a
candidate from winning merely by shrinking pseudo-ground coordinates.

For each pair of consecutive segments, signed turning angle is computed with
`atan2(cross, dot)`. Curvature loss applies Huber to the wrapped difference
between consecutive turning angles, so a smooth curve is allowed while an
abrupt kink is penalized. Local direction loss projects a short probe at each
motion observation and robustly measures angular residual inside its Phase-2
traffic-flow cluster. This is a finite-difference approximation to the local
projective Jacobian, not a global straight-road assumption.

All residuals use quality-weighted means and configurable Huber penalties. The
reported total divides the weighted component sum by the weights that actually
have evidence, and every component includes its sample count. Default equal
weights are a neutral research starting point; the 0.50, 0.35 and 15-degree
Huber transitions are explicit initial robustness priors, not learned or
validated constants. They require event/camera-grouped sensitivity analysis
before promotion.

Vanishing-point loss is intentionally inactive. Curved roads and intersections
must not be forced to share a VP; it can be added later only behind reliable
straight-local-segment RANSAC evidence and independent replay validation.

Phase 5 adds a cross-vehicle local centerline diagnostic. For every observation
in a flow cluster, it selects nearby observations from other track IDs using a
fixed image-space grid. Their quality-weighted coordinate median and robust
projected tangent form a leave-one-track-out moving centerline. Only the
perpendicular distance to that tangent is penalized, so longitudinal timing
differences and legitimate curved paths are not forced onto one global line.
The residual is divided by the square root of the projected image footprint,
making it invariant to uniform pseudo-ground scale. Grid lookup keeps typical
work near `O(N*k)` instead of an all-pairs `O(N^2)` scan.

Without lane labels, a flow cluster can still contain adjacent physical lanes.
Accordingly this term is a relative geometric diagnostic, not a lane-width
measurement and not sufficient to optimize H by itself. Candidate optimization
must combine it with motion, direction and future perspective constraints,
then pass grouped replay and safety validation.

Phase 6 parameterizes each candidate as `H_candidate=P(theta)H_previous` in
normalized pseudo-ground coordinates. The five searched dimensions are log
anisotropy, x/y shear and x/y projective denominator terms. Translation and
rotation are intentionally omitted because the current motion-family losses do
not identify them. Deterministic coordinate search evaluates plus/minus bounded
steps and decays the step each iteration; it is not an unconstrained 9-element
matrix optimizer.

The optimizer objective is a configurable weighted mean of:

1. Phase-4/5 traffic-data loss;
2. robust local-Jacobian anisotropy and projected-area variation;
3. robust image-grid projection displacement from the previous stable H.

Every candidate first passes Phase-3 hard validation. The perspective term is
a safety regularizer, not a claim that real roads are conformal. The temporal
term compares projected geometry after H normalization, avoiding raw-matrix
scale ambiguity.

Confidence reports ten inspectable components: valid-track support, independent
flow support, spatial coverage, duration, observation quality, motion-field
stability, residual quality, lane evidence, candidate improvement and historical
stability. The mean is never the only update condition. Separate hard gates for
track count, flow count, coverage, duration, relative improvement and freeze
confidence must all pass.

For a passing candidate, `alpha=max_alpha*confidence*improvement_score` and the
selected theta is scaled by alpha before recomposition. The proposal is then
validated and re-scored; a non-improving or invalid proposal freezes the old H.
`DYNAMIC_HOMOGRAPHY_OPTIMIZE=0` remains the default. Even when enabled, this
phase returns `applied_to_production=false`; it has no persistent state and does
not replace the backtrack transform.

Phase 7 provides `DynamicHomographyCalibrator`, which owns one camera's
collector, current relative H, version, confidence, rollback snapshots and a
bounded metrics history. Its states are:

```text
UNCALIBRATED -> COLLECTING -> ESTIMATING -> WARMING_UP -> LOCKED
                            -> LOW_CONFIDENCE -> ESTIMATING
                                            LOCKED -> DRIFT_DETECTED -> RECALIBRATING
```

Only a validated Phase-6 proposal increments the H version. WARMING_UP requires
both successful updates and subsequent stable/no-improvement windows before it
can lock. LOCKED evaluates at a configurable interval but does not optimize;
it updates a slow residual baseline only during non-drift windows.
An insufficient-evidence evaluation enters LOW_CONFIDENCE, freezes H, and may
estimate again only after the collector receives more observations.

`detect_calibration_drift` reports separate residual, normalized motion
centroid, spread, direction-tensor and optional background-motion signals. A
single signal cannot change state: it must persist for the configured number
of evaluation windows. A changed image shape is treated separately because the
old coordinate system and buffered points cannot safely be mixed; calibration
freezes until an explicit reset starts collection for the new shape.

Every successful state-H update first stores the old matrix/version/confidence.
`rollback()` restores the last snapshot. `reset()` clears H and lightweight
evidence. Per-window metrics include all fields requested by the calibration
specification and are bounded in memory.

Phase 8 optionally estimates fast camera motion from static background only.
`StaticBackgroundStabilizer.update` refuses to run without an explicit
dynamic-object exclusion mask. `build_dynamic_exclusion_mask` can combine dense
masks, polygons and boxes for known vehicles, people and litter. ORB matches
remaining background features to one fixed reference feature set; ratio-test
matches feed a RANSAC partial-affine current-to-reference transform. Inlier,
translation, rotation, scale and orientation checks reject unsafe estimates.

Only reference keypoints/descriptors and the last valid transform are retained,
not raw-frame history. A transient failure freezes the last valid G. Runtime
composition is `H_runtime=H_calibration@G_current_to_reference`. Both features
are disabled by default and isolated from attribution.

`capture_event_snapshot(timestamp)` returns read-only calibration H, G, runtime
H, version and confidence arrays. Smart Backtrack can consume this snapshot
behind `SMART_BACKTRACK_DYNAMIC_HOMOGRAPHY=1`; it still requires `LOCKED`, the
configured minimum confidence and a fresh geometry validation, otherwise the
event uses image-space costs. `render_calibration_debug` returns
a frame copy containing image tracks/ground points, flow graphics, a
pseudo-ground track inset and calibration statistics; it never enters the
normal renderer.

## Safety boundary and remaining phases

Phase 1 through 8 plus the event consumer expose an opt-in attribution path;
safe defaults keep it disabled. The 58 short-clip replay exercised fallback
only because no isolated clip reached `LOCKED`, so it proves integration safety
but not a real-world accuracy gain. The following remain required before
default promotion:

1. reviewed continuous multi-camera replay and threshold calibration;
2. a paired accuracy comparison where eligible events actually apply H;
3. optional metric scale from external calibration metadata.

The legacy `LITTER_BEV_STABLE` experiment is separate and must not be described
as the new online self-calibrator.
