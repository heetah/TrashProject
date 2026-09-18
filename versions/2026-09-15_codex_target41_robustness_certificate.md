# 2026-09-15 — Codex target-41 robustness certificate

- Author: Codex
- Branch: `heetah-dev`
- Commit: working tree (no commit/push/deploy performed)
- Scope: research/reporting only; no production caller

## Purpose

The target was adjusted to 41 correct attributions out of the fixed 58-case
reviewed development cohort. A point estimate alone is not enough: every case
needs an explanation, replay perturbations must be reproducible, and the
remaining camera/site generalization requirement must stay visible.

## Change

Added `scripts/pipeline/backtrack/target41_robustness.py` and
`scripts/build_target41_robustness_certificate.py` for the strict certificate,
plus `scripts/pipeline/backtrack/replay_evidence.py` and
`scripts/build_target41_replay_evidence.py` for source-linked, per-case replay
verification. The nine-condition artifact is
`/tmp/target41_replay_evidence_full_paired_v3.json` (SHA-256
`620ccc246710237b4ae26b62afaece81035023401ed189ad515f6501793edf8d`); it
also recomputes baseline-to-candidate gains/losses for each condition. The
certificate rebuilt with this paired evidence is
`/tmp/target41_robustness_certificate_replay_paired_v7.json` (SHA-256
`0a39af1a6511fd14b80f1a21cb750c8d119c0044e7a562f9d40274bca15b7ce7`).

- validates the target-41 evidence ledger and its paired gain/loss arithmetic;
- validates the closed six-code explanation vocabulary and emits deterministic
  case-ID lists for each explanation, so every aggregate reason is directly
  traceable to the fixed clip rows;
- verifies each explanation code agrees with the candidate outcome semantics;
- emits one deterministic case-explanation row per clip with baseline/candidate
  outcomes, route tuples, and the case-table observables needed for direct
  reviewer audit;
- verifies the consensus artifact's SHA-256, case coverage, unique replay IDs,
  and absence of unstable cases;
- optionally rehashes and recomputes every per-case route/outcome row from all
  replay condition tables through `replay_evidence.py`, preventing a summary
  artifact from asserting consensus after its source CSV changes;
- records the important provenance boundary
  (`condition_table_sha256_unique_count=1`,
  `all_condition_tables_byte_identical=true`): all nine final
  `case_outcomes.csv` files are byte-identical (SHA-256
  `e4033acf3e6ba49b46e33566e25edb1c603e8370ccf0e93a6633f0fb22d50a80`): the
  certificate proves decision-table invariance, while upstream raw mask audits
  (441/437/437 confidence, 431/437/439 image-size, 440/440 JPEG measurements)
  are the separate evidence that perturbations were applied;
- rejects normalized duplicate condition IDs and invalid table paths before
  any replay file is read;
- reports the 43/58 development point estimate, explanation-code coverage and
  counts, and nine replay conditions with 58/58 consistent route tuples;
- recomputes the exact two-sided paired sign-test diagnostic (`p=0.0625` for
  five gains and zero losses) while explicitly labelling it non-promotional;
- explicitly records `camera_independence.verified=false` and
  `promotion.ready_for_production=false` until an independently reviewed,
  camera/site/source-recording-disjoint holdout and negative set exist.

The CLI also accepts the existing
`/tmp/target41_source_case_paired_bootstrap.json`. When supplied, the
certificate verifies the 54 mapped clips/50 source-lineage groups, exact rate
arithmetic, finite bootstrap intervals, replicate metadata and artifact SHA.
It reports this as source-recording sensitivity only; it never upgrades those
groups to camera IDs or confirmatory evidence.

It can also ingest `/tmp/target41_mask_single_offsets__it1z8re/summary.json`.
The five offset rows are checked independently: candidate results are 40/58,
41/58, 43/58, 42/58 and 40/58 for `-2,-1,0,+1,+2`; all have zero correctness
losses, but only `-1,0,+1` reach the point target. This keeps temporal
missing-observation sensitivity separate from the nine-condition route
consensus.

Earlier intermediate certificates are retained for provenance:
`/tmp/target41_robustness_certificate_complete.json`,
`/tmp/target41_robustness_certificate_explainability_v2.json` (SHA-256
`915b682ddb0eed26182dace5110451a1ea65a1d1679b9e6a56c9fd707f932b0a`), and
`/tmp/target41_robustness_certificate_replay_explainable.json` (SHA-256
`e3609ac7037812db83daa06d6944c0c1cb4f6c35957c19d8ed9e49eac73ad4bf`). They
are superseded by the paired replay certificate above, which additionally
verifies baseline-to-candidate metrics for every condition and records whether
the final condition tables are byte-identical.

No resolver, trajectory, cost, route schema, configuration, model, or UI
behavior changed. The module has no production caller and does not infer truth,
select a route, multiply correlated observations, or call the camera/depth
runtime.

## Frozen development result

- baseline: 38/58;
- candidate: 43/58 (74.14%), target 41/58 reached as a point estimate;
- paired changes: five gains, zero losses;
- exact paired two-sided sign-test diagnostic: `p=0.0625` (development cohort,
  not a production promotion gate);
- explanations cover all 58 cases;
- confidence/scale/compression replay consensus: 9 conditions, 58/58 cases,
  zero unstable cases;
- source-lineage grouped sensitivity: 54 mapped clips in 50 groups, observed
  candidate-minus-baseline +9.26 percentage points (bootstrap 95% interval
  +1.92 to +17.54 points);
- temporal offset sensitivity: 40/58 (`-2`), 41/58 (`-1`), 43/58 (`0`),
  42/58 (`+1`), 40/58 (`+2`); no correctness losses, but target reached only
  at `-1,0,+1`;
- camera-independent evidence: not verified.

Replay stability is controlled perturbation evidence on the same development
cohort. It is not calibrated accuracy, a confidence interval, a negative-set
precision estimate, or cross-camera generalization.

## Validation

The validation run produced:

- focused replay-evidence + robustness + camera worksheet tests: **73 passed**;
- `tests/pipeline`: **521 passed, 12 skipped**;
- `tests`: **548 passed, 12 skipped**;
- replay-evidence, replay CLI, robustness certificate and certificate CLI all
  passed `py_compile`;
- an independent CSV recomputation verified all 58 case IDs, route consensus,
  and the 38→43 / five-gain / zero-loss metrics for each of the nine
  conditions;
- the camera-safe holdout preflight verified the expected 18-case membership
  and all 18 source SHA-256 values, then correctly returned not-ready because
  every review row is still `unreviewed`;
- the blind holdout runtime output preflight found 18/18 analysis JSON files,
  sidecars, and H.264 MP4s with readable video streams; this is only a
  container/output smoke check, not an accuracy or attribution result;
- `git diff --check`: passed.

The commands used were:

```bash
PYTHONPATH=scripts conda run -n rtdetr python -m pytest -q \
  tests/pipeline/test_target41_robustness.py \
  tests/pipeline/test_target41_replay_evidence.py \
  tests/pipeline/test_camera_holdout_review.py
PYTHONPATH=scripts conda run -n rtdetr python -m pytest -q tests/pipeline
PYTHONPATH=scripts conda run -n rtdetr python -m pytest -q tests
conda run -n rtdetr python -m py_compile \
  scripts/pipeline/backtrack/replay_evidence.py \
  scripts/build_target41_replay_evidence.py \
  scripts/pipeline/backtrack/target41_robustness.py \
  scripts/build_target41_robustness_certificate.py
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_target41_replay_evidence.py \
  --condition conf005=/tmp/target41_mask_confidence_stability/conf_0.05/candidate_eval/case_outcomes.csv \
  --condition conf010=/tmp/target41_mask_confidence_stability/conf_0.10/candidate_eval/case_outcomes.csv \
  --condition conf020=/tmp/target41_mask_confidence_stability/conf_0.20/candidate_eval/case_outcomes.csv \
  --condition scale512=/tmp/target41_mask_imgsz_stability/imgsz_512/candidate_eval/case_outcomes.csv \
  --condition scale640=/tmp/target41_mask_imgsz_stability/imgsz_640/candidate_eval/case_outcomes.csv \
  --condition scale768=/tmp/target41_mask_imgsz_stability/imgsz_768/candidate_eval/case_outcomes.csv \
  --condition jpeg90=/tmp/target41_mask_compression_stability/quality_90/candidate_eval/case_outcomes.csv \
  --condition jpeg50=/tmp/target41_mask_compression_stability/quality_50/candidate_eval/case_outcomes.csv \
  --condition base=/tmp/target41_mask_temporal_aggregate/+0_+1/candidate_eval/case_outcomes.csv \
  --baseline /tmp/target41_mask_temporal_aggregate/+0_+1/baseline_eval/case_outcomes.csv \
  --denominator 58 \
  --output /tmp/target41_replay_evidence_full_paired_v3.json
PYTHONPATH=scripts conda run -n rtdetr python scripts/build_target41_robustness_certificate.py \
  --ledger /tmp/target41_evidence_ledger.json \
  --consensus /tmp/target41_mask_decision_consensus.json \
  --replay-evidence /tmp/target41_replay_evidence_full_paired_v3.json \
  --source-case-bootstrap /tmp/target41_source_case_paired_bootstrap.json \
  --temporal-offset-summary /tmp/target41_mask_single_offsets__it1z8re/summary.json \
  --target-count 41 \
  --output /tmp/target41_robustness_certificate_replay_paired_v7.json
git diff --check
```

## Limitations and next gate

The 18-case camera-safe worksheet is still unreviewed. One reviewer must add
truth and camera/site/session/source-recording provenance; only then can a
camera-disjoint paired evaluation be frozen. The certificate intentionally
does not declare the 85% release criterion met.

## Rollback

Remove the research-only module, CLI, focused tests, README section and this
version note, then retain the existing ledger/consensus artifacts. Production
rollback is unnecessary because there is no runtime caller.
