# Production vehicle hard gates

- Date: 2026-08-27
- Author: Codex
- Branch: `heetah-dev`
- Commit: included by the commit containing this note
- Scope: Smart Backtrack physical association gates; litter confirmation and
  detector outputs are unchanged.

## Production change

Direct litter-to-vehicle association now uses the original vehicle bbox with
no outward expansion. For release point `p`, bbox `B`, width `w`, and height
`h`,

```text
D = dist(p, B) / sqrt(w^2 + h^2)
valid_distance = (D <= 0.30)

delta_frames = abs(actor_evidence_frame - release_frame)
valid_time = (delta_frames <= 3) AND (delta_frames / FPS <= 0.25 s)
```

The seconds limit is converted with `floor(FPS * 0.25)` before applying the
3-frame cap. This prevents rounding from admitting a duration greater than
0.25 seconds. The production defaults apply to `C_BA`, `C_BC`, and `C_AC` time
alignment; the `D<=0.30` gate is the direct-vehicle `C_BC` gate. Person distance
remains unchanged at `0.85`.

Legacy research replay remains explicit and reproducible with vehicle bbox
expansion `0.18/0.15`, vehicle distance gate `0.8`, and no frame cap.

## Why `D=0.30`

The frozen 2026-08-26 confirmed-event sidecars and current reviewed mapping
provided 40 correct-vehicle candidate rows and 208 distractor rows. Recomputing
distance from the unexpanded bbox gave a maximum correct-candidate distance of
`0.2960167` (case 145). Therefore `0.30` is the smallest tested one-decimal
engineering threshold that retained every observed correct candidate:

| Unexpanded D gate | Correct retained | Distractors retained |
|---:|---:|---:|
| 0.20 | 39/40 = 97.50% | 54/208 = 25.96% |
| 0.30 | 40/40 = 100.00% | 61/208 = 29.33% |
| 0.40 | 40/40 = 100.00% | 66/208 = 31.73% |
| 0.80 | 40/40 = 100.00% | 93/208 = 44.71% |

For `D=0.30`, the 40/40 correct-candidate coverage has Wilson 95% CI
91.24–100.00%; the 61/208 distractor pass rate has Wilson 95% CI
23.56–35.84%. Relative to the legacy expanded-bbox `D=0.8` gate
(96/208 distractors), production removes 35/96 = 36.46% of those distractor
edges while retaining all 40 observed correct candidates.

The unexpanded definition also had a higher event-cluster-bootstrap candidate
AUC than the expanded definition: 0.8960 (95% CI 0.8561–0.9311) versus 0.8767
(0.8413–0.9091); paired delta +0.01935 (0.00049–0.03796). This is exploratory
evidence from the same small domain, not proof of a universal constant.

## Why `3 frames AND 0.25 seconds`

Among the same 40 correct vehicle rows, actor evidence gaps were 38 at zero
frames, one at two frames/0.20 s, and one at three frames/0.25 s. The hybrid
gate retained 40/40. Among 208 distractor rows, the seconds-only gate retained
184, while the hybrid gate retained 182. Thus three frames is the smallest
observed cap that keeps every labeled positive while adding a detector-cadence
constraint. The gain is modest and must not be overstated.

Seconds and frames serve different purposes: seconds are physically comparable
across 10, 12, and 30 FPS clips; frames bound how many sequential detector
opportunities may be missing. At 10 FPS the effective cap is 2 frames, at
12 FPS it is 3, and at 30 FPS it remains 3.

## Vehicle-only replay result

Replay fixed RT-DETR/tracker candidates and rebuilt only attribution routes.
All 58 usable clips were evaluated; unusable cases 24, 68, 92, 111, and 138
were excluded. Cases 74 and 174 were counted correct according to the user's
manual visual adjudication. Person identity was ignored.

| Trial | Correct/58 | Accuracy | Wilson 95% CI |
|---|---:|---:|---:|
| Legacy control: expanded 0.18/0.15, D=0.8, 0.25 s only | 35/58 | 60.34% | 47.49–71.91% |
| New production: no expansion, D=0.3, 3 frames AND 0.25 s | 34/58 | 58.62% | 45.80–70.37% |

Five clip-level correctness states changed: new production gained cases 149
and 164, and lost cases 42, 159, and 168. The two-sided exact McNemar/sign test
on the discordant clips is `p=1.0`; this sample does not demonstrate a
significant accuracy difference. Sensitivity replays at unexpanded `D=0.2`
and `D=0.4` were also 34/58. An expanded-bbox `D=0.3` hybrid-time ablation was
35/58, identical to the legacy control, indicating that the observed route
changes came mainly from removing expansion rather than from tightening D or T.

The production setting is therefore justified as a more selective,
dimensionless physical gate with full observed positive-candidate coverage.
It is not justified as an end-to-end accuracy improvement on this dataset.

## Validation

- `python -m py_compile` passed for costs, resolver, and study modules.
- Hard-gate targeted suite: 49 passed.
- Full `tests/pipeline`: 211 passed, 12 skipped, 2 failed. Both failures are
  pre-existing detect characterization golden-image SHA mismatches; all other
  characterized values match and the affected detector/render code was not
  modified here.

## Limitations and rollback

Only 41 usable clips have confirmed resolver candidate tables; unconfirmed
true-litter clips cannot contribute distractor geometry. Candidate rows within
a clip are correlated, and the parameter was selected on this dataset. A new
camera/FPS/domain requires a locked external validation set before claiming
generalization.

Rollback is the parent commit. For a research-only legacy replay, set the
explicit `StudyConfig` overrides described above; production does not expose a
silent environment switch for these physical safety gates.
