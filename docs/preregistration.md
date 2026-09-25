# Pre-registration: which paper each outcome licenses

Written before any predictions were collected. `results/` was empty, and
`scripts/verify_gates.py` passed, when this file was committed.

Two hypotheses are live, and they predict **different shapes of the same
curve**:

- **N — non-monotonicity.** Safety is not a monotone function of precision.
  The curve rises and falls; some middle rung beats both its neighbours.
- **D — operating point drift.** Lower precision moves where the guard draws
  its line. The guard flags more of everything, so the safety curve climbs
  *smoothly* as bits fall, and the false positive rate climbs with it.

If D is the mechanism, the curve is usually **monotone**, which is evidence
*against* N. They are not two descriptions of one finding, and whichever the
data supports has to be the one chosen in advance.

This file fixes that choice. Nothing below may be edited after the first
prediction file is written; the git history is the proof.

---

## The umbrella claim

Both N and D are instances of one broader claim:

> A fixed-threshold safety metric is not a reliable function of model
> precision. It moves for reasons that are not changes in the guard's ability
> to tell harmful from harmless.

This is falsifiable, and the outcome that falsifies it is concrete: Gate C
returning `H3_REJECTED` together with Gate D returning `RESIDUAL_GAP` would
mean quantization genuinely degrades the guard and the metrics correctly
report it. That is the honest alternative, and it is the third paper below.

---

## The decision table

Gate C is the hinge. Read it first, then Gate A, then Gate D.

| Gate C | Gate A | Paper | Headline |
|---|---|---|---|
| `H3_CONFIRMED` | `PATTERN_SURVIVES` | **1 — Non-monotonic, and artefactual** | Safety is non-monotonic in precision, and the reversal is an operating point artefact rather than a capability change |
| `H3_CONFIRMED` | `THRESHOLD_SHIFT` | **2 — Not Safer, Just Louder** | Quantization moves the operating point; fixed-threshold safety metrics change while discrimination does not |
| `H3_CONFIRMED` | `MONOTONIC` | **2b — as above, asymmetric** | Same as 2, but detection and false alarms do not move together; report the asymmetry rather than claiming a clean slide |
| `H3_CONFIRMED` | `NO_SIGNIFICANT_DIFFERENCES` | **4 — Null result** | Quantization to 3 bits does not measurably change guard behaviour; a deployment finding, and a real one |
| `H3_REJECTED` | any | **3 — Capability degradation** | Discrimination genuinely degrades below N bits; locate the breakpoint on the ladder and report it |
| `UNDERPOWERED` | any | *No paper yet* | Scale to Tier A and re-gate. This is the only condition authorising data scaling |

Gate D refines whichever paper is selected, and never changes which one:

| Gate D | Adds |
|---|---|
| `REPAIRED` | The fix is free: recalibrate at matched FPR. No retraining, no extra memory. This is the deployment recommendation. |
| `RESIDUAL_GAP` | A difference survives recalibration. That residual — not the raw gap — becomes the target of the mechanism phases (6 and 7). |

Families are evaluated separately and each gate reports `replicated`. If the
two architectures disagree, that is stated in the abstract, not averaged away.

---

## Commitments

1. **Paper 2 is the prior.** Drift predicts a monotone curve, so `THRESHOLD_SHIFT`
   is the most likely outcome. Paper 1 is the stronger result and the less
   likely one. Neither expectation changes how any gate is read.

2. **A reversal counts only if it is significant.** Non-monotonicity is
   licensed by `PATTERN_SURVIVES`, which requires a reversal in the safety
   curve whose paired McNemar test on the harmful prompts survives Holm
   correction. A curve that merely fails to be strictly monotonic is not a
   finding — `verify_gates.py` case 6 produces two or three such wiggles per
   run from sampling noise alone.

3. **Saturation is reported, not interpreted.** If the safety rate sits near
   its ceiling at every rung, no reversal can reach significance whatever the
   models are doing. In that case the result is "this prompt set cannot answer
   the question", not "the pattern is absent", and the finding is a harder
   prompt set — not a weaker claim.

4. **Paper 4 gets written.** A null result is an outcome, not a failure, and
   "quantize the guard freely" is useful to a practitioner.

5. **No claim is written that `claims_to_evidence.csv` has not marked
   `LICENSED`.**
