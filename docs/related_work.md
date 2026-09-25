# Related work, and where this project is different

Literature scan of 2026-09-26. Its job is to answer one question honestly:
**is the contribution still there?** The short answer is yes, but not where
the research plan originally put it.

> **The headline finding of this scan.** "Quantization can improve safety,
> sometimes non-monotonically" is **already published**, more than once. The
> observation is not ours to claim. What is unoccupied is the *measurement* —
> nobody quantizes the guard model itself and evaluates it threshold-free, and
> nobody separates the two mechanisms. Reposition the thesis accordingly:
> not *"we found an effect"* but *"the field has been measuring this with an
> instrument that cannot distinguish two different things, here is one that
> can, and here is what it shows."*

## How to read the status column

Sources were found by search; depth of checking varies and is recorded per
entry. Do not cite anything marked `SNIPPET` without reading it first.

| Status | Means |
|---|---|
| `READ` | Full text or HTML fetched and checked against our claims |
| `ABSTRACT` | Abstract verified directly |
| `SNIPPET` | Search-result summary only — **unverified, must be read** |

---

## 1. Quantization and safety of the *generated* model

The crowded area, and the one that takes our original framing.

| Work | Idea | Status |
|---|---|---|
| [Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment](https://arxiv.org/html/2606.29581v1) | Factorial study. States directly that quantization's effect on refusal "is not always monotonic — compressing a model can either weaken or unintentionally strengthen its tendency to refuse," and that moderate 4-bit sometimes preserves or improves trustworthiness. Reports INT4 keeping or lowering attack success for 7 of 9 models. | `SNIPPET` |
| [Q-resafe](https://arxiv.org/pdf/2506.20251) | Safety risks of quantized LLMs plus quantization-aware safety patching. | `SNIPPET` |
| [Quantization Undoes Alignment](https://arxiv.org/html/2605.15208v1) | Bias emergence in compressed LLMs across models and precision levels. | `SNIPPET` |
| [QuantiBias](https://arxiv.org/html/2607.21063v1) | Quantization-induced bias that standard safety evaluation misses; short-form safeguards stay flat while open-ended generation degrades. | `SNIPPET` |
| [Alignment-Aware Quantization for LLM Safety](https://www.arxiv.org/pdf/2511.07842) | Contrastive alignment loss during PTQ to keep the quantized model aligned. | `SNIPPET` |
| [Preserving Fairness and Safety via Critical Weight Protection](https://arxiv.org/html/2601.12033v2) | Sensitivity-scores weights, keeps the safety-critical ones at higher precision. | `SNIPPET` |
| [Effect of Quantization on Clinical Benchmarks](https://arxiv.org/html/2609.22216) | Accuracy and safety across model families in a high-stakes domain. | `SNIPPET` |

**Where we differ.** Every one of these quantizes the **generator** and
measures its refusal behaviour, attack success rate or bias. None quantizes
the **guardrail classifier** that sits in front of a generator. That is a
different system with a different failure mode: a generator's refusal is a
behaviour, a guard's verdict is a thresholded score, and only the second can
drift without changing.

**Two that need reading before this section is final:**

- [Silent Alarm: A J-Space Protocol for Comparing Danger Recognition Across
  Models and Quantization Levels](https://arxiv.org/pdf/2607.12792) —
  the title is close enough to our question to matter. `SNIPPET`, **read first.**
- [LiteLMGuard](https://arxiv.org/pdf/2505.05619) — on-device prompt filtering
  against quantization-induced risks. Adjacent: it *defends* against
  quantization damage rather than measuring the guard's own drift. `SNIPPET`

**Critical-weight protection is our Phase 7.** The fairness/safety
critical-weight paper is doing, for the generator, what
`build_mixed_precision.py` does for the guard. Cite it as prior art for the
method and be explicit that the target differs.

---

## 2. Quantization and calibration

| Work | Idea | Status |
|---|---|---|
| [When Quantization Affects Confidence of LLMs](https://arxiv.org/abs/2405.00632) | GPTQ-4bit lowers confidence in true labels; the effect varies by model and scale; quantization loss concentrates on samples where the full model was *already* low-confidence. | `ABSTRACT` |

**Where we differ.** This is the nearest miss on the calibration axis, and it
stops one step short of our argument. It measures confidence and calibration
and leaves it there. It does not ask whether the confidence change moves any
*decision* — and the answer, for a fixed threshold, is that it cannot:

```
sigmoid(m / T) >= 0.5   <=>   m >= 0,   for every T > 0
```

Our `calibration_decomposition` fits `sigmoid(a*m + b)` and splits the result
into temperature (`1/a`, which drives ECE) and boundary location (`-b/a`,
which is the only part that moves a decision). No source found does this
decomposition, and its consequence — that "quantization hurt calibration"
cannot by itself explain a safety-rate change — appears to be ours.

Their finding is also a *prediction* we can test: if quantization loss
concentrates on low-confidence samples, our flip rate should rise sharply as
the margin approaches zero. That is exactly what `flip_rate_by_distance`
measures. Cite it as a hypothesis we confirm or refute in a new domain.

---

## 3. Quantization and decision geometry

| Work | Idea | Status |
|---|---|---|
| [Boundary-Aware Quantization: Finite-Scale Decision Geometry of Neural Classifiers](https://arxiv.org/html/2607.01478v1) | Quantization-induced flips concentrate near the full-precision decision boundary — 0.034 overall vs 0.169 inside the low-margin band at 6 bits. Proposes boundary-preservation as a model-selection criterion. Concludes that "quantization can preserve or improve accuracy while changing the reference decision geometry." | `READ` |

**Where we differ — and why this one helps us.** Read the title and it looks
like a collision with `flip_rate_by_distance`. It is not. It studies
one-hidden-layer MLPs, small CNNs and reduced ResNets on digits, MNIST,
Fashion-MNIST and CIFAR-10. It uses geometric diagnostics (Jaccard distance on
boundary masks, local displacement, junction stability) — **no AUROC, no
discrimination/operating-point separation, no calibration decomposition, no
language models, no safety.**

It is the strongest **supporting** citation in this document. An independent
group, in a completely different domain, found that quantization moves the
decision boundary while leaving or improving accuracy. That is our mechanism,
replicated in advance, on toy vision models. Use it in the introduction to
establish that the phenomenon is general and that we are testing it where it
has consequences.

---

## 4. Guardrail evaluation and operating points

| Work | Idea | Status |
|---|---|---|
| [Reasoning's Razor](https://arxiv.org/pdf/2510.21049) | Reasoning improves accuracy but can hurt recall at critical operating points in safety and hallucination detection. | `SNIPPET` |
| [Artificial Analysis guardrail benchmark](https://artificialanalysis.ai/articles/guardrail-safety-benchmark) | Industry benchmark. Reports that some reasoning guards have verdict-token probabilities "almost perfectly polarized, with 99.8% of scores pinned at zero or one, leaving no range to tune a threshold," and that such guards retain ~10% recall at a 1% FPR budget. | `SNIPPET` |
| [Safety Under Scaffolding](https://arxiv.org/pdf/2603.10044) | Measurement artifacts can produce apparent large safety differences that turn out to be artifactual; safety scores are sensitive to how answers are extracted. | `SNIPPET` |
| [When Benchmarks Lie](https://arxiv.org/pdf/2602.14161) | Malicious-prompt classifiers under true distribution shift. | `SNIPPET` |
| [CPU-deployable safety classification](https://arxiv.org/pdf/2608.21570) | Argues the opposite methodological choice: a *fixed* decision threshold for cross-model comparison, explicitly "forbidding the practice of tuning each model to its own best operating point." | `SNIPPET` |

**Where we differ.** *Reasoning's Razor* is the closest methodological
relative — operating-point-aware evaluation of safety detectors — but its
independent variable is reasoning, not precision. Nobody here varies precision.

**The last row is the objection we must answer in the paper.** There is a
real argument that a fixed threshold is the *correct* comparison, because
per-model threshold tuning flatters whichever model you tuned hardest. Our
answer is not that fixed thresholds are wrong; it is that a fixed-threshold
number is a *joint* measurement of discrimination and operating point, and
reporting it alone makes the two indistinguishable. We report both. Write this
as an explicit paragraph, not a footnote.

**`Safety Under Scaffolding` is our closest ally in framing** — same shape of
argument (a measured safety difference turns out to be a measurement artifact),
different cause.

---

## 5. Quantized guard models as deployed artifacts

| Work | Idea | Status |
|---|---|---|
| [Llama Guard 3-8B model card](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | Meta ships an int8 version: ~40% smaller, "negligible impact on the performance of the model," F1 0.936–0.939. | `SNIPPET` |
| [Llama Guard 3-1B-INT4](https://arxiv.org/pdf/2411.17713) | Pruning plus 4-bit for mobile deployment; reports improved safety *and* efficiency. | `SNIPPET` |
| [Qwen3Guard Technical Report](https://arxiv.org/html/2510.14276v1) | The second family in our ladder. Three sizes, Gen and Stream variants. | `SNIPPET` |

**Where we differ, and this is the paper's opening paragraph.** The canonical
claim that quantizing a guard is safe — Meta's own, for the exact model at the
top of our ladder — rests on **F1 at a fixed threshold**. That is precisely the
statistic that cannot distinguish "discrimination is preserved" from "the
operating point moved and the errors happened to cancel." We are not
contradicting the claim. We are showing it was never tested, and supplying the
test. Two int8/int4 points also say nothing about the shape of the curve
between 3 and 16 bits.

---

## 6. What is unoccupied

Ranked by how much weight each can bear.

1. **A fine-grained bit ladder on the guard model itself, evaluated
   threshold-free.** Published work stops at one or two quantization points
   and reports fixed-threshold F1 or ASR. Nothing found runs FP16 → Q3 on a
   guard and reports AUROC, AUPRC, TPR at matched FPR and paired DeLong tests.
   **This is the load-bearing contribution.**

2. **The temperature / boundary-location decomposition, with the inertness
   argument.** Nothing found separates the two, and nothing found states that
   temperature cannot move a fixed-threshold decision. Small, provable,
   and it eliminates a hypothesis the literature treats as live.

3. **Equivalence testing rather than non-significance.** Everything in
   section 1 argues "quantization does / does not hurt safety" from
   significance tests. Confirming that a gap is *small* needs TOST, and a
   study that confirms its own hypothesis by having a weak test has shown
   nothing. Our Gate C cannot be passed that way, and `verify_gates.py`
   demonstrates it against data of known construction.

4. **Pre-registered, demonstrably failable gates.** Unusual in this area.
   `scripts/verify_gates.py` asserts each verdict against synthetic ladders
   built to be a pure threshold slide, a real capability loss, a genuine
   improvement, an underpowered sample, a noise wobble and two families of
   unequal skill.

5. **The repair.** If Gate D returns `REPAIRED`, recalibration at matched FPR
   removes the effect at zero cost — no retraining, no extra memory. A
   practitioner recommendation that follows from the diagnosis.

6. **Two architectures, replication reported rather than averaged.** Every
   analysis is computed within family; `replicated` is a reported field.

---

## 7. What would collapse the novelty

Stated plainly so it can be checked rather than hoped about.

- **A paper that runs a bit ladder on a guard model and reports AUROC.** Not
  found, but absence of evidence after one afternoon of search is weak
  evidence of absence. Re-run this scan before submission.
- **Reading [Silent Alarm](https://arxiv.org/pdf/2607.12792) and finding it
  already compares danger recognition across quantization levels with
  threshold-free metrics.** The single highest-priority item on this page.
- **Polarized scores.** Not a novelty risk but an existence risk: if
  Llama Guard 3's verdict probabilities are pinned at 0 and 1 the way the
  Artificial Analysis benchmark reports for some guards, every threshold-free
  metric in this project is computed on an instrument with one division on its
  scale, and Gate C would confirm H3 for entirely the wrong reason.
  `evaluation/score_range.py` checks this, and `smoke_test.sh` runs it before
  the sweep. **Read its verdict before spending GPU hours.**

---

## Open items

- [ ] Read [Silent Alarm](https://arxiv.org/pdf/2607.12792) in full.
- [ ] Read [When Quantization Affects Confidence](https://arxiv.org/abs/2405.00632)
      in full; confirm it does not compute AUROC anywhere.
- [ ] Upgrade every `SNIPPET` to at least `ABSTRACT` before writing the
      related-work section.
- [ ] Search again closer to submission; this area is moving fast.
- [ ] Check whether any guardrail leaderboard already publishes AUROC per
      quantization level.
