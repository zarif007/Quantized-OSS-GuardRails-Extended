# Change list for the Research Master Plan

Corrections to apply to the 47-section master plan. Section numbers refer to that
document. Nothing here changes the research question or the phase ordering — both are
sound. These fix hardware feasibility, add missing rigour, and close gaps.

---

# GROUP A — Hardware feasibility (blocking; the plan is not runnable without these)

## A1. Add a hardware section at the top

The plan never states what machine it runs on, and several phases assume capabilities
that do not exist. Insert before §1:

```
Dev machine:         Apple M4, 16 GB unified memory, Metal GPU, 21 GB free disk
Experiment machine:  64 GB system RAM, CPU-only (no discrete GPU), 200 GB+ free disk
```

Every item in Group A follows from these two lines.

## A2. §16, §41 Phase 0 — FP16 is assumed but never scheduled, and does not fit on the dev machine

The plan's headline test is *"does recalibrated Q3 match FP16?"*, but FP16 is not in
`MODEL_CONFIGS`, and FP16 Llama-Guard-3-8B is 16.1 GB — it will not load in 16 GB
unified memory.

**Change:** add "create the FP16 baseline" as explicit Phase 0 work, and state that all
FP16 runs happen on the 64 GB machine. Note also that Q8_0 **cannot** be substituted as
an FP16 proxy: the paper's thesis is that quantization shifts the boundary, so using Q8
as the reference assumes the null hypothesis.

## A3. §7 — add prefix KV caching as a hard requirement

Llama Guard's official template runs several hundred tokens (task instruction + full
S1–S13 taxonomy) before the user prompt. On CPU, re-prefilling that for every
classification makes FP16 impractical and the six-precision sweep a multi-day job.

The prefix is identical for every prompt, and only one token's logits are needed.

**Change:** add a subsection to §7 specifying:

```
ONCE per model load:
    prefill the taxonomy prefix
    save the KV state

PER PROMPT:
    restore KV state
    prefill ~30 prompt tokens + response header
    read logits at the final position
    generate ZERO tokens
```

Use `llm.save_state()` / `llm.load_state()` in llama-cpp-python, or manage context
directly with `llm.eval()` and read `llm.scores[-1]`. Add an acceptance check: measured
speedup must exceed 5x, otherwise the caching is not working.

This is a requirement, not an optimisation. Without it Phase 2 onward is infeasible.

## A4. §18 — the listed quantization algorithms are CUDA-only and cannot run

`AWQ`, `GPTQ` and `bnb-NF4` all require CUDA kernels. The experiment machine has no
discrete GPU.

**Change:** replace the §18 comparison set with the llama.cpp-native set, which still
gives five genuinely distinct 4-bit algorithms:

```
Q4_0      legacy round-to-nearest
Q4_K_S    k-quant, small superblocks
Q4_K_M    k-quant, medium superblocks
IQ4_XS    i-quant, importance-matrix based
IQ4_NL    i-quant, non-linear
```

3-bit mirror: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ3_XS`.

The research question is unchanged — *is bit width or algorithm the real variable?* —
only the instruments change.

## A5. §31 — the calibration-data experiment must be rebuilt on llama-imatrix

§31 proposes comparing C4-calibrated versus safety-calibrated AWQ/GPTQ. Not runnable
without CUDA.

**Change:** use `llama-imatrix`, which computes an importance matrix from any corpus:

```
imatrix-generic   wikitext / C4-style text
imatrix-safety    harmful prompts + benign prompts + refusal examples
```

Quantize with each, compare resulting decision boundaries. Same question, CPU-native,
and arguably more novel — no one has done imatrix calibration-source ablation for guard
models.

## A6. §27 — the layer sweep must not use GGUF rebuilds

§27 implies quantizing one layer at a time in GGUF. That means ~32 separate model builds
at roughly 5 GB each, plus a `llama-quantize` binary that is not installed.

**Change:** run the sweep in PyTorch with in-memory fake quantization:

```
load Llama-Guard-3-8B in FP16 under PyTorch (16 GB, fits in 64 GB)
for each transformer layer n:
    fake-quantize layer n's weights in memory (quantize -> dequantize)
    evaluate on a 200-prompt stratified subset
    record: mean margin shift, margin variance change, TPR/FPR delta, AUROC delta
    restore original weights
```

No disk writes, no rebuilds, and precise control over what is perturbed.

**Required caveat in the methods section:** RTN fake-quant does not reproduce k-quant
block structure. This is a *sensitivity probe used to rank layers*; the ranking is
validated with real GGUF builds in §29.

## A7. §29 — note the toolchain dependency

Confirming the mixed-precision result needs `llama-quantize --tensor-type`, which is not
present in the environment (`llama-cpp-python` ships the inference library, not the
quantizer).

**Change:** add "build llama.cpp from source" as an explicit scheduled task, required by
both §29 and §31.

## A8. §46 — the minimum viable paper has a hardware dependency

**Change:** state that even the MVP requires the 64 GB machine, because it includes FP16.

---

# GROUP B — Scientific rigour (the plan is directionally right but not falsifiable as written)

## B1. §8, §41 Phase 2 — "approximately unchanged" is not a decision criterion

§8 says to check whether AUROC is "approximately unchanged," and §41 has a decision point
with no numbers. That permits rationalising whatever the data shows.

**Change:** pre-register the criterion before running the experiment:

```
max pairwise AUROC difference < 0.02, bootstrap CIs overlap
    -> H3 confirmed. Proceed. Paper is "Not Safer, Just Louder."

max pairwise AUROC difference > 0.02, CIs do NOT overlap (DeLong)
    -> H3 rejected. Real capability degradation.
       Paper pivots to characterising WHICH capability degrades.
       Later phases still run; the framing changes, not the work.

gap > 0.02 but CIs overlap
    -> underpowered. Scale to Tier A datasets FIRST, then re-gate.
       This is the ONLY condition that authorises early data scaling.
```

Apply the same treatment to every other decision point in §41.

## B2. §11 — the 62% base rate is an artifact of your own dataset mixture

The plan presents 62% as a property of safety benchmarks. It is not — it results from
concatenating HarmBench (all unsafe, 200) with XSTest (250 safe / 200 unsafe). A reviewer
will call that mixture arbitrary.

**Change:** state this explicitly and reframe. The stronger claim is not "benchmarks are
imbalanced" but:

> The base rate is an experimenter choice, which is precisely why fixed-threshold
> cross-precision comparison is fragile.

## B3. §13 — margin analysis must be quantitative, not visual

Pattern A versus Pattern B is presented as a histogram comparison. Two histograms can be
read either way by eye.

**Change:** fit mean and variance separately per precision, report both with bootstrap
CIs, and state the decision rule:

```
mean shifts, variance stable      -> Pattern A -> boundary drift (H3)
variance grows significantly      -> Pattern B -> quantization noise (H1)
```

## B4. §37 — commit to one multiple-comparison method

§37 lists Holm, Bonferroni and Benjamini-Hochberg "depending on the hypothesis family"
without deciding.

**Change:** define the hypothesis family explicitly and commit to one method. Holm or
Benjamini-Hochberg preferred over Bonferroni given the comparison count grows to
6 precisions × 2-3 models × 5+ datasets. Tie DeLong's test specifically to the Phase 2
AUROC gate.

## B5. §19, §41 Phase 4 — lock data scaling behind a gate

The plan says not to start with ten datasets, but never states what unlocks them.

**Change:** add an explicit rule — *no dataset downloads are authorised until threshold
recalibration (§15) is complete*, with the single exception of the "ambiguous / under-
powered" branch in B1.

## B6. §41 Phase 0 — defer the Qwen template

Phase 0 currently includes implementing the Qwen3Guard template. Qwen is not needed until
Phase 4.

**Change:** move it to Phase 4 and keep Phase 0 to a single model, so the gate is clean.

## B7. §36 — add the threshold-free efficiency metric explicitly

§36 correctly demotes GES but does not define its replacement.

**Change:** specify the substitute:

```
GES-tf        = (TPR @ FPR=0.05) / (Latency x Memory)
Safety per GB = (TPR @ FPR=0.05) / memory_gb
```

---

# GROUP C — Missing sections (add these)

## C1. Label-noise audit — add as §6b, before Phase 1

Llama Guard's S1–S13 taxonomy may legitimately disagree with HarmBench and XSTest labels
on some items.

**Add:** hand-check 50 prompts stratified across HarmBench and both XSTest subsets.
Record the disagreement rate. It is a ceiling on every metric in the paper and belongs in
the limitations section. Above roughly 10%, it becomes a headline caveat.

## C2. Related-work sweep — add before Phase 1

47 sections and none ask what is already published.

**Add:** one day establishing whether the calibration / decision-boundary analysis has
already been done for guard models. Search quantization + safety alignment, guard model
calibration, PTQ + refusal behaviour. Better to find a collision now than after §27.

## C3. Claims-to-evidence table — add near §44

§44 lists seven strong claims but does not tie them to experiments.

**Add** a table mapping each claim to the specific gate that licenses it:

| Claim | Licensed by |
|---|---|
| Quantization does not change guardrail safety monotonically | Phase 0 pattern survives |
| Apparent gains are operating-point drift, not discrimination | Phase 2 AUROC gate confirms H3 |
| Fixed-threshold evaluation misranks precisions under imbalance | §11 |
| Realistic prevalence reverses the benchmark ranking | §11 + ToxicChat |
| Threshold recalibration recovers the difference for free | §15 |
| Bit width is not the correct independent variable | §18 |
| A small subset of layers drives the drift | §27 |
| Safety-aware mixed precision beats uniform quantization | §30 |

Rule: claim only what its gate actually licensed.

## C4. Time and disk budget per phase

The plan has no cost estimates, so there is no way to tell whether a phase fits in a
weekend or a month.

**Add:** per-phase estimates of wall-clock time (measure once after A3 caching lands) and
peak disk. Note that the full model set — 8B ladder + FP16 + Qwen ladder + algorithm axis
— runs well over 100 GB, so include a download-evaluate-delete policy.

---

# GROUP D — Structure

## D1. Split the document

47 sections is a reference document, not a plan. Nobody executes 47 sections in order,
and an agent handed it will thrash.

**Change:** split into two files.

```
PLAN.md       phases, steps, gates, decision criteria, immediate next steps
              -- this is what gets executed

REFERENCE.md  dataset catalogue (Tiers A-F), metric definitions, hypotheses,
              preliminary results tables, optional extensions, venue targets
              -- this is what gets consulted
```

The operative content of the current document is the 15-step list at the end plus the
gates. Everything else is reference.

## D2. Mark all preliminary numbers as pre-template-fix

§3, §4 and §5 report results produced with the wrong prompt template.

**Change:** label those tables explicitly as **pre-template-fix, to be recomputed after
Phase 0**. Otherwise they will be copied into the paper unchanged.

---

# Summary of what is NOT changing

The research question, the H1/H2/H3 framing, the phase ordering, §42's "what not to do",
and the §46 minimum viable paper are all sound and should be kept as written. The
validity → mechanism → replication → intervention → deployment progression is the right
spine for this project.
