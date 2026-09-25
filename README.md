# Quantized OSS Guardrails

Decision-boundary drift in quantized LLM guardrails.

> **Not Safer, Just Louder** — quantization shifts a guard model's operating point, causing
> fixed-threshold safety metrics to change even when discrimination is stable.

This file is the operating manual. Plan changes live in `CHANGES_FOR_PLANNER.md`.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/install_engine.sh
```

`llama-cpp-python` is **not** in `requirements.txt`. The correct wheel is
backend-specific, and a plain `pip install llama-cpp-python` gives a CPU-only
build that would silently run the whole experiment on the host CPU of a GPU
pod. `install_engine.sh` detects the backend, installs the matching build, and
verifies `llama_supports_gpu_offload()`. `run_model.py` refuses to start a CUDA
run on a CPU-only wheel.

`torch` and `transformers` are only needed for the Phase 6 layer sweep:

```bash
pip install -r requirements-mechanism.txt
```

`llama.cpp` binaries are only needed for Phase 5 (imatrix) and Phase 7 (mixed precision):

```bash
bash scripts/build_toolchain.sh
export LLAMA_QUANTIZE=third_party/llama.cpp/build/bin/llama-quantize
export LLAMA_IMATRIX=third_party/llama.cpp/build/bin/llama-imatrix
```

---

## Hardware

| | Dev machine | Local CPU box | RunPod pod |
|---|---|---|---|
| Memory | 16 GB unified (M4) | 64 GB RAM | 24 GB VRAM (4090) |
| Compute | Metal | CPU only | CUDA |
| Disk | ~21 GB | 200 GB+ | 250 GB network volume |

FP16 Llama-Guard-3-8B is 16.1 GB of weights plus ~0.5 GB of KV cache at
`n_ctx=4096`, so it fits a 24 GB card with headroom. `--n-gpu-layers 0` forces
CPU; `-1` offloads everything. Q8 is never a substitute for FP16.

### Which numbers depend on the hardware

Two kinds of number come out of this pipeline.

**Decision metrics** — safety rate, precision/recall/F1, FPR/FNR, AUROC/AUPRC,
ECE/Brier, flip and category analysis — are functions of the logits. They do
not depend on the machine, provided every precision is scored on the *same*
backend and build. llama.cpp's CUDA kernels are not bitwise identical to its
CPU kernels, and Gate C's tolerance is an AUROC gap of 0.02, so a sweep that
mixed backends would be indefensible.

**Efficiency metrics** — latency, throughput, memory — are properties of the
(model, quantization, engine, hardware) configuration, not of quantization
alone. Running all precisions on one pod is therefore a *more* controlled
comparison, not a weaker one. Two consequences for how they are written up:

- Name the hardware in the claim. Not "Q4 is 3x faster than FP16", but "on an
  RTX 4090 with llama.cpp <version>, Q4 is 3x faster than FP16."
- Expect the speed gap to be small on GPU. CPU inference is bandwidth-starved,
  so low precision wins big; a short single-prompt prefill on a 4090 is
  launch-overhead bound, so precisions converge and 4-bit can even lose to FP16
  on dequantization cost. Memory differences stay large and real. "On GPU,
  quantization buys memory rather than latency" is a result, not a failure.

Every run records an environment fingerprint (`env_hash`: backend, GPU name,
driver, llama.cpp version, processor). `analyze.py` reports whether all
prediction files share one, and `--require-same-hardware` makes it fatal.
`environments.csv` lists what was found. Partial GPU offload — llama.cpp
quietly leaving some layers on the host when VRAM is short — aborts the run,
because a CPU/GPU hybrid is not comparable to a full offload.

---

## Setup

Ordered. Each step is checkable; do not skip to inference before step 6 passes.

**1 — Accept the four gated datasets.** One click each, granted instantly (they
gate on `auto`, no manual review). Needed for Phases 4, 5 and 8, not for the
core result.

```
https://huggingface.co/datasets/allenai/wildguardmix
https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
https://huggingface.co/datasets/walledai/StrongREJECT
https://huggingface.co/datasets/lmsys/lmsys-chat-1m
```

Phase 6 also needs `meta-llama/Llama-Guard-3-8B`, which is gated the same way.

**2 — Authenticate.**

```bash
huggingface-cli login
```

**3 — Bootstrap the pod** (RunPod or any CUDA host). Puts the HF cache on the
persistent volume, installs the backend-matched engine build, prints the
environment fingerprint.

```bash
VOLUME=/workspace bash scripts/setup_runpod.sh
```

On the local CPU box instead: `pip install -r requirements.txt && bash scripts/install_engine.sh`.

**4 — (Not needed for this paper.)** Every model in scope is published on the
hub, so no local quantization is required. The `build_toolchain.sh` /
`build_missing_quants.py` path exists for the algorithm-axis follow-up, which
needs `q4_0` — a file the upstream repo never published.

**5 — Preflight.** Checks engine build, GPU offload support, NVML, HF auth,
gated access, every model filename, disk headroom and template fingerprints.
Exits non-zero on anything fatal.

```bash
python scripts/preflight.py --full
```

**6 — Smoke test.** Two precisions end to end, metrics printed.

```bash
MODELS="q3 q4" N=40 bash scripts/smoke_test.sh
```

On a GPU pod, confirm `Device memory (VRAM delta)` is close to the GGUF size —
if it is not, llama.cpp is not fully offloading and no timing is meaningful.

**7 — Prefetch weights.** Otherwise each model downloads lazily on first use,
which puts a multi-gigabyte transfer inside the run: a pod interrupted mid-sweep
re-fetches, and a network failure surfaces as a failed phase rather than a
failed download.

```bash
python scripts/prefetch_models.py --check --models bit-ladder
```

```bash
python scripts/prefetch_models.py --models bit-ladder
```

Weights land in `$MODEL_WEIGHTS_DIR` (`setup_runpod.sh` points it at the volume).
`HF_HOME` alone is not enough — GGUFs are fetched with an explicit `cache_dir`,
which takes precedence, so without this variable ~145 GB lands on the pod's
ephemeral container disk and is lost when the pod stops.

Group sizes: `bit-ladder` 49 GB, `algorithm-4bit` 24 GB, `algorithm-3bit` 15 GB,
`all-families-ladder` 92.5 GB, everything 151 GB. Prefetch the groups for the
phases you are about to run rather than the whole set; use `--evict` on
`run_phase.py` if the volume cannot hold a group.

**8 — Datasets, then run.**

```bash
python scripts/download_datasets.py --core
```

```bash
MODELS="bit-ladder" RESUME=1 bash scripts/run_everything.sh
```

`RESUME=1` lets a preempted pod continue instead of rescoring: predictions are
checkpointed every 50 rows, and a resume onto a changed prompt set is refused.

Read `results/tables/gates.json` before doing anything else.

---

## Phases

| Phase | Command | Gate |
|---|---|---|
| 0 Validity | `python scripts/run_phase.py --phase 0 --n-gpu-layers 0` | A: pattern survives the official template |
| 1 Scores | `python scripts/verify_scorer.py --model q4 --dataset xstest` | B: >99% agreement, >5x cache speedup |
| 2 Threshold-free | `python evaluation/analyze.py` | C: paired AUROC gap equivalent within 0.02 (TOST) |
| 3 Calibration | included in `analyze.py` | D: recalibration collapses the gap |
| 4 Scale | `python scripts/run_phase.py --phase 4 --evict` | replication across models and traffic |
| 5 Algorithms | *deferred to a follow-up paper* | do same-bit algorithms diverge? |
| 5b imatrix | `python scripts/build_imatrix.py` | does safety calibration data help? |
| 6 Mechanism | `python evaluation/layer_sweep.py --n-prompts 200` | is drift concentrated in few layers? |
| 7 Mixed precision | `python scripts/build_mixed_precision.py --k 1 2 4 8` | Q3 memory, FP16 behaviour? |
| 8 Extensions | `python scripts/run_phase.py --phase 8` | adversarial, multilingual, response-level |
| 9 Deployment | included in `analyze.py` | safety per GB, cascade |
| 9b Throughput | `python scripts/benchmark_throughput.py --models bit-ladder` | single-stream prompts/s and prefill tokens/s |

Phases 0–3 run on the 650 prompts you already have. **No dataset downloads are authorised
until Gate D passes**, except the underpowered branch of Gate C.

---

## Models

Keys are `family:precision`. Short aliases: `fp16 q8 q6 q5 q4 q3 q2` map to the 8B family.

```
llama-guard-3-8b    fp16 bf16 q8_0 q6_k q5_k_m q5_k_s q4_k_m q4_0 iq4_xs
                    q3_k_l q3_k_m q3_k_s iq3_xs q2_k
qwen3guard-gen-8b   fp16 q8_0 q6_k q5_k_m q4_k_m q3_k_m q2_k

**Scope of this paper: 12 models — the two bit ladders, Q3 to FP16.**

```
llama-guard-3-8b    fp16 q8_0 q6_k q5_k_m q4_k_m q3_k_m
qwen3guard-gen-8b   fp16 q8_0 q6_k q5_k_m q4_k_m q3_k_m
```

`q2_k` is out of scope and not in `BIT_LADDER`. The question is what *moderate*
quantization does to a guard's operating point, and 2-bit is where k-quants
start losing the model itself — a genuine capability break at the bottom rung
should not decide a gate about the middle of the ladder. It stays in the
registry as a labelled control: run it explicitly with
`--models llama-guard-3-8b:q2_k` for a "where does it finally break" figure.

That is `all-families-ladder` (92.5 GB), the default for `preflight.py`,
`prefetch_models.py` and `verify_models.py`. It answers one question: what does
lower precision do to a guard's operating point, and does the pattern replicate
across two architectures.

The remaining seven registry entries — `bf16 q5_k_s q4_0 iq4_xs q3_k_l q3_k_s
iq3_xs` — are **reserved for a follow-up paper** on the algorithm axis: at a
fixed bit budget, does the compression method change the decision? They stay in
the registry and `evaluation/gates.py` keeps a working test
(`algorithm_axis_report`, McNemar + DeLong, Holm-corrected, per axis), so that
work resumes from a known-good state. Nothing in the default path downloads or
runs them, and their two claims read `NOT_TESTED` in `claims_to_evidence.csv`,
which is the correct record for this paper.

If you pick that work up: `q4_0` is not on the hub and must be built
(`build_missing_quants.py`), and consider restoring `q4_k_s` and `iq4_nl` —
dropped here as duplicates, but useful there as within-family controls.
```

Group specs: `bit-ladder`, `algorithm-4bit`, `algorithm-3bit`, `all-families-ladder`,
`<family>:all`.

```bash
python scripts/run_phase.py --phase 0 --models bit-ladder --dry-run
```

Prints disk requirements before downloading anything. Filenames in `models/registry.py` are
best-effort; if one 404s the loader lists what the repo actually contains.

---

## Datasets

25 specs across six tiers in `scripts/datasets_registry.py`. All 20 ungated specs are
verified to load and normalize; the 5 gated ones need step 1 above. Re-probe any time:

```bash
python scripts/verify_datasets.py --tier A
```

`GATED` means the terms are not accepted (step 1). `LOAD_FAIL` means a wrong id/config/split.
`NORMALIZER` means it loads but the field names differ. Then:

```bash
python scripts/download_datasets.py --tier A
python scripts/download_datasets.py --composite --composite-base-rate 0.05
```

Tiers: **A** core safety · **B** over-refusal · **C** category/severity · **D** adversarial ·
**E** multilingual · **F** response-level.

---

## Disk management

Weights go to `$MODEL_WEIGHTS_DIR`, defaulting to `models/weights`. Set it to a
persistent path on any pod.

The full model set is ~151 GB. Either prefetch per group (step 7) or evict:

```bash
python scripts/run_phase.py --phase 5 --evict --n-gpu-layers 0
```

Downloads, scores, deletes the GGUF, moves on. Predictions are kept.

---

## Layout

```
models/
  registry.py        model families x precisions x algorithms
  templates.py       official Llama Guard 3 and Qwen3Guard prompts
  llm_loader.py      GGUF scorer: predict_score, prefix KV cache, label tokens
  hf_loader.py       PyTorch scorer for the layer sweep
  fake_quant.py      RTN quantize-dequantize, output-projection asymmetry
scripts/
  datasets_registry.py   24 dataset specs with normalizers
  download_datasets.py   materialize to datasets/normalized/*.csv
  verify_datasets.py     probe every spec without full download
  data_loader.py         load, validate, stratified subset
  run_model.py           score one model on one dataset
  run_phase.py           orchestrate a phase, with disk checks and eviction
  verify_scorer.py       Gate B
  verify_gates.py        gates vs synthetic data of known truth
  label_audit.py         Gate A label-noise audit
  preflight.py           environment, models, datasets, templates: all checks
  prefetch_models.py     download weights ahead of a run
  smoke_test.sh          two precisions end to end, metrics printed
  build_missing_quants.py  q4_0 / bf16, absent from the upstream repo
  benchmark_throughput.py single-stream prompts/s and prefill tokens/s
  install_engine.sh      backend-matched llama-cpp-python build
  setup_runpod.sh        pod bootstrap: volume, engine, fingerprint
  build_toolchain.sh     compile llama-quantize and llama-imatrix
  build_imatrix.py       generic vs safety-domain calibration
  build_mixed_precision.py  protect top-k sensitive layers
evaluation/
  hardware.py          backend detection, GPU metadata, VRAM sampling
  profiling.py         backend-aware memory, median-based latency
  metrics.py           legacy + threshold-free metrics
  threshold_analysis.py AUROC, AUPRC, ROC, TPR@FPR, base-rate, log-DOR
  calibration.py       ECE, Brier, reliability, temperature, recalibration
  statistical_tests.py McNemar, DeLong, bootstrap, Holm, BH
  disagreement.py      agreement matrix, flips, flip-rate vs distance
  categories.py        per-category, severity weighting, expected cost
  deployment.py        safety per GB, Pareto, iso-memory, cascade
  layer_sweep.py       PyTorch fake-quant sensitivity probe
  gates.py             gate logic, hardware consistency, claims-to-evidence
  analyze.py           runs everything, writes 21 tables and 9 figures
```

---

## Outputs

`results/tables/` — environments, throughput, summary_metrics, pairwise_tests,
base_rate_sensitivity, per_dataset,
recalibration, error_decomposition, per_category, category_degradation, per_language,
severity_weighted_risk, expected_cost, agreement_matrix, flip_summary,
flip_rate_by_distance, borderline_examples, safety_per_gb, memory_pareto, iso_memory,
uncertainty_cascade, claims_to_evidence, gates.json

`results/figures/` — roc_overlay, roc_zoom_low_fpr, threshold_sweep, base_rate_sensitivity,
margin_distributions, reliability, flip_rate_vs_distance, memory_pareto, uncertainty_cascade

---

## Gates

Criteria are fixed in `evaluation/gates.py` before any data is seen. That
commitment only means something if the criteria can be failed, so
`scripts/verify_gates.py` runs every gate against synthetic prediction sets
whose truth is known by construction — a pure operating-point slide, a real
capability loss, an underpowered sample, and two families of unequal intrinsic
skill — and asserts each verdict. Run it before the sweep and after any edit to
`gates.py`:

```bash
python scripts/verify_gates.py
```


This paper makes one claim: quantization shifts a guard's operating point. The second
claim — that the algorithm matters at fixed bit width — is deferred, and its gate is
implemented but reads `NOT_TESTED`.

Every within-ladder analysis is computed **within a family** and then compared
across families: the four gates, the flip tables (each family flips against its
own FP16), the base-rate crossover, and the uncertainty cascade.
The ladder is a within-architecture question: a table holding both families
sorted by bit width interleaves `llama-guard-3-8b:q8_0` with
`qwen3guard-gen-8b:q8_0`, so any trend or spread taken over the pooled table
measures the gap between two different models rather than the effect of
precision. Each gate reports `per_family` verdicts plus `replicated`, which is
what the second family exists to answer.

The same rule applies outside the gates. A flip is "this prompt changed verdict
when I quantized *this* model", so scoring `qwen3guard-gen-8b:q2_k` against
`llama-guard-3-8b:fp16` measures the distance between two different guards —
and because the two boundaries sit in different places, that artefact has a
direction: as one family's ladder drifts toward the other's boundary its
apparent flip rate *falls* with precision, reversing the trend the flip
analysis exists to show. A base-rate "crossover" pooled across families is
usually just the stronger architecture winning, and a cascade pairing one
family's Q2 with the other's FP16 is an architecture swap, not a precision one.

**A — validity.** This is the gate for the headline claim: that safety is not
monotonic in precision.

`PATTERN_SURVIVES` requires a *significant reversal*. A reversal is a pair of
rungs where the safety curve moves against its own overall trend; it counts
only if the paired McNemar test for that specific pair, restricted to the
harmful prompts, survives Holm correction. Seven rungs measured on a few
hundred prompts will nearly always contain a small dip by chance — 12 of 12
synthetic runs in `verify_gates.py` case 6 produce two or three — so reading
"not strictly monotonic" as "non-monotonic" would license the paper's central
claim on sampling error. `strictly_monotonic_in_sample` is reported separately
from `non_monotonic_confirmed`, and only the second licenses anything.

One practical warning: the test is run on the safety rate, which saturates.
If a guard detects ~99% of harmful prompts at every precision, no reversal can
reach significance regardless of what the model is doing — a real bump gets
compressed into a one-prompt difference. Check `summary_metrics.csv` for
saturation before concluding the pattern is absent; a ceiling is a property of
the prompt set, not a result.

If no reversal survives, the ladder is monotonic → If the ladder *is* monotonic but the
true positive rate and the false positive rate rise together (correlation ≥ 0.5
across the ladder), the verdict is `THRESHOLD_SHIFT`, not a refutation: a guard
flagging more of everything has moved its operating point, which is this
paper's claim rather than a failure of it. `MONOTONIC` — safety moving with
precision while the false positive rate does not follow — is the status that
retires the anomaly and makes the finding "prompt formatting explains reported
quantization-safety effects."

**B — scorer.** `p_unsafe >= 0.5` reproduces argmax labels on >99% of rows. Only the
agreement half is binding. The prefix-cache speedup is informational and its target is
backend-dependent (CPU 5x, Metal 3x, CUDA 1.2x): on a GPU the forward pass is short and
launch-overhead bound, so a correct cache saves far less wall clock. `verify_scorer.py`
also checks that the cache does not change any score, which matters on CUDA where
`save_state`/`load_state` round-trips the KV cache through host memory.

**C — discrimination.** The hinge of the paper, so the test is built to be
failable. Every precision scores the same prompts, so each pair is compared
with a **paired** DeLong test, and the gap is judged by **equivalence**, not by
non-significance:

- `H3_REJECTED_DEGRADATION` / `H3_REJECTED_IMPROVEMENT` — some pair differs by
  ≥ 0.02 AUROC *and* its DeLong test survives Holm correction. Significance
  alone is not enough; on a large enough sample a 0.001 gap is significant and
  irrelevant. The suffix says which end of the ladder the gap favours, after
  orienting each pair by bit width: `auroc_delta` is `auc_a - auc_b` over an
  arbitrary pair ordering, so its raw sign carries no information. Both
  suffixes reject H3, but they are opposite findings — degradation is the
  ordinary result, improvement is the strong form of this project's premise
  and the one outcome Gate D cannot repair away. `H3_REJECTED_MIXED` means
  pairs point both ways or the families disagree; bare `H3_REJECTED` means bit
  widths were unavailable and the direction could not be established.
- `H3_CONFIRMED` — every pair passes TOST: the 90% CI on the paired difference
  lies entirely inside ±0.02.
- `UNDERPOWERED` — neither. The only condition authorising early data scaling.

Confirming H3 is a claim that the gap is *small*, and "we failed to find a
difference" does not support it — a weak test fails to find anything. The
earlier criterion (do two marginal bootstrap CIs overlap?) had both failure
modes: on a few hundred prompts those CIs are wide enough to overlap almost
regardless of the truth, so confirmation was near-automatic and rejection near
unreachable. Under the equivalence rule weak data lands on `UNDERPOWERED`,
which is the honest verdict, and the hypothesis can no longer be confirmed by
the weakness of its own test.

**P — peak improvement.** Does any quantized rung actually beat its own
family's FP16 on safety rate, with the paired McNemar test on the harmful
prompts surviving correction? This is the premise the research plan opens with
and nothing else tests it: Gate A asks whether the curve *reverses*, which is a
different question — a curve can reverse without any rung beating FP16, and a
rung can beat FP16 on a perfectly monotone curve.

`IMPROVEMENT_CONFIRMED` does not on its own license "quantization makes guards
safer". Read it with C and D: with `H3_CONFIRMED` + `REPAIRED` the gain is real
but free, reproducible by moving FP16's own threshold, so it is a property of
the operating point rather than of quantization; only
`H3_REJECTED_IMPROVEMENT` licenses the unqualified claim. The three readings
are fixed in `docs/preregistration.md`.

**D — repair.** Recalibration at matched FPR collapses within-family
cross-precision TPR spread below 0.02.

`claims_to_evidence.csv` marks every claim `LICENSED`, `NOT_LICENSED` or `NOT_TESTED`.
Never write a claim the table has not licensed.

---

## Known gaps

- The layer sweep uses RTN fake quantization, which does not reproduce k-quant block
  structure. It ranks layers; Phase 7 validates the ranking with real GGUF builds. This
  caveat belongs in the methods section.
- The layer sweep runs in bfloat16 (`--dtype auto`; float16 on MPS). float32 is rejected
  outright: an 8B model needs ~32 GB in float32, beyond the experiment hardware, and the
  extra mantissa is irrelevant to a 3-4 bit RTN perturbation whose relative Frobenius
  error is 0.11 (4-bit) to 0.22 (3-bit) against bfloat16's 0.0017.
- `benchmark_throughput.py` measures a single request stream. Concurrent batched serving
  would need llama.cpp's server with continuous batching — a different system and a
  different experiment. Say "single-stream" in any throughput claim.
- Qwen3Guard emits three labels. `--controversial-policy` (default `strict`) decides how
  Controversial folds into the binary decision; all three logits are written to the
  predictions CSV so the choice can be revisited without re-running. "Controversial" is
  not a single token in Qwen's vocabulary — it starts with the shared prefix `" Cont"` —
  so its logit is a slight over-estimate. The safe/unsafe tokens are clean, so the binary
  margin is unaffected.
- Three label mappings in `datasets_registry.py` are judgement calls, each marked with a
  comment at its normalizer: PHTest drops its `controversial` class rather than forcing a
  binary label; XSafety excludes its `commonsense` category, which is not a harm probe;
  RTP-LX binarizes its 1-5 mean annotator toxicity at 3.0. Each changes the base rate of
  its dataset and belongs in the methods section.

## Template provenance

Both prompt templates are transcribed from the model's own `chat_template` — the GGUF
metadata for Llama Guard 3, `tokenizer_config.json` for Qwen3Guard — not from prose docs.
This matters because Gate A asks whether prompt formatting explains reported
quantization-safety effects, and an approximated template would make that untestable.

Each template records its deviations in `models/templates.py`. There is one, common to
both: we score the token after the prompt rather than generating, so a fixed continuation
is appended to put the scoring position exactly on the label. Everything before that point
is byte-identical to `apply_chat_template`.

**Any predictions collected before this correction are not comparable.** The Llama Guard
template was missing category S14 (Code Interpreter Abuse) and the Qwen template was a
guess that omitted the safety-policy block and the empty `<think>` block. Both fingerprints
changed; `template_fingerprint` in every prediction file records which was used.
