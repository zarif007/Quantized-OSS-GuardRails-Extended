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

## Quick start

Local CPU box:

```bash
python scripts/download_datasets.py --core
python scripts/verify_scorer.py --model q4 --dataset xstest --n 40 --n-gpu-layers 0
MODELS="bit-ladder" GPU_LAYERS=0 bash scripts/run_everything.sh
```

RunPod (or any CUDA host):

```bash
VOLUME=/workspace bash scripts/setup_runpod.sh
python scripts/download_datasets.py --core
python scripts/verify_scorer.py --model q4 --dataset xstest --n 40
MODELS="bit-ladder" RESUME=1 bash scripts/run_everything.sh
```

`setup_runpod.sh` points `HF_HOME` at the persistent volume (container disk is
wiped when the pod stops, and the full model set is ~145 GB of GGUF plus ~32 GB
of safetensors), installs the CUDA engine build, and prints the environment
fingerprint. `RESUME=1` makes a preempted pod resume instead of rescoring:
`run_model.py` checkpoints the CSV every 50 rows and refuses to resume onto a
prompt set that no longer matches.

Before either, confirm the machine measures what you think it measures:

```bash
MODELS="q3 q4" N=40 bash scripts/smoke_test.sh
```

Scores two precisions on 40 prompts and prints the environment fingerprint,
safety metrics, and efficiency metrics side by side. On a GPU pod, check that
`Device memory (VRAM delta)` roughly matches the GGUF size — if it does not,
llama.cpp is not fully offloading and the timings mean nothing.

Read `results/tables/gates.json` before doing anything else.

---

## Phases

| Phase | Command | Gate |
|---|---|---|
| 0 Validity | `python scripts/run_phase.py --phase 0 --n-gpu-layers 0` | A: pattern survives the official template |
| 1 Scores | `python scripts/verify_scorer.py --model q4 --dataset xstest` | B: >99% agreement, >5x cache speedup |
| 2 Threshold-free | `python evaluation/analyze.py` | C: AUROC gap < 0.02 with overlapping CIs |
| 3 Calibration | included in `analyze.py` | D: recalibration collapses the gap |
| 4 Scale | `python scripts/run_phase.py --phase 4 --evict` | replication across models and traffic |
| 5 Algorithms | `python scripts/run_phase.py --phase 5 --evict` | do same-bit algorithms diverge? |
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
llama-guard-3-8b    fp16 bf16 q8_0 q6_k q5_k_m q5_k_s q4_k_m q4_k_s q4_0
                    iq4_nl iq4_xs q3_k_l q3_k_m q3_k_s iq3_xs q2_k
qwen3guard-gen-8b   fp16 q8_0 q6_k q5_k_m q4_k_m q3_k_m q2_k
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

24 specs across six tiers in `scripts/datasets_registry.py`. Only HarmBench and XSTest are
verified; the rest need probing before use:

```bash
python scripts/verify_datasets.py --tier A
```

`LOAD_FAIL` means a wrong id/config/split or a gated dataset. `NORMALIZER` means it loads but
the field names differ — fix its normalizer. Then:

```bash
python scripts/download_datasets.py --tier A
python scripts/download_datasets.py --composite --composite-base-rate 0.05
```

Tiers: **A** core safety · **B** over-refusal · **C** category/severity · **D** adversarial ·
**E** multilingual · **F** response-level.

---

## Disk management

The full model set exceeds 100 GB. Use eviction:

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
  label_audit.py         Gate A label-noise audit
  smoke_test.sh          two precisions end to end, metrics printed
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

Criteria are fixed in `evaluation/gates.py` before any data is seen.

**A — validity.** Non-monotonic differences still significant after Holm correction, and not
monotonic in bit width. If it fails, the finding becomes "prompt formatting explains reported
quantization-safety effects."

**B — scorer.** `p_unsafe >= 0.5` reproduces argmax labels on >99% of rows. Only the
agreement half is binding. The prefix-cache speedup is informational and its target is
backend-dependent (CPU 5x, Metal 3x, CUDA 1.2x): on a GPU the forward pass is short and
launch-overhead bound, so a correct cache saves far less wall clock. `verify_scorer.py`
also checks that the cache does not change any score, which matters on CUDA where
`save_state`/`load_state` round-trips the KV cache through host memory.

**C — discrimination.** `H3_CONFIRMED` if max pairwise AUROC gap < 0.02 with overlapping
bootstrap CIs. `H3_REJECTED` if gap > 0.02 with DeLong separation. `UNDERPOWERED` otherwise —
the only condition authorising early data scaling.

**D — repair.** Recalibration at matched FPR collapses cross-precision TPR spread below 0.02.

`claims_to_evidence.csv` marks every claim `LICENSED`, `NOT_LICENSED` or `NOT_TESTED`.
Never write a claim the table has not licensed.

---

## Known gaps

- Qwen3Guard template in `models/templates.py` is a best guess — verify against its model card
  before trusting any Qwen result.
- GGUF filenames for `fp16` and the Qwen family are unverified; the loader reports the real
  file list on failure.
- 22 of 24 dataset specs are unverified. Run `verify_datasets.py` first.
- The layer sweep uses RTN fake quantization, which does not reproduce k-quant block
  structure. It ranks layers; Phase 7 validates the ranking with real GGUF builds. This
  caveat belongs in the methods section.
- The layer sweep runs in bfloat16 (`--dtype auto`; float16 on MPS). float32 is rejected
  outright: an 8B model needs ~32 GB in float32, beyond the experiment hardware, and the
  extra mantissa is irrelevant to a 3-4 bit RTN perturbation. bfloat16 is preferred over
  float16 because it keeps float32's exponent range.
- `benchmark_throughput.py` measures a single request stream. Concurrent batched serving
  would need llama.cpp's server with continuous batching — a different system and a
  different experiment. Say "single-stream" in any throughput claim.
- Phase 6 pulls the gated `meta-llama/Llama-Guard-3-8B`; the pod needs
  `huggingface-cli login` or `HF_TOKEN` with the license accepted.
