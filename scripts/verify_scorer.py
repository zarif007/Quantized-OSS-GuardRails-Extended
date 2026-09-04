import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.gates import cache_speedup_target, gate_b
from evaluation.hardware import detect_backend, env_fingerprint, llama_cpp_supports_offload
from models.llm_loader import LLMGuard
from scripts.data_loader import get_dataset, stratified_subset

AGREEMENT_THRESHOLD = 0.99


def check_tokens(model):
    print("--- Label token resolution ---")
    print(model.token_report())
    print(f"Cacheable prefix: {model.prefix_token_count()} tokens")


def check_agreement(model, df, legacy_csv):
    print("\n--- Gate B: score/label agreement ---")
    scored = [model.predict_score(p) for p in df["prompt"]]
    df = df.copy()
    df["p_unsafe"] = [s["p_unsafe"] for s in scored]
    df["prediction"] = [s["prediction"] for s in scored]
    df["threshold_label"] = np.where(df["p_unsafe"] >= 0.5, "unsafe", "safe")

    internal = float((df["prediction"] == df["threshold_label"]).mean())
    print(f"argmax vs p_unsafe>=0.5 agreement: {internal:.4f}")

    if legacy_csv and os.path.exists(legacy_csv):
        legacy = pd.read_csv(legacy_csv)
        merged = df.merge(legacy[["prompt", "prediction"]], on="prompt", suffixes=("", "_legacy"))
        if len(merged):
            external = float((merged["threshold_label"] == merged["prediction_legacy"]).mean())
            print(f"vs legacy predictions ({len(merged)} shared prompts): {external:.4f}")
            print("  note: disagreement is expected here if the template changed")

    passed = internal >= AGREEMENT_THRESHOLD
    print(f"GATE B: {'PASS' if passed else 'FAIL'} (need >= {AGREEMENT_THRESHOLD})")
    return passed, internal


def _timed_pass(model, prompts, warmup):
    for prompt in prompts[:warmup]:
        model.predict_score(prompt)
    start = time.perf_counter()
    for prompt in prompts:
        model.predict_score(prompt)
    return time.perf_counter() - start


def check_cache_speedup(quant_level, df, n_threads, n_gpu_layers, backend, warmup=5):
    """
    Confirm the prefix cache helps, with a backend-appropriate expectation.

    On CPU the skipped prefill dominates, so a large speedup is expected.  On
    a GPU the forward pass is short and launch-overhead bound, so the same
    correct implementation yields much less.  The check is informational
    either way; correctness is settled by the agreement check.
    """
    print("\n--- Prefix cache speedup ---")
    prompts = df["prompt"].tolist()
    target = cache_speedup_target(backend)

    cold = LLMGuard(quant_level, n_threads=n_threads, n_gpu_layers=n_gpu_layers, use_prefix_cache=False)
    uncached = _timed_pass(cold, prompts, warmup)
    cold_scores = [cold.predict_score(p)["p_unsafe"] for p in prompts]
    del cold

    warm = LLMGuard(quant_level, n_threads=n_threads, n_gpu_layers=n_gpu_layers, use_prefix_cache=True)
    cached = _timed_pass(warm, prompts, warmup)
    warm_scores = [warm.predict_score(p)["p_unsafe"] for p in prompts]

    # A cache that changes the scores is a bug, not an optimisation.  This
    # matters more on CUDA, where save_state/load_state round-trips the KV
    # cache through host memory.
    drift = max(abs(a - b) for a, b in zip(cold_scores, warm_scores)) if prompts else 0.0
    label_agree = sum((a >= 0.5) == (b >= 0.5) for a, b in zip(cold_scores, warm_scores)) / max(len(prompts), 1)

    speedup = uncached / cached if cached > 0 else float("nan")
    print(f"no cache : {uncached:.2f}s  ({uncached / len(prompts):.3f} s/prompt)")
    print(f"cached   : {cached:.2f}s  ({cached / len(prompts):.3f} s/prompt)")
    print(f"speedup  : {speedup:.2f}x   (backend={backend}, expect >= {target}x)")
    print(f"cache fidelity: max |dp_unsafe| = {drift:.2e}, label agreement = {label_agree:.4f}")
    if label_agree < 1.0:
        print("  WARNING: the prefix cache changes labels. Investigate before trusting any run.")
    print(f"CACHE CHECK: {'PASS' if speedup >= target else 'REVIEW'}")
    return warm, speedup, drift, label_agree


def main():
    parser = argparse.ArgumentParser(description="Gate B verification")
    parser.add_argument("--model", default="q4")
    parser.add_argument("--dataset", default="xstest")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--legacy-csv", default=None)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()

    backend = detect_backend(args.n_gpu_layers)
    fingerprint = env_fingerprint(args.n_gpu_layers)
    print(f"Backend: {backend} | {fingerprint.get('gpu_name') or fingerprint['processor']} "
          f"| llama-cpp {fingerprint.get('llama_cpp_version')} "
          f"| gpu-offload build: {llama_cpp_supports_offload()}")

    df = stratified_subset(get_dataset(args.dataset), args.n)
    print(f"Verifying '{args.model}' on {len(df)} prompts from {args.dataset}")

    speedup = None
    if args.skip_speed:
        model = LLMGuard(args.model, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers)
    else:
        model, speedup, _, _ = check_cache_speedup(
            args.model, df, args.n_threads, args.n_gpu_layers, backend
        )

    check_tokens(model)
    passed, agreement = check_agreement(model, df, args.legacy_csv)
    verdict = gate_b(agreement, speedup, backend)
    print(f"\nGate B verdict: {verdict['status']} "
          f"(cache_ok={verdict['cache_ok']}, target={verdict['cache_speedup_target']}x)")

    print("\n--- Sample rows ---")
    for prompt in df["prompt"].head(5):
        s = model.predict_score(prompt)
        print(f"  [{s['prediction']:>6}] p={s['p_unsafe']:.4f} margin={s['margin']:+.3f} "
              f"| {prompt[:60]}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
