"""
benchmark_throughput.py — engine-level throughput for one model.

What this measures
------------------
Two figures, both for a single request stream:

  prompts_per_sec        completed guard decisions per second
  prefill_tokens_per_sec prompt tokens processed per second

and it sweeps `n_batch`, the llama.cpp prefill chunk size, because that is
the knob through which quantization actually shows up in GPU throughput:
low-precision weights move fewer bytes but pay a dequantization cost per
tile, so the crossover point is batch-size dependent.

What this does NOT measure
--------------------------
Multi-tenant batched serving.  llama.cpp processes one sequence at a time
here, so these are serialised numbers.  Reporting them as "throughput" is
fine as long as the thesis says "single-stream"; a claim about concurrent
serving would need llama.cpp's server with continuous batching, which is a
different system and a different experiment.

Cached vs uncached
------------------
`--mode cached` reuses the prefix KV cache, which is how the evaluation
runs score prompts, and is the number that matches the deployment story.
`--mode uncached` prefills the whole template every time, which is the
number that stresses the engine and separates precisions most clearly.
Both are written; neither is the "real" one on its own.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hardware import detect_backend, env_fingerprint, fingerprint_hash
from evaluation.profiling import ModelProfiler, latency_summary
from models.llm_loader import LLMGuard, resolve_model_path
from models.registry import expand_many, get_config, sort_keys
from scripts.data_loader import get_dataset, stratified_subset


def time_stream(model, prompts, warmup):
    for prompt in prompts[:warmup]:
        model.predict_score(prompt)

    latencies, tokens = [], []
    for prompt in prompts:
        start = time.perf_counter()
        scored = model.predict_score(prompt)
        latencies.append(time.perf_counter() - start)
        tokens.append(scored.get("eval_tokens", 0))
    return np.array(latencies), np.array(tokens)


def benchmark_one(model_key, prompts, n_batch, n_gpu_layers, n_threads, n_ctx,
                  cached, warmup, flash_attn):
    config = get_config(model_key)
    backend = detect_backend(n_gpu_layers)
    model_path = resolve_model_path(config)

    profiler = ModelProfiler(model_path=model_path, backend=backend)
    profiler.before_load()
    model = LLMGuard(
        quant_level=model_key,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        flash_attn=flash_attn,
        n_ctx=n_ctx,
        use_prefix_cache=cached,
    )
    profiler.after_load()
    mem = profiler.stats()

    latencies, tokens = time_stream(model, prompts, warmup)
    summary = latency_summary(latencies)
    total_time = float(latencies.sum())

    row = {
        "model": model_key,
        "precision": config["precision"],
        "family": config["family"],
        "bits": config["bits"],
        "n_batch": n_batch,
        "mode": "cached" if cached else "uncached",
        "backend": backend,
        "n_prompts": len(prompts),
        "prompts_per_sec": len(prompts) / total_time if total_time > 0 else float("nan"),
        "prefill_tokens_per_sec": float(tokens.sum()) / total_time if total_time > 0 else float("nan"),
        "mean_prompt_tokens": float(tokens.mean()) if tokens.size else float("nan"),
        **summary,
        **{f"memory_{k}": v for k, v in mem.items()},
    }
    del model
    return row


def main():
    parser = argparse.ArgumentParser(description="Single-stream throughput benchmark")
    parser.add_argument("--models", nargs="+", default=["bit-ladder"])
    parser.add_argument("--dataset", default="xstest")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--n-batch", nargs="+", type=int, default=[512],
                        help="llama.cpp prefill chunk sizes to sweep")
    parser.add_argument("--mode", choices=["cached", "uncached", "both"], default="both")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each configuration to expose host noise")
    parser.add_argument("--output", default="results/tables/throughput.csv")
    args = parser.parse_args()

    keys = sort_keys(expand_many(args.models))
    df = stratified_subset(get_dataset(args.dataset), args.n_prompts)
    prompts = df["prompt"].tolist()

    modes = [True, False] if args.mode == "both" else [args.mode == "cached"]
    fingerprint = env_fingerprint(args.n_gpu_layers)
    env_hash = fingerprint_hash(fingerprint)

    print(f"Throughput benchmark | {len(keys)} models | {len(prompts)} prompts "
          f"| n_batch={args.n_batch} | env {env_hash}")
    print(f"  {fingerprint.get('gpu_name') or fingerprint['processor']}")

    rows = []
    for key in keys:
        for n_batch in args.n_batch:
            for cached in modes:
                for repeat in range(1, args.repeats + 1):
                    label = f"{key} n_batch={n_batch} {'cached' if cached else 'uncached'} r{repeat}"
                    print(f"\n--> {label}")
                    try:
                        row = benchmark_one(
                            key, prompts, n_batch, args.n_gpu_layers, args.n_threads,
                            args.n_ctx, cached, args.warmup, args.flash_attn,
                        )
                    except Exception as exc:
                        print(f"    failed: {exc}")
                        continue
                    row.update({"repeat": repeat, "env_hash": env_hash,
                                "gpu_name": fingerprint.get("gpu_name")})
                    rows.append(row)
                    print(f"    {row['prompts_per_sec']:.2f} prompts/s | "
                          f"{row['prefill_tokens_per_sec']:.0f} tok/s | "
                          f"median {row['latency_median_sec']*1000:.1f} ms")
                    pd.DataFrame(rows).to_csv(args.output, index=False)

    if not rows:
        print("\nNo successful configurations.")
        return

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({"env_hash": env_hash, **fingerprint,
                   "dataset": args.dataset, "n_prompts": len(prompts),
                   "n_batch_sweep": args.n_batch, "modes": args.mode}, fh, indent=2, default=str)

    print("\n=== Throughput ===")
    cols = ["model", "n_batch", "mode", "prompts_per_sec",
            "prefill_tokens_per_sec", "latency_median_sec", "memory_total_memory_mb"]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    print(f"\nWrote {args.output}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
