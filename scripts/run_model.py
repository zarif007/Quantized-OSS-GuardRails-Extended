import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hardware import (
    CUDA,
    detect_backend,
    env_fingerprint,
    fingerprint_hash,
    llama_cpp_supports_offload,
)
from evaluation.profiling import InferenceProfiler, ModelProfiler, latency_summary
from models.llm_loader import LLMGuard, resolve_model_path
from models.registry import MODEL_CONFIGS, get_config
from scripts.data_loader import describe, get_dataset

ERROR_ROW = {
    "prediction": "error",
    "p_unsafe": float("nan"),
    "logit_safe": float("nan"),
    "logit_unsafe": float("nan"),
    "logit_controversial": float("nan"),
    "p_controversial": float("nan"),
    "margin": float("nan"),
    "raw_output": "",
    "eval_tokens": 0,
    "context_tokens": 0,
}


def preflight(backend: str, allow_cpu_wheel: bool):
    """
    Refuse to start a GPU run on a CPU-only llama-cpp-python wheel.

    `pip install llama-cpp-python` ships a CPU build.  Without this check the
    run completes normally on a rented GPU pod while executing entirely on the
    host CPU, and every latency number is silently wrong.
    """
    if backend != CUDA:
        return
    supports = llama_cpp_supports_offload()
    if supports is False and not allow_cpu_wheel:
        print(
            "\nERROR: CUDA is present but this llama-cpp-python build has no GPU support.\n"
            "The run would execute on the host CPU and every timing would be wrong.\n"
            "Rebuild with:\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-binary '
            "llama-cpp-python llama-cpp-python\n"
            "or run scripts/setup_runpod.sh. Pass --allow-cpu-wheel to override.\n"
        )
        sys.exit(2)
    if supports is None:
        print("  note: could not determine whether this llama.cpp build supports offload")


def load_done_ids(out_csv: str, resume: bool):
    if not resume or not os.path.exists(out_csv):
        return set(), [], {}
    try:
        existing = pd.read_csv(out_csv)
    except Exception:
        return set(), [], {}
    rows = existing.to_dict("records")
    print(f"  resuming: {len(rows)} rows already present in {out_csv}")
    prompts = dict(zip(existing["prompt_id"].astype(str), existing["prompt"].astype(str)))
    return set(prompts), rows, prompts


def check_resume_alignment(done_prompts: dict, df, dataset_name: str, allow_mismatch: bool):
    """
    A resume is only safe if prompt ids still point at the same prompts.

    prompt_id is built from the DataFrame index, so resuming with a different
    --subset (or a re-downloaded dataset) would silently glue two different
    prompt sets into one file.
    """
    mismatches = []
    for idx, row in df.iterrows():
        pid = f"{dataset_name.lower()}_{idx}"
        stored = done_prompts.get(pid)
        if stored is not None and stored != str(row["prompt"]):
            mismatches.append(pid)
    if not mismatches:
        return
    print(
        f"\nERROR: resume mismatch on {len(mismatches)} prompt ids "
        f"(e.g. {mismatches[:3]}). The existing CSV was produced from a different "
        f"prompt set — most likely a different --subset or a changed dataset.\n"
        f"Delete the CSV and start over, or re-run with the original arguments. "
        f"Pass --allow-resume-mismatch to override.\n"
    )
    if not allow_mismatch:
        sys.exit(5)


def run_model(
    model_name: str,
    dataset_name: str,
    subset: int = None,
    output_dir: str = "results/predictions",
    n_threads: int = None,
    n_gpu_layers: int = -1,
    n_batch: int = None,
    flash_attn: bool = False,
    controversial_policy: str = "strict",
    no_prefix_cache: bool = False,
    local_path: str = None,
    label: str = None,
    template: str = None,
    n_ctx: int = 4096,
    warmup: int = 5,
    latency_repeats: int = 1,
    latency_subset: int = 100,
    checkpoint_every: int = 50,
    resume: bool = False,
    allow_resume_mismatch: bool = False,
    allow_partial_offload: bool = False,
    allow_cpu_wheel: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    backend = detect_backend(n_gpu_layers)
    preflight(backend, allow_cpu_wheel)

    if local_path:
        config = get_config("llama-guard-3-8b:fp16")
        config = dict(config, local_path=local_path, key=label or "custom",
                      precision=label or "custom", algorithm="custom",
                      size_gb=round(os.path.getsize(local_path) / 1024**3, 3))
        if template:
            config["template"] = template
    else:
        try:
            config = get_config(model_name)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    tag = (label or config["key"]).upper()
    slug = (label or config["key"]).replace(":", "__")
    stem = f"predictions_{slug}_{dataset_name.lower()}"
    out_csv = os.path.join(output_dir, f"{stem}.csv")

    print(f"\n[{tag}] backend: {backend}")
    print(f"[{tag}] Loading dataset: {dataset_name}")
    df = get_dataset(dataset_name, subset_size=subset)
    print(f"[{tag}] {describe(df)}")

    done_ids, results, done_prompts = load_done_ids(out_csv, resume)
    if done_ids:
        check_resume_alignment(done_prompts, df, dataset_name, allow_resume_mismatch)

    print(f"[{tag}] Loading model")
    try:
        model_path = resolve_model_path(config)

        mem_profiler = ModelProfiler(model_path=model_path, backend=backend)
        mem_profiler.before_load()
        model = LLMGuard(
            quant_level=config["key"] if not local_path else model_name,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            flash_attn=flash_attn,
            controversial_policy=controversial_policy,
            n_ctx=n_ctx,
            use_prefix_cache=not no_prefix_cache,
            config_override=config if local_path else None,
        )
        mem_profiler.after_load()
    except Exception as exc:
        print(f"Failed to load {tag}: {exc}")
        sys.exit(1)

    mem = mem_profiler.stats()
    print(f"\n[{tag}] Memory profile:\n{mem_profiler.summary()}")

    if mem["offload_ok"] is False and not allow_partial_offload:
        print(
            f"\nERROR: partial GPU offload detected for {tag}. Latency and memory from a "
            f"CPU/GPU hybrid are not comparable to a fully offloaded run.\n"
            f"Use a larger GPU, lower --n-ctx, or pass --allow-partial-offload to record "
            f"it anyway (the CSV will be flagged).\n"
        )
        sys.exit(3)

    print(f"[{tag}] Label tokens:\n{model.token_report()}")
    if model.template.is_ternary:
        print(f"[{tag}] Controversial policy: {controversial_policy}")
    print(f"[{tag}] Cacheable prefix: {model.prefix_token_count()} tokens")

    fingerprint = env_fingerprint(n_gpu_layers)
    env_hash = fingerprint_hash(fingerprint)
    print(f"[{tag}] Environment {env_hash}: {fingerprint.get('gpu_name') or fingerprint['processor']}")

    meta = model.run_metadata()
    meta.update({
        "dataset": dataset_name.lower(),
        "n_prompts": len(df),
        "warmup_prompts": warmup,
        "latency_repeats": latency_repeats,
        "env_hash": env_hash,
        **fingerprint,
        **{f"memory_{k}": v for k, v in mem.items()},
    })

    # Warm up before any measured prompt.  The first calls after load pay for
    # kernel/graph initialisation and allocator growth, which would otherwise
    # be recorded as ordinary latency samples.
    if warmup > 0:
        print(f"\n[{tag}] Warmup ({warmup} prompts, discarded)")
        for prompt in df["prompt"].head(warmup):
            try:
                model.predict_score(prompt)
            except Exception:
                pass

    pending = [(idx, row) for idx, row in df.iterrows()
               if f"{dataset_name.lower()}_{idx}" not in done_ids]
    print(f"\n[{tag}] Scoring {len(pending)} prompts ({len(done_ids)} already done)")

    def flush():
        pd.DataFrame(results).to_csv(out_csv, index=False)

    for n, (idx, row) in enumerate(tqdm(pending, total=len(pending)), start=1):
        prompt = row["prompt"]

        inf_profiler = InferenceProfiler()
        inf_profiler.start()
        try:
            scored = model.predict_score(prompt)
        except Exception as exc:
            print(f"Error scoring prompt {idx}: {exc}")
            scored = dict(ERROR_ROW)
        latency_sec = inf_profiler.stop()

        results.append({
            "prompt_id": f"{dataset_name.lower()}_{idx}",
            "prompt": prompt,
            "ground_truth": row["ground_truth"],
            "dataset": row["dataset"],
            "prediction": scored["prediction"],
            "p_unsafe": scored["p_unsafe"],
            "logit_safe": scored["logit_safe"],
            "logit_unsafe": scored["logit_unsafe"],
            "logit_controversial": scored.get("logit_controversial", float("nan")),
            "p_controversial": scored.get("p_controversial", float("nan")),
            "margin": scored["margin"],
            "raw_output": scored["raw_output"],
            "latency_sec": latency_sec,
            "eval_tokens": scored.get("eval_tokens", 0),
            "context_tokens": scored.get("context_tokens", 0),
            "model": label or config["key"],
            "precision": config["precision"],
            "family": config["family"],
            "algorithm": config["algorithm"],
            "bits": config["bits"],
            "quantization_method": meta["quantization_method"],
            "category": getattr(row, "category", None),
            "language": getattr(row, "language", "en"),
            "level": getattr(row, "level", "prompt"),
            "template_fingerprint": meta["template_fingerprint"],
            "controversial_policy": meta.get("controversial_policy"),
            "backend": backend,
            "gpu_name": fingerprint.get("gpu_name"),
            "env_hash": env_hash,
            "offload_ok": mem["offload_ok"],
            "weights_mb": mem["weights_mb"],
            "vram_mb": mem["vram_mb"],
            "host_mb": mem["host_mb"],
            "total_memory_mb": mem["total_memory_mb"],
        })

        # Checkpoint so a preempted pod loses at most `checkpoint_every` rows.
        if checkpoint_every and n % checkpoint_every == 0:
            flush()

    flush()

    scored_rows = [r for r in results if r["prediction"] != "error"]
    meta.update(latency_summary([r["latency_sec"] for r in scored_rows]))
    total_tokens = sum(r.get("eval_tokens", 0) for r in scored_rows)
    total_time = sum(r["latency_sec"] for r in scored_rows)
    meta["prefill_tokens_per_sec"] = (total_tokens / total_time) if total_time > 0 else float("nan")
    meta["single_stream_prompts_per_sec"] = (
        1.0 / meta["latency_median_sec"] if meta["latency_median_sec"] > 0 else float("nan")
    )

    if latency_repeats > 1:
        meta["latency_repeat_file"] = run_latency_repeats(
            model, df, dataset_name, latency_repeats, latency_subset, output_dir, slug, tag
        )

    out_meta = os.path.join(output_dir, f"{stem}.meta.json")
    with open(out_meta, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n[{tag}] median latency {meta['latency_median_sec']*1000:.1f} ms "
          f"(p95 {meta['latency_p95_sec']*1000:.1f} ms) | "
          f"{meta['prefill_tokens_per_sec']:.0f} prefill tok/s")
    print(f"[{tag}] Predictions -> {out_csv}")
    print(f"[{tag}] Metadata    -> {out_meta}")

    del model


def run_latency_repeats(model, df, dataset_name, repeats, subset_size, output_dir, slug, tag):
    """
    Re-time a fixed prompt subset several times.

    The scores are already fixed by the main pass; this exists only to show
    how much of the latency spread is run-to-run noise on a shared host,
    which is the part a reviewer will question about cloud measurements.
    """
    prompts = df["prompt"].head(subset_size).tolist()
    rows = []
    print(f"\n[{tag}] Latency repeats: {repeats} passes over {len(prompts)} prompts")
    for repeat in range(1, repeats + 1):
        for i, prompt in enumerate(tqdm(prompts, desc=f"  pass {repeat}", leave=False)):
            profiler = InferenceProfiler()
            profiler.start()
            try:
                scored = model.predict_score(prompt)
                tokens = scored.get("eval_tokens", 0)
            except Exception:
                tokens = 0
            rows.append({
                "repeat": repeat,
                # Position within the timing subset, not the dataset-wide
                # prompt_id: this file is standalone and is not joined back.
                "position": i,
                "dataset": dataset_name.lower(),
                "latency_sec": profiler.stop(),
                "eval_tokens": tokens,
            })

    path = os.path.join(output_dir, f"latency_{slug}_{dataset_name.lower()}.csv")
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)

    per_pass = frame.groupby("repeat")["latency_sec"].median()
    spread = float(per_pass.max() - per_pass.min()) / float(per_pass.median())
    print(f"[{tag}] per-pass median latency: "
          f"{', '.join(f'{v*1000:.1f}ms' for v in per_pass)} "
          f"(spread {spread:.1%} of median)")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one guard model on one dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--output-dir", default="results/predictions")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=None,
                        help="llama.cpp batch size; affects prefill throughput")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--controversial-policy", default="strict",
                        choices=["strict", "lenient", "binary"],
                        help="how Qwen3Guard's third label folds into the binary "
                             "decision; ignored by binary guards")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5,
                        help="prompts scored and discarded before timing starts")
    parser.add_argument("--latency-repeats", type=int, default=1,
                        help="extra timing passes over a fixed subset")
    parser.add_argument("--latency-subset", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="rows between CSV flushes; 0 disables checkpointing")
    parser.add_argument("--resume", action="store_true",
                        help="skip prompts already present in the output CSV")
    parser.add_argument("--allow-resume-mismatch", action="store_true",
                        help="resume even if stored prompt ids no longer match the dataset")
    parser.add_argument("--allow-partial-offload", action="store_true")
    parser.add_argument("--allow-cpu-wheel", action="store_true")

    args = parser.parse_args()
    run_model(
        model_name=args.model,
        dataset_name=args.dataset,
        subset=args.subset,
        output_dir=args.output_dir,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        flash_attn=args.flash_attn,
        controversial_policy=args.controversial_policy,
        no_prefix_cache=args.no_prefix_cache,
        local_path=args.local_path,
        label=args.label,
        template=args.template,
        n_ctx=args.n_ctx,
        warmup=args.warmup,
        latency_repeats=args.latency_repeats,
        latency_subset=args.latency_subset,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        allow_resume_mismatch=args.allow_resume_mismatch,
        allow_partial_offload=args.allow_partial_offload,
        allow_cpu_wheel=args.allow_cpu_wheel,
    )
