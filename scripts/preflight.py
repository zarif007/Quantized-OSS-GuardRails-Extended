"""
preflight.py — verify a machine can run the experiment before it starts.

Every check here corresponds to something that has actually gone wrong or
would fail silently: a CPU-only engine build on a GPU pod, a model file the
upstream repo never published, a gated dataset, a missing NVML that turns the
memory column into NaN.  Run it on the pod first; it takes about a minute and
touches the network but downloads nothing.

    python scripts/preflight.py                 # models + environment
    python scripts/preflight.py --datasets A B  # also probe those tiers
    python scripts/preflight.py --full          # everything

Exit code is 0 only if nothing is FAIL.  WARN means degraded but usable.
"""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, FAIL = "OK", "WARN", "FAIL"
RESULTS = []


def record(section, name, status, detail=""):
    RESULTS.append((section, name, status, detail))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"  [{mark}] {name:<38} {detail}")


# ----------------------------------------------------------------------


def check_environment(n_gpu_layers):
    print("\n=== Environment ===")
    from evaluation.hardware import (
        CUDA,
        detect_backend,
        env_fingerprint,
        fingerprint_hash,
        llama_cpp_supports_offload,
    )

    backend = detect_backend(n_gpu_layers)
    fp = env_fingerprint(n_gpu_layers)
    record("env", "backend", OK, f"{backend} | {fp.get('gpu_name') or fp['processor']}")
    record("env", "environment fingerprint", OK, fingerprint_hash(fp))

    version = fp.get("llama_cpp_version")
    if not version:
        record("env", "llama-cpp-python", FAIL, "not importable; run scripts/install_engine.sh")
    else:
        record("env", "llama-cpp-python", OK, version)

    offload = llama_cpp_supports_offload()
    if backend == CUDA and offload is False:
        record("env", "engine GPU support", FAIL,
               "CPU-only wheel on a CUDA host; run scripts/install_engine.sh")
    elif offload is False:
        record("env", "engine GPU support", WARN, "CPU-only build (fine for --n-gpu-layers 0)")
    else:
        record("env", "engine GPU support", OK, str(offload))

    if backend == CUDA:
        try:
            import pynvml  # noqa: F401
            record("env", "NVML (memory measurement)", OK, "importable")
        except ImportError:
            record("env", "NVML (memory measurement)", FAIL,
                   "pip install nvidia-ml-py — VRAM would be unmeasured")
        total = fp.get("gpu_total_memory_mb")
        if total:
            headroom = OK if total >= 20000 else WARN
            record("env", "GPU memory", headroom,
                   f"{total/1024:.1f} GB (fp16 8B needs ~17.5 GB)")
    return backend


def check_torch():
    print("\n=== PyTorch (Phase 6 layer sweep only) ===")
    try:
        import torch
    except ImportError:
        record("torch", "torch", WARN,
               "not installed; needed only for Phase 6 (requirements-mechanism.txt)")
        return
    from evaluation.hardware import default_torch_dtype, torch_device

    device = torch_device("auto")
    record("torch", "torch", OK, f"{torch.__version__} device={device} "
                                f"dtype={default_torch_dtype(device)}")


def check_hf_auth():
    print("\n=== HuggingFace access ===")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    try:
        who = HfApi().whoami(token=token)
        record("hf", "authentication", OK, f"{who.get('name')} "
                                          f"({'HF_TOKEN' if token else 'cached login'})")
        authed = True
    except Exception:
        record("hf", "authentication", WARN,
               "not logged in; gated datasets and Phase 6 weights will fail. "
               "Run: huggingface-cli login")
        authed = False
    return authed


def check_gated_datasets():
    """
    Test real access, not just visibility.

    dataset_info and list_repo_files succeed on a gated repo whose terms have
    not been accepted -- the metadata is public and only the data is blocked.
    A HEAD on an actual data file is the cheapest request that distinguishes
    them, and it transfers no data.
    """
    print("\n=== Gated dataset access ===")
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    from scripts.datasets_registry import gated_urls

    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    blocked = []
    for repo, url in gated_urls().items():
        try:
            files = api.list_repo_files(repo, repo_type="dataset", token=token)
            data = [f for f in files if f.endswith((".parquet", ".csv", ".json", ".jsonl"))]
            target = data[0] if data else files[0]
            get_hf_file_metadata(hf_hub_url(repo, target, repo_type="dataset"), token=token)
            record("gated", repo, OK, "terms accepted, data readable")
        except Exception as exc:
            blocked.append((repo, url))
            record("gated", repo, WARN, f"{type(exc).__name__} — not accessible")

    if blocked:
        print("\n  -> open each and click 'Agree and access repository' "
              "(instant, no review):")
        for repo, url in blocked:
            print(f"     {url}")
        print("     then: huggingface-cli login")


def check_models(groups):
    print("\n=== Model files ===")
    from huggingface_hub import list_repo_files

    from models.llm_loader import BUILT_DIR, DEFAULT_WEIGHTS_DIR
    from models.registry import expand_many, get_config, sort_keys, total_disk_gb

    # On a pod this must point at the persistent volume; the container disk is
    # ephemeral and far smaller than the model set.
    on_volume = os.path.isabs(DEFAULT_WEIGHTS_DIR) or os.environ.get("MODEL_WEIGHTS_DIR")
    record("models", "weights directory",
           OK if on_volume else WARN,
           f"{os.path.abspath(DEFAULT_WEIGHTS_DIR)}"
           + ("" if on_volume else "  (MODEL_WEIGHTS_DIR unset — fine locally, "
                                   "but on a pod set it to the volume)"))

    keys = sort_keys(expand_many(groups))
    by_repo = {}
    for key in keys:
        by_repo.setdefault(get_config(key)["repo"], []).append(key)

    missing = []
    for repo, repo_keys in by_repo.items():
        try:
            available = {f for f in list_repo_files(repo) if f.endswith(".gguf")}
        except Exception as exc:
            record("models", repo, FAIL, f"{type(exc).__name__}: {str(exc)[:60]}")
            continue
        for key in repo_keys:
            filename = get_config(key)["filename"]
            if filename in available:
                record("models", key, OK, "on hub")
            elif os.path.exists(os.path.join(BUILT_DIR, filename)):
                record("models", key, OK, "built locally")
            else:
                missing.append(key)
                record("models", key, FAIL,
                       f"{filename} not on hub and not in {BUILT_DIR}/")

    if missing:
        print(f"\n  -> build them: python scripts/build_missing_quants.py "
              f"--only {' '.join(k.split(':')[-1] for k in missing)}")

    required = total_disk_gb(keys)
    probe_dir = DEFAULT_WEIGHTS_DIR if os.path.isdir(DEFAULT_WEIGHTS_DIR) else "."
    free = shutil.disk_usage(probe_dir).free / 1000**3
    status = OK if free > required * 1.15 else WARN
    record("models", "disk space", status,
           f"{required:.0f} GB needed, {free:.0f} GB free "
           f"({'prefetch a smaller group or use --evict' if status == WARN else 'ok'})")
    return keys


def check_templates():
    print("\n=== Prompt templates ===")
    from models.templates import TEMPLATES, template_fingerprint

    for name, tpl in TEMPLATES.items():
        record("templates", name, OK,
               f"{template_fingerprint(tpl)} | {len(tpl.prefix)} char prefix | "
               f"{'ternary' if tpl.is_ternary else 'binary'}")
        if not tpl.source:
            record("templates", f"{name} provenance", WARN, "no source recorded")


def check_datasets(tiers):
    print(f"\n=== Datasets (tiers {', '.join(tiers)}) ===")
    from scripts.datasets_registry import DATASETS
    from scripts.verify_datasets import probe

    names = [n for n, s in DATASETS.items() if s.tier in tiers]
    for name in names:
        result = probe(name)
        status = {"OK": OK, "GATED": WARN, "NORMALIZER": FAIL, "LOAD_FAIL": FAIL}[result["status"]]
        record("datasets", name, status, f"{result['status']}: {result['detail'][:70]}")


# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Pre-run environment and asset check")
    parser.add_argument("--models", nargs="*", default=["all-families-ladder"],
                        help="default is this paper's scope: both bit ladders, 14 models")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tiers to probe, e.g. A B. Slow; omitted by default")
    parser.add_argument("--full", action="store_true", help="probe every dataset tier")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    print("=" * 78)
    print(" Preflight")
    print("=" * 78)

    check_environment(args.n_gpu_layers)
    check_torch()
    authed = check_hf_auth()
    check_models(args.models)
    check_templates()
    check_gated_datasets()
    if not authed:
        print("  (not authenticated — gated results above are inconclusive)")

    tiers = ["A", "B", "C", "D", "E", "F"] if args.full else (args.datasets or [])
    if tiers:
        check_datasets([t.upper() for t in tiers])

    fails = [r for r in RESULTS if r[2] == FAIL]
    warns = [r for r in RESULTS if r[2] == WARN]

    print("\n" + "=" * 78)
    print(f" {len(RESULTS) - len(fails) - len(warns)} ok, {len(warns)} warn, {len(fails)} fail")
    if fails:
        print("\n Must fix before running:")
        for _, name, _, detail in fails:
            print(f"   - {name}: {detail}")
    if warns:
        print("\n Degraded but runnable:")
        for _, name, _, detail in warns:
            print(f"   - {name}: {detail}")
    if not fails and not warns:
        print("\n Ready.")
    print("=" * 78)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
