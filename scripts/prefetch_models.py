"""
prefetch_models.py — download model weights ahead of a run.

Weights are otherwise fetched lazily, the first time each model is scored.
That works, but it puts a multi-gigabyte download inside the run: a pod
interrupted at hour three has to re-fetch, and a download failure surfaces as
a failed phase rather than a failed download.  Pulling everything first makes
the run itself purely compute.

Files land in MODEL_WEIGHTS_DIR (see models/llm_loader.DEFAULT_WEIGHTS_DIR).
On a pod that must be the persistent volume -- setup_runpod.sh sets it.  The
container disk is ephemeral and the full set is ~145 GB.

    python scripts/prefetch_models.py --check
    python scripts/prefetch_models.py --models bit-ladder
    python scripts/prefetch_models.py --models all-families-ladder
"""

import argparse
import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download, try_to_load_from_cache

from models.llm_loader import BUILT_DIR, DEFAULT_WEIGHTS_DIR
from models.registry import expand_many, get_config, sort_keys


def _scan_cache(filename):
    """
    Find a real weight file by exact name under the weights directory.

    This exists because the hub cache helper is not reliable here.  The cache
    records "this file is absent upstream" as a zero-byte marker under
    .no_exist/, and an earlier attempt at the wrong filename casing
    (Llama-Guard-3-8B.F16.gguf) left one.  On a case-insensitive filesystem
    -- macOS by default -- the lookup for the real Llama-Guard-3-8B.f16.gguf
    collides with that marker and the helper reports a present 16 GB file as
    missing, which would trigger a pointless re-download.

    Walking for an exact, case-sensitive basename match and ignoring both
    .no_exist paths and empty files avoids the collision on every platform.
    """
    if not os.path.isdir(DEFAULT_WEIGHTS_DIR):
        return None
    for root, dirs, files in os.walk(DEFAULT_WEIGHTS_DIR):
        if ".no_exist" in root:
            dirs[:] = []
            continue
        for name in files:
            if name == filename:
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) > 0:
                        return path
                except OSError:
                    continue
    return None


def local_state(key):
    """(present, path, how) for one model key, without touching the network."""
    config = get_config(key)
    built = os.path.join(BUILT_DIR, config["filename"])
    if os.path.exists(built) and os.path.getsize(built) > 0:
        return True, built, "built"

    found = _scan_cache(config["filename"])
    if found:
        return True, found, "cached"

    cached = try_to_load_from_cache(
        config["repo"], config["filename"], cache_dir=DEFAULT_WEIGHTS_DIR
    )
    if isinstance(cached, str) and os.path.exists(cached):
        return True, cached, "cached"
    return False, None, "missing"


def report(keys):
    print(f"weights dir: {os.path.abspath(DEFAULT_WEIGHTS_DIR)}")
    print(f"\n{'model':<32} {'size_gb':>8}  state")
    print("-" * 62)
    have_gb = need_gb = 0.0
    missing = []
    for key in keys:
        size = get_config(key)["size_gb"]
        present, _, how = local_state(key)
        if present:
            have_gb += size
        else:
            need_gb += size
            missing.append(key)
        print(f"{key:<32} {size:>8.2f}  {how}")
    print("-" * 62)
    free_gb = shutil.disk_usage(DEFAULT_WEIGHTS_DIR if os.path.isdir(DEFAULT_WEIGHTS_DIR)
                               else ".").free / 1000**3
    print(f"present: {have_gb:.1f} GB   to download: {need_gb:.1f} GB   free: {free_gb:.1f} GB")
    return missing, need_gb, free_gb


def main():
    parser = argparse.ArgumentParser(description="Download model weights before a run")
    parser.add_argument("--models", nargs="+", default=["all-families-ladder"],
                        help="registry keys or group specs (bit-ladder, algorithm-4bit, "
                             "all-families-ladder, <family>:all)")
    parser.add_argument("--check", action="store_true", help="report only, download nothing")
    parser.add_argument("--force", action="store_true", help="proceed despite low disk")
    args = parser.parse_args()

    keys = sort_keys(expand_many(args.models))
    missing, need_gb, free_gb = report(keys)

    if args.check:
        return
    if not missing:
        print("\nEverything is already present.")
        return

    # Leave headroom: the run itself writes predictions, and a full disk during
    # a download leaves a truncated file that looks present on the next check.
    if need_gb > free_gb * 0.9 and not args.force:
        print(f"\nNot enough disk: need {need_gb:.1f} GB, {free_gb:.1f} GB free.\n"
              f"Options: fetch a smaller group, point MODEL_WEIGHTS_DIR at a larger "
              f"volume, or run the phase with --evict (download, score, delete).\n"
              f"Pass --force to try anyway.")
        sys.exit(1)

    print(f"\nDownloading {len(missing)} files ({need_gb:.1f} GB)")
    os.makedirs(DEFAULT_WEIGHTS_DIR, exist_ok=True)
    failed = []
    for i, key in enumerate(missing, start=1):
        config = get_config(key)
        print(f"\n[{i}/{len(missing)}] {key}  ({config['size_gb']:.2f} GB)")
        start = time.time()
        try:
            path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["filename"],
                cache_dir=DEFAULT_WEIGHTS_DIR,
                token=os.environ.get("HF_TOKEN") or None,
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:140]}")
            failed.append(key)
            continue

        got_gb = os.path.getsize(path) / 1000**3
        elapsed = time.time() - start
        # A size far off the registry usually means a truncated transfer, which
        # would otherwise be discovered as a corrupt-model error mid-run.
        delta = abs(got_gb - config["size_gb"]) / max(config["size_gb"], 1e-9)
        flag = "" if delta < 0.05 else "  <- size differs from registry by >5%, re-download"
        print(f"  ok  {got_gb:.2f} GB in {elapsed/60:.1f} min{flag}")

    print(f"\ndownloaded: {len(missing) - len(failed)}   failed: {len(failed)}")
    for key in failed:
        print(f"  {key}")
    if failed:
        sys.exit(1)
    print("\nAll requested weights are local. Verify one before the sweep:")
    print(f"  python scripts/verify_scorer.py --model {keys[0]} --dataset xstest --n 20 --skip-speed")


if __name__ == "__main__":
    main()
