"""
build_missing_quants.py — produce the registry entries the upstream repo lacks.

mradermacher/Llama-Guard-3-8B-GGUF does not publish every quantization the
registry names.  Missing: Q4_0 and BF16.  Q4_0 is the one that matters -- it is
the legacy round-to-nearest scheme that the k-quants are meant to improve on,
and it is one third of Phase 5's 4-bit algorithm axis, so without it the phase
cannot answer whether the algorithm matters at fixed bit width.  BF16 belongs
to no phase group and is optional.

They are built here from the family's own f16 GGUF with llama-quantize, so
they come from exactly the same source weights as the published variants.
Output goes to models/weights/built/, which resolve_model_path checks before
the hub -- built variants are then usable through their normal registry keys.

    bash scripts/build_toolchain.sh
    export LLAMA_QUANTIZE=third_party/llama.cpp/build/bin/llama-quantize
    python scripts/build_missing_quants.py
"""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import list_repo_files

from models.llm_loader import BUILT_DIR, resolve_model_path
from models.registry import MODEL_CONFIGS, RESERVED_PRECISIONS, get_config

# llama-quantize type name per registry precision key.
QUANT_TYPES = {
    "q4_0": "Q4_0", "q4_1": "Q4_1", "q5_0": "Q5_0", "q5_1": "Q5_1",
    "q2_k": "Q2_K", "q3_k_s": "Q3_K_S", "q3_k_m": "Q3_K_M", "q3_k_l": "Q3_K_L",
    "q4_k_s": "Q4_K_S", "q4_k_m": "Q4_K_M", "q5_k_s": "Q5_K_S", "q5_k_m": "Q5_K_M",
    "q6_k": "Q6_K", "q8_0": "Q8_0", "bf16": "BF16",
    "iq3_xs": "IQ3_XS", "iq3_s": "IQ3_S", "iq3_m": "IQ3_M",
    "iq4_nl": "IQ4_NL", "iq4_xs": "IQ4_XS",
}

# These need activation statistics; llama-quantize refuses or degrades badly
# without an importance matrix.
NEEDS_IMATRIX = {"iq3_xs", "iq3_s", "iq3_m", "iq2_xs", "iq2_xxs", "iq1_s"}


def missing_from_hub(family: str):
    """Registry entries for `family` whose file is absent from its hub repo."""
    keys = [k for k, c in MODEL_CONFIGS.items() if c["family"] == family and ":" in k]
    repo = get_config(keys[0])["repo"]
    available = {f for f in list_repo_files(repo) if f.endswith(".gguf")}

    missing = []
    for key in keys:
        config = get_config(key)
        if config["filename"] in available:
            continue
        if os.path.exists(os.path.join(BUILT_DIR, config["filename"])):
            continue
        missing.append(key)
    return repo, available, missing


def build(key: str, source_key: str, n_threads, imatrix, dry_run: bool):
    config = get_config(key)
    precision = config["precision"]
    quant_type = QUANT_TYPES.get(precision)
    if not quant_type:
        print(f"  [skip] {key}: no llama-quantize type known for '{precision}'")
        return None
    if precision in NEEDS_IMATRIX and not imatrix:
        print(f"  [skip] {key}: {quant_type} needs --imatrix "
              f"(build one with scripts/build_imatrix.py)")
        return None

    binary = os.environ.get("LLAMA_QUANTIZE")
    if not dry_run and (not binary or not os.path.exists(binary)):
        raise RuntimeError(
            "LLAMA_QUANTIZE is not set or does not exist. Run scripts/build_toolchain.sh, "
            "then: export LLAMA_QUANTIZE=third_party/llama.cpp/build/bin/llama-quantize"
        )

    out_path = os.path.join(BUILT_DIR, config["filename"])
    if dry_run:
        print(f"  [dry-run] {key} -> {quant_type}  ({out_path})")
        return out_path

    os.makedirs(BUILT_DIR, exist_ok=True)
    source_path = resolve_model_path(get_config(source_key))

    cmd = [binary]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [source_path, out_path, quant_type]
    if n_threads:
        cmd += [str(n_threads)]

    print(f"  [build] {key} -> {quant_type}")
    print(f"          {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # registry size_gb mirrors the hub's listed sizes, which are decimal GB,
    # so compare like with like rather than against GiB.
    size_gb = os.path.getsize(out_path) / 1000**3
    delta = abs(size_gb - config["size_gb"]) / max(config["size_gb"], 1e-9)
    flag = "" if delta < 0.05 else "  <- differs from the registry by more than 5%"
    print(f"  [ok]    {out_path}  ({size_gb:.2f} GB, registry expects "
          f"{config['size_gb']} GB){flag}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build registry quantizations missing from the hub")
    parser.add_argument("--family", default="llama-guard-3-8b")
    parser.add_argument("--source", default=None,
                        help="source model key (default: <family>:fp16)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="build only these precision keys, e.g. q4_0 bf16")
    parser.add_argument("--include-reserved", action="store_true",
                        help="also build precisions reserved for the algorithm-axis "
                             "follow-up; this paper's 14-model scope needs none of them")
    parser.add_argument("--imatrix", default=None)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_key = args.source or f"{args.family}:fp16"
    repo, available, missing = missing_from_hub(args.family)

    print(f"family: {args.family}")
    print(f"repo:   {repo}  ({len(available)} gguf files published)")

    if args.only:
        wanted = {f"{args.family}:{p}" for p in args.only}
        missing = [k for k in missing if k in wanted]
    elif not args.include_reserved:
        reserved = {f"{args.family}:{p}" for p in RESERVED_PRECISIONS}
        skipped = [k for k in missing if k in reserved]
        missing = [k for k in missing if k not in reserved]
        if skipped:
            print(f"\nskipping {len(skipped)} reserved for the algorithm-axis "
                  f"follow-up (--include-reserved to build them):")
            for key in skipped:
                print(f"  {key}")

    if not missing:
        print("\nNothing to build: every registry entry is either published or already built.")
        return

    print(f"\nmissing from the hub ({len(missing)}):")
    for key in missing:
        print(f"  {key:<32} -> {get_config(key)['filename']}")

    print(f"\nsource: {source_key}")
    print("Each build reads the full f16 file and writes a new one; expect several "
          "minutes and ~5 GB of disk per variant.\n")

    built, skipped = [], []
    for key in missing:
        result = build(key, source_key, args.n_threads, args.imatrix, args.dry_run)
        (built if result else skipped).append(key)

    print(f"\nbuilt: {len(built)}   skipped: {len(skipped)}")
    if built and not args.dry_run:
        print(f"Files in {BUILT_DIR}/ are picked up automatically by their registry key.")
        print("Verify one before using it, e.g.:")
        print(f"  python scripts/verify_scorer.py --model {built[0]} --dataset xstest --n 20 --skip-speed")


if __name__ == "__main__":
    main()
