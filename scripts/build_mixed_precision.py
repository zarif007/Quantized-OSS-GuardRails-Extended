import argparse
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import resolve_model_path
from models.registry import get_config

TENSOR_PATTERNS = [
    "blk.{layer}.attn_q.weight",
    "blk.{layer}.attn_k.weight",
    "blk.{layer}.attn_v.weight",
    "blk.{layer}.attn_output.weight",
    "blk.{layer}.ffn_gate.weight",
    "blk.{layer}.ffn_up.weight",
    "blk.{layer}.ffn_down.weight",
]


def protected_tensor_args(layers, protect_type):
    args = []
    for layer in layers:
        for pattern in TENSOR_PATTERNS:
            args += ["--tensor-type", f"{pattern.format(layer=layer)}={protect_type}"]
    return args


def build_variant(source_model, layers, protect_type, base_type, out_path, n_threads, imatrix=None):
    binary = os.environ.get("LLAMA_QUANTIZE")
    if not binary or not os.path.exists(binary):
        raise RuntimeError("LLAMA_QUANTIZE not set. Run scripts/build_toolchain.sh first.")

    model_path = resolve_model_path(get_config(source_model))
    cmd = [binary]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += protected_tensor_args(layers, protect_type)
    cmd += [model_path, out_path, base_type]
    if n_threads:
        cmd += [str(n_threads)]

    print(f"  building k={len(layers)} protect={protect_type} base={base_type}")
    print("  " + " ".join(cmd[:6]) + f" ... ({len(cmd)} args)")
    subprocess.run(cmd, check=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Safety-aware mixed-precision GGUF builder")
    parser.add_argument("--source-model", default="llama-guard-3-8b:fp16")
    parser.add_argument("--sensitivity-csv", default="results/mechanism/layer_sensitivity.csv")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--protect-type", default="Q8_0")
    parser.add_argument("--base-type", default="Q3_K_M")
    parser.add_argument("--imatrix", default=None)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--out-dir", default="models/mixed_precision")
    args = parser.parse_args()

    if not os.path.exists(args.sensitivity_csv):
        raise SystemExit(
            f"Missing {args.sensitivity_csv}. Run evaluation/layer_sweep.py first (Phase 6)."
        )

    sweep = pd.read_csv(args.sensitivity_csv).sort_values("sensitivity", ascending=False)
    ranked = sweep["layer"].tolist()
    print(f"Layer sensitivity ranking (most sensitive first): {ranked[:12]}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for k in args.k:
        layers = ranked[:k]
        name = f"mixed_k{k}_{args.protect_type}_{args.base_type}.gguf".lower()
        out_path = os.path.join(args.out_dir, name)
        build_variant(args.source_model, layers, args.protect_type, args.base_type,
                      out_path, args.n_threads, args.imatrix)
        manifest.append({
            "k": k,
            "protected_layers": ",".join(map(str, layers)),
            "protect_type": args.protect_type,
            "base_type": args.base_type,
            "path": out_path,
            "size_gb": round(os.path.getsize(out_path) / 1024**3, 3) if os.path.exists(out_path) else None,
        })

    df = pd.DataFrame(manifest)
    manifest_path = os.path.join(args.out_dir, "mixed_precision_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nWrote {manifest_path}")
    print("\nScore each with:")
    for row in manifest:
        print(f"  python scripts/run_model.py --model custom --local-path {row['path']} "
              f"--label mixed-k{row['k']} --dataset xstest")


if __name__ == "__main__":
    main()
