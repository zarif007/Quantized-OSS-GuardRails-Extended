import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import resolve_model_path
from models.registry import get_config
from scripts.data_loader import get_datasets

REFUSAL_TEMPLATES = [
    "I can't help with that request.",
    "I'm not able to provide that information.",
    "That falls outside what I can assist with.",
    "I won't provide instructions for that.",
]


def build_corpus(kind: str, out_path: str, n: int, seed: int) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if kind == "generic":
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="datasets/cache")
        lines = [t.strip() for t in ds["text"] if len(t.strip()) > 80][:n]
    else:
        df = get_datasets(["harmbench", "xstest"])
        harmful = df[df["ground_truth"] == "unsafe"]["prompt"].tolist()
        benign = df[df["ground_truth"] == "safe"]["prompt"].tolist()
        lines = []
        for i in range(max(len(harmful), len(benign))):
            if i < len(harmful):
                lines.append(harmful[i])
                lines.append(REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)])
            if i < len(benign):
                lines.append(benign[i])
        lines = lines[:n]

    with open(out_path, "w") as fh:
        fh.write("\n\n".join(lines))
    print(f"  corpus '{kind}': {len(lines)} chunks -> {out_path}")
    return out_path


def run_imatrix(model_key: str, corpus_path: str, out_path: str, n_threads: int, chunks: int):
    binary = os.environ.get("LLAMA_IMATRIX")
    if not binary or not os.path.exists(binary):
        raise RuntimeError(
            "LLAMA_IMATRIX not set or missing. Run scripts/build_toolchain.sh first, then export it."
        )
    model_path = resolve_model_path(get_config(model_key))
    cmd = [binary, "-m", model_path, "-f", corpus_path, "-o", out_path, "--chunks", str(chunks)]
    if n_threads:
        cmd += ["-t", str(n_threads)]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_quantize(model_key: str, imatrix_path: str, quant_type: str, out_path: str, n_threads: int):
    binary = os.environ.get("LLAMA_QUANTIZE")
    if not binary or not os.path.exists(binary):
        raise RuntimeError("LLAMA_QUANTIZE not set. Run scripts/build_toolchain.sh first.")
    model_path = resolve_model_path(get_config(model_key))
    cmd = [binary]
    if imatrix_path:
        cmd += ["--imatrix", imatrix_path]
    cmd += [model_path, out_path, quant_type]
    if n_threads:
        cmd += [str(n_threads)]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Importance-matrix calibration experiment")
    parser.add_argument("--source-model", default="llama-guard-3-8b:fp16")
    parser.add_argument("--quant-type", default="IQ4_XS")
    parser.add_argument("--corpus-size", type=int, default=400)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--out-dir", default="models/imatrix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    built = []

    for kind in ["generic", "safety"]:
        corpus = build_corpus(kind, os.path.join(args.out_dir, f"corpus_{kind}.txt"),
                              args.corpus_size, args.seed)
        imat = run_imatrix(args.source_model, corpus,
                           os.path.join(args.out_dir, f"imatrix_{kind}.dat"),
                           args.n_threads, args.chunks)
        gguf = run_quantize(args.source_model, imat, args.quant_type,
                            os.path.join(args.out_dir, f"guard_{args.quant_type}_{kind}.gguf"),
                            args.n_threads)
        built.append({"calibration": kind, "imatrix": imat, "gguf": gguf, "quant_type": args.quant_type})

    manifest = pd.DataFrame(built)
    manifest_path = os.path.join(args.out_dir, "imatrix_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}")
    print("\nScore each with:")
    for row in built:
        print(f"  python scripts/run_model.py --model custom --local-path {row['gguf']} "
              f"--label imatrix-{row['calibration']} --dataset xstest")


if __name__ == "__main__":
    main()
