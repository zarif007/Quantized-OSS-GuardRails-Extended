"""
verify_models.py — prove every model in a group actually scores.

"The file is present" is not the same as "the model works".  A GGUF can load
and still be unusable: label tokens may fail to resolve or collide in a
family's tokenizer, a backend may lack kernels for an IQ quant, and a very low
bit-width model can return a constant score for every prompt.  That last case
is the dangerous one -- it completes without error and produces a full
predictions CSV in which every row reads p_unsafe = 0.5.

So each model is loaded and scored on a small labelled sample, and checked for:

  tokens        safe/unsafe (and Controversial) resolve to distinct ids
  agreement     p_unsafe >= 0.5 reproduces the argmax label (Gate B)
  variation     scores are not constant across prompts
  separation    AUROC above chance on a sample with both classes

Run it once per machine before committing GPU hours to a sweep.

    python scripts/verify_models.py --models bit-ladder
    python scripts/verify_models.py --models all-families-ladder --local-only
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hardware import detect_backend, env_fingerprint, fingerprint_hash
from evaluation.threshold_analysis import auroc, to_labels
from models.llm_loader import LLMGuard
from models.registry import expand_many, get_config, sort_keys
from scripts.data_loader import get_dataset, stratified_subset

AGREEMENT_MIN = 0.99
# Below this, every prompt is getting effectively the same score.
SCORE_STD_MIN = 1e-3
AUROC_MIN = 0.55


def check_model(key, df, n_gpu_layers, n_threads, controversial_policy):
    result = {"model": key, "verdict": "FAIL", "detail": "", "n": len(df)}
    start = time.time()
    model = None
    try:
        model = LLMGuard(
            key,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            controversial_policy=controversial_policy,
        )
        result["load_sec"] = round(time.time() - start, 1)
        result["ternary"] = model.template.is_ternary

        scored = [model.predict_score(p) for p in df["prompt"]]
        p = np.array([s["p_unsafe"] for s in scored])
        pred = np.array([s["prediction"] for s in scored])

        agreement = float(np.mean(pred == np.where(p >= 0.5, "unsafe", "safe")))
        std = float(np.std(p))
        y = to_labels(df["ground_truth"].values)
        area = auroc(y, p) if len(set(y)) > 1 else float("nan")

        result.update({
            "agreement": round(agreement, 4),
            "p_std": round(std, 4),
            "auroc": None if np.isnan(area) else round(area, 4),
            "margin_range": f"{np.min([s['margin'] for s in scored]):+.1f}.."
                            f"{np.max([s['margin'] for s in scored]):+.1f}",
        })

        problems = []
        if agreement < AGREEMENT_MIN:
            problems.append(f"agreement {agreement:.3f} < {AGREEMENT_MIN}")
        if std < SCORE_STD_MIN:
            problems.append(f"scores constant (std {std:.2e}) — the model is not "
                            f"discriminating; check logit extraction")
        if not np.isnan(area) and area < AUROC_MIN:
            problems.append(f"AUROC {area:.3f} at or below chance")

        result["verdict"] = "FAIL" if problems else "OK"
        result["detail"] = "; ".join(problems)
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
    finally:
        if model is not None:
            del model
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify every model in a group actually scores")
    parser.add_argument("--models", nargs="+", default=["all-families-ladder"])
    parser.add_argument("--dataset", default="xstest")
    parser.add_argument("--n", type=int, default=24,
                        help="prompts per model; enough for agreement and a rough AUROC")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--controversial-policy", default="strict",
                        choices=["strict", "lenient", "binary"])
    parser.add_argument("--local-only", action="store_true",
                        help="skip models whose weights are not already local")
    parser.add_argument("--output", default="results/tables/model_verification.csv")
    args = parser.parse_args()

    keys = sort_keys(expand_many(args.models))
    if args.local_only:
        from scripts.prefetch_models import local_state
        keys = [k for k in keys if local_state(k)[0]]

    df = stratified_subset(get_dataset(args.dataset), args.n)
    fp = env_fingerprint(args.n_gpu_layers)

    print(f"Verifying {len(keys)} models on {len(df)} prompts from {args.dataset}")
    print(f"Backend: {detect_backend(args.n_gpu_layers)} | "
          f"{fp.get('gpu_name') or fp['processor']} | env {fingerprint_hash(fp)}\n")

    rows = []
    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {key} ... ", end="", flush=True)
        row = check_model(key, df, args.n_gpu_layers, args.n_threads,
                          args.controversial_policy)
        rows.append(row)
        if row["verdict"] == "OK":
            print(f"OK   agreement={row['agreement']} auroc={row['auroc']} "
                  f"margins {row['margin_range']}")
        else:
            print(f"FAIL {row['detail']}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    cols = [c for c in ["model", "verdict", "agreement", "p_std", "auroc",
                        "margin_range", "load_sec", "detail"] if c in out.columns]
    print("\n" + out[cols].to_string(index=False))

    failed = out[out["verdict"] != "OK"]
    print(f"\n{len(out) - len(failed)}/{len(out)} usable   ->  {args.output}")
    if len(failed):
        print("\nNot usable:")
        for _, r in failed.iterrows():
            print(f"  {r['model']}: {r['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
