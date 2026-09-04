import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.threshold_analysis import auroc, rates_at, to_labels
from models.fake_quant import LayerPerturbation, output_projection_asymmetry, quantization_error
from models.hf_loader import HFGuard
from scripts.data_loader import get_datasets, stratified_subset

ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


def score_dataset(model: HFGuard, df: pd.DataFrame, progress: str = "") -> pd.DataFrame:
    rows = []
    iterator = tqdm(df.itertuples(), total=len(df), desc=progress, leave=False)
    for row in iterator:
        s = model.predict_score(row.prompt)
        rows.append({"prompt": row.prompt, "ground_truth": row.ground_truth, **s})
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict:
    y = to_labels(scored["ground_truth"].values)
    p = scored["p_unsafe"].values
    point = rates_at(y, p, 0.5)
    return {
        "auroc": auroc(y, p),
        "tpr": point["tpr"],
        "fpr": point["fpr"],
        "flag_rate": point["flag_rate"],
        "balanced_accuracy": point["balanced_accuracy"],
        "margin_mean": float(np.mean(scored["margin"])),
        "margin_std": float(np.std(scored["margin"], ddof=1)),
    }


def run_sweep(
    family: str,
    datasets: list,
    n_prompts: int,
    bits: int,
    group_size: int,
    output_dir: str,
    dtype: str,
    device: str,
    scope: str,
    layer_limit: int = None,
):
    os.makedirs(output_dir, exist_ok=True)

    df = stratified_subset(get_datasets(datasets), n_prompts)
    print(f"Layer sweep on {len(df)} prompts, {bits}-bit RTN fake quantization, scope={scope}")

    model = HFGuard(family=family, dtype=dtype, device=device)
    print(f"  device={model.device} dtype={model.torch_dtype}")
    model.build_prefix_cache()

    baseline_scored = score_dataset(model, df, "baseline")
    baseline = summarize(baseline_scored)
    baseline_scored.to_csv(os.path.join(output_dir, "layer_sweep_baseline.csv"), index=False)
    print(f"baseline: auroc={baseline['auroc']:.4f} tpr={baseline['tpr']:.4f} "
          f"fpr={baseline['fpr']:.4f} margin_mean={baseline['margin_mean']:+.4f}")

    targets = None
    if scope == "attention":
        targets = ATTENTION_TARGETS
    elif scope == "mlp":
        targets = MLP_TARGETS

    layers = model.layers()
    n_layers = len(layers) if layer_limit is None else min(layer_limit, len(layers))

    rows = []
    for i in range(n_layers):
        layer = layers[i]
        with LayerPerturbation(layer, bits=bits, group_size=group_size, targets=targets):
            model.invalidate_prefix_cache()
            model.build_prefix_cache()
            scored = score_dataset(model, df, f"layer {i}")
        stats = summarize(scored)

        weights = [m.weight.data for _, m in layer.named_modules() if hasattr(m, "weight") and m.weight.dim() == 2]
        err = quantization_error(weights[0], bits, group_size) if weights else {}

        row = {
            "layer": i,
            "scope": scope,
            "bits": bits,
            "auroc": stats["auroc"],
            "auroc_delta": stats["auroc"] - baseline["auroc"],
            "tpr_delta": stats["tpr"] - baseline["tpr"],
            "fpr_delta": stats["fpr"] - baseline["fpr"],
            "flag_rate_delta": stats["flag_rate"] - baseline["flag_rate"],
            "margin_mean": stats["margin_mean"],
            "margin_shift": stats["margin_mean"] - baseline["margin_mean"],
            "margin_std": stats["margin_std"],
            "margin_std_ratio": stats["margin_std"] / max(baseline["margin_std"], 1e-9),
            **{f"weight_{k}": v for k, v in err.items()},
        }
        row["sensitivity"] = abs(row["margin_shift"]) + 10 * abs(row["auroc_delta"])
        rows.append(row)
        print(f"  layer {i:>2}: margin_shift={row['margin_shift']:+.4f} "
              f"auroc_delta={row['auroc_delta']:+.4f} sensitivity={row['sensitivity']:.4f}")

        model.invalidate_prefix_cache()
        model.build_prefix_cache()

    sweep = pd.DataFrame(rows).sort_values("sensitivity", ascending=False).reset_index(drop=True)
    sweep["rank"] = np.arange(1, len(sweep) + 1)
    sweep.to_csv(os.path.join(output_dir, "layer_sensitivity.csv"), index=False)

    asym = output_projection_asymmetry(
        model.lm_head_weight(), model.safe_ids, model.unsafe_ids, bits, group_size
    )
    with open(os.path.join(output_dir, "output_projection_asymmetry.json"), "w") as fh:
        json.dump({"baseline": baseline, "asymmetry": asym, "bits": bits}, fh, indent=2)

    print("\n=== Output projection asymmetry ===")
    for k, v in asym.items():
        print(f"  {k}: {v:+.6f}")

    print("\n=== Top sensitive layers ===")
    print(sweep.head(8)[["rank", "layer", "margin_shift", "auroc_delta", "sensitivity"]].to_string(index=False))
    print(f"\nWrote {output_dir}/layer_sensitivity.csv")
    return sweep


def main():
    parser = argparse.ArgumentParser(description="PyTorch fake-quantization layer sensitivity probe")
    parser.add_argument("--family", default="llama-guard-3-8b")
    parser.add_argument("--datasets", nargs="+", default=["harmbench", "xstest"])
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--scope", choices=["all", "attention", "mlp"], default="all")
    parser.add_argument("--layer-limit", type=int, default=None)
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16"],
                        help="auto = bfloat16 (float16 on MPS). float32 is unavailable: "
                             "an 8B model needs ~32 GB in float32, and the extra mantissa "
                             "is irrelevant to a 3-4 bit RTN perturbation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/mechanism")
    args = parser.parse_args()

    run_sweep(
        family=args.family,
        datasets=args.datasets,
        n_prompts=args.n_prompts,
        bits=args.bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device=args.device,
        scope=args.scope,
        layer_limit=args.layer_limit,
    )


if __name__ == "__main__":
    main()
