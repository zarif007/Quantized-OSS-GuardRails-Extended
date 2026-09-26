"""
Why does the boundary move?  A static test of the output-projection hypothesis.

The guard's decision is a difference of two logits, `logit(unsafe) -
logit(safe)`, and each of those is one row of the output projection dotted
with the final hidden state.  Quantization perturbs those rows.  If it
perturbs them *unevenly* -- the unsafe rows shifted differently from the safe
rows -- the margin picks up a systematic bias, and a constant bias on the
margin is exactly a decision-boundary shift.  That is the mechanism the rest
of the paper measures the consequences of, stated at the level of the weights.

This script tests it without running the model.  It loads the output
projection, quantizes it at each bit width, and compares the perturbation on
the safe label rows against the unsafe label rows.  No prompts, no forward
pass, no GPU: the weight matrix is the whole experiment.

    python scripts/output_asymmetry.py --family llama-guard-3-8b

Then join it against the main run and see whether the two agree:

    python scripts/output_asymmetry.py --family llama-guard-3-8b \
        --decomposition results/tables/calibration_decomposition.csv

THREE THINGS THIS DOES NOT DO, which belong in the methods section:

1.  RTN is not a k-quant.  `rtn_quantize_dequantize` rounds to a symmetric
    grid over fixed groups; Q4_K_M uses super-blocks with their own scales and
    minima.  The perturbation measured here is therefore not the perturbation
    the GGUF actually suffers.  It is a hypothesis generator, and the test of
    the hypothesis is whether the asymmetry it predicts tracks the boundary
    shift the real GGUFs produced -- which is what `--decomposition` reports.

2.  llama.cpp does not quantize every tensor to the nominal width.  A Q4_K_M
    build commonly stores `output.weight` at Q6_K, so the row that this script
    is about may be carrying more bits than the model's name suggests.  The
    sweep therefore runs every bit width and leaves the matching to you: read
    the GGUF's tensor types and compare against the row for the width the
    output projection actually uses, not the one on the label.

3.  The test is underpowered, and a null result is not "no asymmetry".  The
    statistic compares a handful of rows against a handful of rows -- four
    label spellings, often fewer distinct first tokens -- so its noise floor
    is high.  On synthetic weights with an 8x asymmetry planted in the unsafe
    rows, the permutation test finds it at 3 bits (p = 0.007) and misses it at
    4 and 6 (p = 0.48, p = 0.71).  Report a non-significant row as "not
    detectable at this row count", never as evidence the rows are perturbed
    evenly.

4.  Correlation over six rungs is not causation.  Even a clean match between
    asymmetry and boundary shift is six paired points per family.  The causal
    test is the intervention: hold the output projection at full precision,
    rebuild, and see whether the drift disappears.  That is Phase 7, and this
    script only tells you whether it is worth running.
"""
import argparse
import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The bit widths a k-quant ladder rung is named after.  fp16 is the baseline
# and is not quantized: its asymmetry is zero by construction, which is the
# control the other rows are read against.
LADDER_BITS = {"q8_0": 8, "q6_k": 6, "q5_k_m": 5, "q4_k_m": 4, "q3_k_m": 3, "q2_k": 2}


def _resolve_label_ids(tokenizer, variants) -> List[int]:
    """First token of each spelling, as the scorers do.

    Identical to `HFGuardScorer._resolve`.  The scoring position sees one
    token, so the label is decided by the first token of whichever spelling
    the template puts there -- and it is those rows of the output projection,
    not whole words, that this analysis is about.
    """
    ids = []
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks and toks[0] not in ids:
            ids.append(toks[0])
    return ids


def asymmetry_null(weight, n_safe: int, n_unsafe: int, bits: int, group_size: int,
                   draws: int = 2000, seed: int = 0):
    """Where does the observed asymmetry sit against random rows of the same size?

    `asymmetry_mean` is a difference between two means taken over a handful of
    rows -- four label spellings, often collapsing to fewer distinct first
    tokens.  Over that few rows the statistic is noisy: on rows drawn from one
    distribution, with no asymmetry planted at all, a 3-bit sweep still
    produces values within a factor of five of a planted 8x effect.  So the
    raw number cannot be read on its own, and "the label rows are perturbed
    unevenly" is not a finding -- every pair of row sets is perturbed
    unevenly by some amount.

    This builds the null the number has to beat: the same statistic on
    randomly chosen row sets of the same two sizes, from the same matrix, at
    the same bit width.  The observed value's position in that distribution is
    what makes it evidence.
    """
    import numpy as np
    import torch

    from models.fake_quant import rtn_quantize_dequantize

    # Quantize once -- the perturbation does not depend on which rows we are
    # about to look at, and re-quantizing per draw would make this unrunnable.
    delta = (rtn_quantize_dequantize(weight, bits, group_size) - weight).float()
    row_means = delta.mean(dim=1).numpy()

    rng = np.random.default_rng(seed)
    n_rows = row_means.shape[0]
    take = n_safe + n_unsafe
    null = np.empty(draws, dtype=float)
    for i in range(draws):
        pick = rng.choice(n_rows, size=take, replace=False)
        null[i] = row_means[pick[n_safe:]].mean() - row_means[pick[:n_safe]].mean()
    return null


def load_output_projection(hf_id: str, token: Optional[str], cache_dir: Optional[str]):
    """The output projection alone, without materializing the whole model.

    An 8B model is ~16 GB in bf16; its output projection is ~1 GB.  Pulling
    only the shard that holds it keeps this runnable on a laptop, which is the
    difference between a check you actually run and one you plan to.  Falls
    back to a full load when the repo is not sharded safetensors, or when the
    projection is tied to the input embedding and has no row of its own.
    """
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token, cache_dir=cache_dir)

    candidates = ("lm_head.weight", "model.lm_head.weight",
                  "model.embed_tokens.weight", "embed_tokens.weight")
    try:
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        index_path = hf_hub_download(hf_id, "model.safetensors.index.json",
                                     token=token, cache_dir=cache_dir)
        weight_map = json.load(open(index_path))["weight_map"]
        name = next((c for c in candidates if c in weight_map), None)
        if name is None:
            raise KeyError(f"no output projection among {candidates}")
        shard = hf_hub_download(hf_id, weight_map[name], token=token, cache_dir=cache_dir)
        print(f"  output projection: {name} from {weight_map[name]}")
        return tokenizer, load_file(shard)[name]
    except Exception as exc:
        print(f"  shard-only load unavailable ({type(exc).__name__}: {exc});"
              f" falling back to a full model load")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            token=token, cache_dir=cache_dir)
        return tokenizer, model.get_output_embeddings().weight.detach()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="llama-guard-3-8b")
    ap.add_argument("--bits", type=int, nargs="*", default=[2, 3, 4, 5, 6, 8],
                    help="widths to sweep; run them all and match the GGUF's "
                         "actual output.weight type rather than the model name")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--null-draws", type=int, default=2000,
                    help="random row sets used to build the null the observed "
                         "asymmetry is judged against")
    ap.add_argument("--decomposition", default=None,
                    help="calibration_decomposition.csv from the main run; "
                         "joins on bit width and reports the correlation")
    ap.add_argument("--out", default="results/tables/output_asymmetry.csv")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from models.fake_quant import output_projection_asymmetry
    from models.registry import FAMILIES
    from models.templates import get_template

    if args.family not in FAMILIES:
        print(f"Unknown family '{args.family}'. Known: {sorted(FAMILIES)}")
        return 2

    hf_id = FAMILIES[args.family]["hf_id"]
    # get_template keys on the template name ("llama_guard_3"), not the family
    # key ("llama-guard-3-8b"); the registry holds the mapping.
    template = get_template(FAMILIES[args.family]["template"])
    print(f"{args.family}  ({hf_id})")

    tokenizer, weight = load_output_projection(
        hf_id, os.environ.get("HF_TOKEN"), os.environ.get("MODEL_WEIGHTS_DIR"))

    safe_ids = _resolve_label_ids(tokenizer, template.safe_variants)
    unsafe_ids = _resolve_label_ids(tokenizer, template.unsafe_variants)
    if set(safe_ids) & set(unsafe_ids):
        print("safe and unsafe label tokens collide; the asymmetry would be "
              "measured between overlapping row sets and would mean nothing")
        return 1
    print(f"  safe rows:   {safe_ids}")
    print(f"  unsafe rows: {unsafe_ids}")
    print(f"  projection:  {tuple(weight.shape)}\n")

    rows = []
    for bits in sorted(args.bits):
        stats = output_projection_asymmetry(weight, safe_ids, unsafe_ids,
                                            bits=bits, group_size=args.group_size)
        null = asymmetry_null(weight, len(safe_ids), len(unsafe_ids), bits,
                              args.group_size, draws=args.null_draws)
        obs = stats["asymmetry_mean"]
        # Two-sided: the hypothesis is that the label rows are perturbed
        # differently from arbitrary rows, in either direction.
        p = float((np.abs(null) >= abs(obs)).mean())
        z = float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
        stats.update({"null_sd": float(null.std()), "z_vs_null": z, "p_vs_null": p})
        rows.append({"family": args.family, "bits": bits, **stats})
        mark = "  *" if p < 0.05 else ""
        print(f"  {bits}-bit  asymmetry_mean={obs:+.3e}  z={z:+.2f}  p={p:.3f}{mark}  "
              f"safe_rel_norm={stats['safe_relative_norm']:.4f}  "
              f"unsafe_rel_norm={stats['unsafe_relative_norm']:.4f}")

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    print("  z and p compare the label rows against random row sets of the same "
          "sizes.\n  An asymmetry that does not beat that null is the noise floor "
          "of a few-row statistic,\n  not a mechanism.")

    # The hypothesis is not "the rows are perturbed unevenly" -- they always
    # will be, by some amount, and a nonzero number on its own says nothing.
    # It is that the unevenness tracks the boundary shift the real GGUFs
    # produced.  Without that join this table is a curiosity.
    if args.decomposition and os.path.exists(args.decomposition):
        decomp = pd.read_csv(args.decomposition)
        decomp = decomp[decomp["model"].str.startswith(f"{args.family}:")].copy()
        decomp["bits"] = (decomp["model"].str.split(":").str[-1]
                          .map(LADDER_BITS))
        merged = decomp.dropna(subset=["bits"]).merge(table, on="bits", how="inner")
        print("\n=== Does the asymmetry track the measured boundary shift? ===")
        if len(merged) < 3:
            print(f"  only {len(merged)} rungs matched; need at least 3")
        else:
            cols = ["model", "bits", "asymmetry_mean", "location_shift"]
            print(merged[[c for c in cols if c in merged.columns]]
                  .sort_values("bits", ascending=False).round(6).to_string(index=False))
            if "location_shift" in merged.columns:
                r = float(np.corrcoef(merged["asymmetry_mean"],
                                      merged["location_shift"])[0, 1])
                print(f"\n  r(asymmetry_mean, location_shift) = {r:+.3f}  over "
                      f"{len(merged)} rungs")
                print("  A strong correlation supports the output-projection "
                      "hypothesis; it does not establish it.\n  The causal test is "
                      "Phase 7: hold this tensor at full precision, rebuild, and\n"
                      "  see whether the drift disappears.")
    elif args.decomposition:
        print(f"\n  {args.decomposition} not found -- run the main sweep first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
