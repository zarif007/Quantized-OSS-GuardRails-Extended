import argparse
import os
import sys
import traceback

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.datasets_registry import DATASETS, TIER_A, by_tier, get_spec

CACHE_DIR = "datasets/cache"
NORMALIZED_DIR = "datasets/normalized"


def normalized_path(name: str) -> str:
    return os.path.join(NORMALIZED_DIR, f"{name}.csv")


def materialize(name: str, force: bool = False, max_rows: int = None) -> str:
    spec = get_spec(name)
    out_path = normalized_path(name)
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {name} already at {out_path}")
        return out_path

    from datasets import load_dataset

    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    kwargs = {"cache_dir": CACHE_DIR, "token": os.environ.get("HF_TOKEN") or None}
    if spec.config:
        ds = load_dataset(spec.hf_id, spec.config, split=spec.split, **kwargs)
    else:
        ds = load_dataset(spec.hf_id, split=spec.split, **kwargs)

    rows = spec.normalize(ds)
    df = pd.DataFrame([r for r in rows if r.get("prompt")])
    df = df.dropna(subset=["prompt", "ground_truth"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)

    df["dataset"] = name
    df["level"] = spec.level
    df["tier"] = spec.tier
    df.to_csv(out_path, index=False)

    counts = df["ground_truth"].value_counts().to_dict()
    base_rate = counts.get("unsafe", 0) / max(len(df), 1)
    print(f"  [ok]   {name}: {len(df)} rows  {counts}  harmful_base_rate={base_rate:.3f}")
    return out_path


def build_realistic_composite(
    harmful_source: str = "toxicchat",
    benign_source: str = "toxicchat",
    base_rate: float = 0.05,
    n_total: int = 5000,
    seed: int = 42,
    name: str = "realistic_traffic",
) -> str:
    frames = []
    for src in {harmful_source, benign_source}:
        path = normalized_path(src)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{src} not materialized; run download first")
        frames.append(pd.read_csv(path))
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["prompt"])

    harmful = pool[pool["ground_truth"] == "unsafe"]
    benign = pool[pool["ground_truth"] == "safe"]
    n_harmful = int(round(n_total * base_rate))
    n_benign = n_total - n_harmful
    if len(harmful) < n_harmful or len(benign) < n_benign:
        n_harmful = min(n_harmful, len(harmful))
        n_benign = min(n_benign, len(benign))
        print(f"  [warn] pool too small; using {n_harmful} harmful / {n_benign} benign")

    df = pd.concat(
        [harmful.sample(n_harmful, random_state=seed), benign.sample(n_benign, random_state=seed)]
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["dataset"] = name
    out_path = normalized_path(name)
    df.to_csv(out_path, index=False)
    print(f"  [ok]   {name}: {len(df)} rows, harmful_base_rate={n_harmful / max(len(df),1):.3f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Materialize datasets into normalized CSVs")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--core", action="store_true", help="harmbench + xstest only")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--composite", action="store_true")
    parser.add_argument("--composite-base-rate", type=float, default=0.05)
    parser.add_argument("--composite-n", type=int, default=5000)
    args = parser.parse_args()

    if args.core:
        names = ["harmbench", "xstest"]
    elif args.datasets:
        names = args.datasets
    elif args.tier:
        names = by_tier(args.tier.upper())
    else:
        names = ["harmbench", "xstest"]

    print(f"Materializing {len(names)} dataset(s)")
    failed = []
    for name in names:
        try:
            materialize(name, force=args.force, max_rows=args.max_rows)
        except Exception as exc:
            failed.append((name, str(exc).split("\n")[0]))
            print(f"  [FAIL] {name}: {exc}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()

    if args.composite:
        try:
            build_realistic_composite(
                base_rate=args.composite_base_rate, n_total=args.composite_n
            )
        except Exception as exc:
            print(f"  [FAIL] realistic_traffic: {exc}")

    if failed:
        print("\nFailed datasets (update hf_id/config/split in scripts/datasets_registry.py):")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
