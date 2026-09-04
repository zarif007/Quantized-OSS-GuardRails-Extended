import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.templates import LLAMA_GUARD_3_CATEGORIES
from scripts.data_loader import get_datasets

WARNING_THRESHOLD = 0.10


def build_sheet(datasets, n, seed, out_path):
    df = get_datasets(datasets)
    df["stratum"] = df["dataset"] + "_" + df["ground_truth"]

    per_stratum = max(1, n // df["stratum"].nunique())
    sample = (
        df.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(min(per_stratum, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    sample = sample.head(n)

    sheet = pd.DataFrame({
        "audit_id": range(1, len(sample) + 1),
        "dataset": sample["dataset"],
        "stratum": sample["stratum"],
        "prompt": sample["prompt"],
        "dataset_label": sample["ground_truth"],
        "taxonomy_label": "",
        "taxonomy_category": "",
        "agrees": "",
        "reason": "",
    })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.to_csv(out_path, index=False)

    print(f"Wrote {len(sheet)} rows to {out_path}")
    print("\nFill in for each row:")
    print("  taxonomy_label     safe | unsafe   (your judgement under the Llama Guard taxonomy)")
    print("  taxonomy_category  e.g. S1, S9, or blank if safe")
    print("  agrees             yes | no")
    print("  reason             short note, required when agrees=no")
    print("\nTaxonomy reference:")
    print(LLAMA_GUARD_3_CATEGORIES)
    print(f"\nThen run: python scripts/label_audit.py score --sheet {out_path}")


def score_sheet(path):
    df = pd.read_csv(path)
    filled = df[df["agrees"].astype(str).str.strip().str.lower().isin(["yes", "no"])]
    if filled.empty:
        print("No completed rows found. Fill the 'agrees' column with yes/no first.")
        return

    df["agrees_norm"] = df["agrees"].astype(str).str.strip().str.lower()
    disagreements = filled[filled["agrees"].astype(str).str.strip().str.lower() == "no"]
    rate = len(disagreements) / len(filled)

    print(f"Completed rows:  {len(filled)} / {len(df)}")
    print(f"Disagreements:   {len(disagreements)}")
    print(f"Label-noise rate: {rate:.3f}")

    by_stratum = (
        filled.assign(dis=filled["agrees"].astype(str).str.strip().str.lower() == "no")
        .groupby("stratum")["dis"].agg(["size", "sum", "mean"])
        .rename(columns={"size": "n", "sum": "disagreements", "mean": "rate"})
    )
    print("\nBy stratum:")
    print(by_stratum.to_string())

    verdict = "MAJOR_LIMITATION" if rate > WARNING_THRESHOLD else "ACCEPTABLE"
    print(f"\nVERDICT: {verdict} (threshold {WARNING_THRESHOLD:.0%})")
    if verdict == "MAJOR_LIMITATION":
        print("Report this ceiling prominently; it bounds every metric in the paper.")

    out = os.path.splitext(path)[0] + "_result.csv"
    by_stratum.reset_index().assign(overall_rate=rate, verdict=verdict).to_csv(out, index=False)
    print(f"\nWrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Label-noise audit sheet builder and scorer")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--datasets", nargs="+", default=["harmbench", "xstest"])
    build.add_argument("--n", type=int, default=50)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--out", default="results/audit/label_audit_sheet.csv")

    score = sub.add_parser("score")
    score.add_argument("--sheet", default="results/audit/label_audit_sheet.csv")

    args = parser.parse_args()
    if args.command == "build":
        build_sheet(args.datasets, args.n, args.seed, args.out)
    else:
        score_sheet(args.sheet)


if __name__ == "__main__":
    main()
