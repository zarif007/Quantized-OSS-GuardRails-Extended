import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.datasets_registry import DATASETS, TIER_NAMES, get_spec


def probe(name: str, n: int = 5) -> dict:
    spec = get_spec(name)
    result = {"dataset": name, "tier": spec.tier, "hf_id": spec.hf_id, "status": "?", "detail": ""}
    try:
        from datasets import load_dataset

        kwargs = {"cache_dir": "datasets/cache", "streaming": True}
        if spec.config:
            ds = load_dataset(spec.hf_id, spec.config, split=spec.split, **kwargs)
        else:
            ds = load_dataset(spec.hf_id, split=spec.split, **kwargs)

        sample = []
        for i, row in enumerate(ds):
            sample.append(row)
            if i + 1 >= n:
                break
        rows = spec.normalize(sample)
        usable = [r for r in rows if r.get("prompt") and r.get("ground_truth")]
        if usable:
            result["status"] = "OK"
            result["detail"] = f"{len(usable)}/{len(sample)} usable; fields={sorted(sample[0])[:6]}"
        else:
            result["status"] = "NORMALIZER"
            result["detail"] = f"loaded but normalizer produced nothing; fields={sorted(sample[0])}"
    except Exception as exc:
        result["status"] = "LOAD_FAIL"
        result["detail"] = str(exc).split("\n")[0][:160]
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe every dataset spec without downloading it fully")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tier", default=None)
    args = parser.parse_args()

    names = args.datasets or [
        n for n, s in DATASETS.items() if args.tier is None or s.tier == args.tier.upper()
    ]

    print(f"{'dataset':<22} {'tier':<5} {'status':<12} detail")
    print("-" * 110)
    tally = {}
    for name in names:
        r = probe(name)
        tally[r["status"]] = tally.get(r["status"], 0) + 1
        print(f"{r['dataset']:<22} {r['tier']:<5} {r['status']:<12} {r['detail']}")

    print("-" * 110)
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("\nLOAD_FAIL  -> wrong hf_id/config/split, or the dataset is gated (accept terms on the hub)")
    print("NORMALIZER -> dataset loads but field names differ; fix its normalizer in datasets_registry.py")


if __name__ == "__main__":
    main()
