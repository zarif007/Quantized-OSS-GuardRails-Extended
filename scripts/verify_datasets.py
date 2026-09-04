import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.datasets_registry import DATASETS, TIER_NAMES, gated_urls, get_spec

HF_TOKEN = os.environ.get("HF_TOKEN") or None


# Sample deep enough that a normalizer which filters rows (dropping an
# excluded category, or an ambiguous label) is not mistaken for a broken one.
# Several of these datasets are ordered by category, so the first handful of
# rows can all belong to a class the spec deliberately discards.
PROBE_ROWS = 200


def probe(name: str, n: int = PROBE_ROWS) -> dict:
    spec = get_spec(name)
    result = {"dataset": name, "tier": spec.tier, "hf_id": spec.hf_id, "status": "?", "detail": ""}
    try:
        from datasets import load_dataset

        kwargs = {"cache_dir": "datasets/cache", "streaming": True, "token": HF_TOKEN}
        if spec.config:
            ds = load_dataset(spec.hf_id, spec.config, split=spec.split, **kwargs)
        else:
            ds = load_dataset(spec.hf_id, split=spec.split, **kwargs)

        # Several of these datasets are stored grouped by category, so a
        # head-of-file sample can be entirely one class -- and a normalizer
        # that deliberately excludes that class then looks broken.  Shuffle
        # through a buffer so the sample spans more of the file.
        try:
            ds = ds.shuffle(seed=0, buffer_size=n * 10)
        except Exception:
            pass

        sample = []
        for i, row in enumerate(ds):
            sample.append(row)
            if i + 1 >= n:
                break
        rows = spec.normalize(sample)
        usable = [r for r in rows if r.get("prompt") and r.get("ground_truth")]
        if usable:
            labels = {}
            for r in usable:
                labels[r["ground_truth"]] = labels.get(r["ground_truth"], 0) + 1
            kept = f"{len(usable)}/{len(sample)} kept"
            result["status"] = "OK"
            result["detail"] = f"{kept}; labels={labels}"
        else:
            result["status"] = "NORMALIZER"
            result["detail"] = (f"loaded but normalizer produced nothing from {len(sample)} rows; "
                                f"fields={sorted(sample[0])}")
    except Exception as exc:
        message = str(exc).split("\n")[0]
        # Separate "you have not accepted the terms" from "the spec is wrong".
        # Only the first is actionable by the user rather than by the code.
        if "gated" in message.lower() or "authenticated" in message.lower() or "401" in message:
            result["status"] = "GATED"
            result["detail"] = f"accept terms at {spec.hub_url} then set HF_TOKEN"
        else:
            result["status"] = "LOAD_FAIL"
            result["detail"] = message[:160]
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe every dataset spec without downloading it fully")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tier", default=None)
    args = parser.parse_args()

    names = args.datasets or [
        n for n, s in DATASETS.items() if args.tier is None or s.tier == args.tier.upper()
    ]

    print(f"HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET — gated datasets will fail'}")
    print()
    print(f"{'dataset':<22} {'tier':<5} {'status':<12} detail")
    print("-" * 110)
    tally = {}
    for name in names:
        r = probe(name)
        tally[r["status"]] = tally.get(r["status"], 0) + 1
        print(f"{r['dataset']:<22} {r['tier']:<5} {r['status']:<12} {r['detail']}")

    print("-" * 110)
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("\nOK         -> loads and normalizes; ready to download")
    print("GATED      -> accept the terms on the hub, then export HF_TOKEN. All of these")
    print("              gate automatically: accepting grants access immediately.")
    print("LOAD_FAIL  -> wrong hf_id/config/split; fix the spec in datasets_registry.py")
    print("NORMALIZER -> loads but field names differ; fix its normalizer in datasets_registry.py")

    if tally.get("GATED"):
        print("\nAccept these (one click each, then `huggingface-cli login`):")
        for url in gated_urls().values():
            print(f"  {url}")

    sys.exit(1 if (tally.get("LOAD_FAIL") or tally.get("NORMALIZER")) else 0)


if __name__ == "__main__":
    main()
