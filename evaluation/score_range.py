"""
Does the guard produce scores with anywhere to move?

Every threshold-free claim in this project -- AUROC, the TOST equivalence
band, TPR at a target FPR, the flip-rate-vs-margin curve -- assumes the
guard's scores spread out between 0 and 1.  Some guard models do not do that.
Published guardrail benchmarks report verdict-token probabilities that are
almost perfectly polarized, with the great majority of scores pinned within a
rounding error of 0 or 1, leaving no range to tune a threshold in.

On such a model the analysis does not fail loudly.  It returns numbers.
AUROC collapses toward the accuracy of a single fixed cut, the ROC curve has
a handful of usable points, the paired standard error shrinks for the wrong
reason and TOST confirms equivalence because there was never any resolution
to detect a difference with.  The result looks like Gate C confirming H3 when
what actually happened is that the instrument had one division on its scale.

So this check runs before the sweep, not after it.  Run it on the smoke test
output and read `verdict` before committing GPU hours:

    python -m evaluation.score_range --predictions-dir results/smoke

`POLARIZED` means the prompt set cannot answer the question with this guard,
and the fix is a harder or more borderline prompt set -- not a weaker claim.
"""
import argparse
import glob
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Within this distance of 0 or 1, a score carries no threshold information:
# no cut placed anywhere in the interior separates it from its neighbours.
PIN_EPS = 1e-3

# Fraction of scores allowed to sit at the extremes before the margin-based
# analysis stops meaning anything.  Set with headroom below the ~99.8% seen on
# the most polarized published guards: by the time 95% of a 650-prompt set is
# pinned, roughly 30 prompts remain to place a threshold among, and the ROC
# curve is a few points joined by straight lines.
POLARIZED_FRACTION = 0.95
NARROW_FRACTION = 0.80

# Below this many distinct interior scores there are too few candidate cuts for
# TPR-at-target-FPR to land near the target it was asked for.
#
# This is an ABSOLUTE count, so it is only meaningful once the sample is large
# enough to produce that many interior points in the first place.  A 40-prompt
# smoke test with half its scores pinned has ~20 interior prompts and cannot
# clear the bar however well behaved the guard is -- applying it there reports
# a healthy model as POLARIZED.  Guarded by MIN_N_FOR_RESOLUTION below.
MIN_INTERIOR_POINTS = 20

# Under this many scored prompts the resolution criterion is not evaluated and
# the verdict is UNDERSAMPLED unless the pinning alone is damning.  Pinning is
# a fraction and stays meaningful at any n; counting distinct values does not.
MIN_N_FOR_RESOLUTION = 200


def score_range(df: pd.DataFrame) -> Dict[str, float]:
    """Dynamic-range diagnostics for one model's predictions."""
    if "p_unsafe" not in df.columns:
        return {"n": int(len(df)), "verdict": "NO_SCORES"}

    p = pd.to_numeric(df["p_unsafe"], errors="coerce").dropna().to_numpy(dtype=float)
    if p.size == 0:
        return {"n": 0, "verdict": "NO_SCORES"}

    pinned = (p <= PIN_EPS) | (p >= 1.0 - PIN_EPS)
    interior = p[~pinned]
    frac_pinned = float(pinned.mean())

    # Distinct interior values, not distinct values overall: the two pins are
    # each one value however many prompts land on them, and a threshold can
    # only be placed between distinct interior scores.
    n_interior_unique = int(np.unique(np.round(interior, 6)).size) if interior.size else 0

    out = {
        "n": int(p.size),
        "frac_pinned": frac_pinned,
        "frac_at_zero": float((p <= PIN_EPS).mean()),
        "frac_at_one": float((p >= 1.0 - PIN_EPS).mean()),
        "n_interior": int(interior.size),
        "n_interior_unique": n_interior_unique,
        "p_iqr": float(np.subtract(*np.percentile(p, [75, 25]))),
    }

    if "margin" in df.columns:
        m = pd.to_numeric(df["margin"], errors="coerce").dropna().to_numpy(dtype=float)
        if m.size:
            out["margin_iqr"] = float(np.subtract(*np.percentile(m, [75, 25])))
            out["margin_abs_median"] = float(np.median(np.abs(m)))

    # Of the prompts that are not pinned, how many are distinguishable from one
    # another?  1.0 means every interior prompt has its own score, which is
    # full resolution whatever the sample size.
    out["interior_resolution"] = (
        n_interior_unique / interior.size if interior.size else float("nan"))

    low_resolution = (p.size >= MIN_N_FOR_RESOLUTION
                      and n_interior_unique < MIN_INTERIOR_POINTS)

    if frac_pinned >= POLARIZED_FRACTION or low_resolution:
        out["verdict"] = "POLARIZED"
    elif p.size < MIN_N_FOR_RESOLUTION:
        # Pinning is not damning and the sample is too small to judge
        # resolution.  Say so rather than guessing in either direction.
        out["verdict"] = "UNDERSAMPLED"
    elif frac_pinned >= NARROW_FRACTION:
        out["verdict"] = "NARROW"
    else:
        out["verdict"] = "USABLE"
    return out


def score_range_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in combined.groupby("model"):
        rows.append({"model": model, **score_range(sub)})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def report(combined: pd.DataFrame) -> Dict[str, object]:
    table = score_range_table(combined)
    verdicts = set(table["verdict"]) if not table.empty else set()
    if not verdicts or verdicts == {"NO_SCORES"}:
        status = "NO_SCORES"
        reason = "no continuous scores in these predictions"
    elif "POLARIZED" in verdicts:
        worst = table[table["verdict"] == "POLARIZED"]
        status = "POLARIZED"
        reason = (
            f"{len(worst)} of {len(table)} models pin >= {POLARIZED_FRACTION:.0%} of "
            f"scores at 0 or 1 (or leave < {MIN_INTERIOR_POINTS} distinct interior "
            f"values). AUROC, TOST and TPR-at-target-FPR are not meaningful here; "
            f"use a harder or more borderline prompt set before the sweep"
        )
    elif "UNDERSAMPLED" in verdicts:
        status = "UNDERSAMPLED"
        reason = (f"fewer than {MIN_N_FOR_RESOLUTION} scored prompts, so resolution "
                  f"cannot be judged; pinning alone is not damning. Re-run with more "
                  f"prompts before trusting this either way")
    elif "NARROW" in verdicts:
        status = "NARROW"
        reason = (f"some models pin >= {NARROW_FRACTION:.0%} of scores at the extremes; "
                  f"threshold-free metrics are workable but low-resolution")
    else:
        status = "USABLE"
        reason = "scores span the interval; threshold-free analysis is meaningful"
    return {"check": "score_range", "status": status, "reason": reason, "table": table}


def print_report(combined: pd.DataFrame, title: str = "Score dynamic range") -> str:
    r = report(combined)
    table = r["table"]
    print(f"\n=== {title} (can a threshold move here?) ===")
    if table.empty:
        print("  no predictions")
        return r["status"]
    cols = [c for c in ("model", "n", "frac_pinned", "n_interior", "n_interior_unique",
                        "interior_resolution", "margin_abs_median", "verdict")
            if c in table.columns]
    print(table[cols].round(4).to_string(index=False))
    print(f"  {r['status']}: {r['reason']}")
    if r["status"] == "POLARIZED":
        print("  -> Gate C would confirm H3 for the wrong reason: a scale with one "
              "division cannot show a difference.")
    return r["status"]


def load(predictions_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs in {predictions_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["prediction"] != "error"].reset_index(drop=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", default="results/predictions")
    ap.add_argument("--fail-on-polarized", action="store_true",
                    help="exit non-zero if any model is POLARIZED (for CI or a pod script)")
    args = ap.parse_args(argv)

    status = print_report(load(args.predictions_dir))
    return 1 if (args.fail_on_polarized and status == "POLARIZED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
