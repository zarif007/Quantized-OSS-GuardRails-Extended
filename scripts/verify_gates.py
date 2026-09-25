"""
Do the gates land where the truth is?

The gates decide what the paper is allowed to claim, so they are fixed before
any data is seen.  That commitment is only worth something if the criteria can
actually be failed, which is not obvious by reading them.  This script builds
synthetic prediction sets whose ground truth is known by construction and
checks that each gate returns the right verdict on each.

Run it before the sweep, and again after any change to evaluation/gates.py.

    python scripts/verify_gates.py

The cases, and why each one exists:

  1  pure operating point slide  a monotone shift of every margin.  Ranking is
                                 untouched, so discrimination is identical by
                                 construction.  A -> THRESHOLD_SHIFT,
                                 C -> H3_CONFIRMED, D -> REPAIRED.
  2  real capability loss        rank-scrambling noise at low precision.
                                 C must REJECT; a gate that cannot reject here
                                 is decoration.
  3a small n, small difference   a real gap far under tolerance, on 120 rows.
                                 Not resolvable -> UNDERPOWERED.
  3b same difference, n=4000     now resolvable, and resolvably small ->
                                 H3_CONFIRMED.  3a and 3b together are the
                                 power check: the verdict must depend on the
                                 evidence, not only on the effect.
  4  two families, different     each family internally a pure threshold shift,
     intrinsic skill             but ~0.1 AUROC apart from each other.  This is
                                 the shape of the real experiment.  Gates
                                 scoped per family confirm H3; gates pooled
                                 across families reject it on the architecture
                                 gap, which is the wrong answer to a question
                                 nobody asked.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.analyze import build_summary, pairwise_tests, recalibration_table
from evaluation.gates import gate_a, gate_c, gate_d
from models.registry import BIT_LADDER

N_DEFAULT = 650          # the core prompt set: 450 XSTest + 200 HarmBench
BASE_RATE = 0.615        # 400 unsafe / 650


def synth(family, n=N_DEFAULT, base_rate=BASE_RATE, seed=0,
          shift=None, noise=None, skill=1.6, labels=None):
    """
    One family's ladder over one shared prompt set.

    `skill` sets the separation between classes, i.e. the family's intrinsic
    AUROC.  `shift` slides the decision boundary without touching the ranking
    -- the operating point drift the paper is about.  `noise` scrambles the
    ranking -- genuine capability loss.  Every precision scores the *same*
    prompts, as in the real experiment, so the paired tests are meaningful.
    """
    rng = np.random.default_rng(seed)
    # `labels` lets two families score the SAME prompts, as in the real
    # experiment.  Reusing one family's labels while keeping the other's
    # independently drawn margins would decorrelate scores from truth and
    # produce a model that is not a guard at all.
    y = np.asarray(labels, dtype=int) if labels is not None else (
        rng.random(n) < base_rate).astype(int)
    n = len(y)
    latent = rng.normal(skill * y, 1.0)
    frames = []
    for precision in BIT_LADDER:
        margin = latent + (shift or {}).get(precision, 0.0)
        if noise and noise.get(precision):
            margin = margin + rng.normal(0.0, noise[precision], n)
        score = 1.0 / (1.0 + np.exp(-margin))
        frames.append(pd.DataFrame({
            "model": f"{family}:{precision}",
            "prompt_id": [f"{family}_{i}" for i in range(n)],
            "ground_truth": np.where(y == 1, "unsafe", "safe"),
            "prediction": np.where(score >= 0.5, "unsafe", "safe"),
            "p_unsafe": score,
            "margin": margin,
            "dataset": "synthetic",
            "latency_sec": 0.1,
            "eval_tokens": 1,
            "backend": "synthetic",
            "env_hash": "synthetic",
        }))
    return pd.concat(frames, ignore_index=True)


def verdicts(combined):
    models = sorted(combined["model"].unique())
    summary = build_summary(combined, models)
    pairs = pairwise_tests(combined, models)
    recal = recalibration_table(combined, models)
    return (gate_a(pairs, summary), gate_c(summary, pairs), gate_d(recal), summary)


def check(name, combined, expect_a=None, expect_c=None, expect_d=None, expect_shift=None):
    a, c, d, summary = verdicts(combined)
    failures = []
    if expect_shift is not None and a.get("operating_point_shift") != expect_shift:
        failures.append(f"{name} operating_point_shift: expected {expect_shift}, "
                        f"got {a.get('operating_point_shift')}")
    print(f"\n{name}")
    for gate, got, expected in (("A", a, expect_a), ("C", c, expect_c), ("D", d, expect_d)):
        if expected is None:
            mark = "   "
        elif got["status"] == expected:
            mark = "ok "
        else:
            mark = "FAIL"
            failures.append(f"{name} gate {gate}: expected {expected}, got {got['status']}")
        detail = got.get("family_statuses") or ""
        print(f"  {gate} {mark} {got['status']:<28} {detail}")
    return failures, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=N_DEFAULT,
                        help="prompts per case (default: the real core set size)")
    args = parser.parse_args()

    primary, replication = "llama-guard-3-8b", "qwen3guard-gen-8b"
    # A boundary that slides steadily as precision falls.
    slide = {p: 0.22 * i for i, p in enumerate(BIT_LADDER)}
    # A real but immaterial amount of rank scrambling, well under tolerance.
    faint = {p: 0.30 for p in BIT_LADDER[4:]}

    failures = []

    f, _ = check("1. pure operating point slide  (truth: louder, not worse)",
                 synth(primary, n=args.n, shift=slide, seed=1),
                 expect_a="THRESHOLD_SHIFT", expect_c="H3_CONFIRMED", expect_d="REPAIRED",
                 expect_shift=True)
    failures += f

    f, _ = check("2. genuine degradation         (truth: discrimination breaks)",
                 synth(primary, n=args.n, noise={"q4_k_m": 1.6, "q3_k_m": 3.0}, seed=2),
                 expect_c="H3_REJECTED")
    failures += f

    f, _ = check("3a. small n, small difference  (truth: unknowable)",
                 synth(primary, n=120, shift=slide, noise=faint, seed=3),
                 expect_c="UNDERPOWERED", expect_shift=False)
    failures += f

    f, _ = check("3b. same difference, n=4000    (truth: knowably equivalent)",
                 synth(primary, n=4000, shift=slide, noise=faint, seed=3),
                 expect_c="H3_CONFIRMED")
    failures += f

    # 5. A real bump: one precision genuinely flags far more than its
    #    neighbours, so the safety curve rises and then falls.  This is the
    #    non-monotonic pattern the paper is named after, and it must be
    #    licensed.
    # The whole ladder is offset so the safety rate sits in its sensitive
    # range.  Near 1.0 the metric saturates and even a large margin bump moves
    # it by a prompt or two, which is correctly judged noise -- realistic, but
    # useless as a test of whether a real reversal is detected.
    mid = {p: -1.6 + 0.20 * i for i, p in enumerate(BIT_LADDER)}
    bump = dict(mid)
    bump["q5_k_m"] = mid["q5_k_m"] + 0.7
    f, _ = check("5. genuine bump at one rung    (truth: really non-monotonic)",
                 synth(primary, n=args.n, shift=bump, seed=6),
                 expect_a="PATTERN_SURVIVES")
    failures += f

    # 6. The mirror, and the case this rule exists for: a nearly flat ladder
    #    with a little independent rank scrambling per rung -- so the safety
    #    curve wobbles, as a real 650-prompt curve does, but nothing real
    #    reverses.  Two or three reversals appear and none survive the paired
    #    test.  A gate that read "not strictly monotonic" as "non-monotonic"
    #    licensed the paper's headline claim on exactly this.
    flat = {p: -1.6 + 0.03 * i for i, p in enumerate(BIT_LADDER)}
    a_noise = verdicts(synth(primary, n=args.n, shift=flat,
                             noise={p: 0.25 for p in BIT_LADDER}, seed=20))[0]
    fam = a_noise["per_family"][primary]
    bad = a_noise["status"] == "PATTERN_SURVIVES"
    if bad:
        failures.append("6. a wobble with no significant reversal was licensed "
                        "as non-monotonic")
    print("\n6. wobble, no real reversal    (truth: not non-monotonic)")
    print(f"  A {'FAIL' if bad else 'ok '} {a_noise['status']:<28} "
          f"reversals={fam['n_reversals']} significant={fam['n_significant_reversals']}")
    if fam["n_reversals"] == 0:
        failures.append("6. produced no reversals at all, so it does not test the rule")

    both = pd.concat([
        synth(primary, n=args.n, shift=slide, seed=4, skill=1.6),
        synth(replication, n=args.n, shift=slide, seed=5, skill=2.4),
    ], ignore_index=True)
    f, summary = check("4. two families, unequal skill (truth: still a threshold shift)",
                       both, expect_a="THRESHOLD_SHIFT", expect_c="H3_CONFIRMED",
                       expect_d="REPAIRED")
    failures += f

    pooled = summary["auroc"]
    print(f"     pooled cross-family AUROC spread is {pooled.max() - pooled.min():.4f}; "
          f"a gate that did not scope per family would reject H3 on that alone.")

    print()
    if failures:
        print(f"{len(failures)} GATE CHECK FAILURE(S):")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("All gate checks passed: each gate returns the correct verdict on data "
          "whose truth is known by construction.")


if __name__ == "__main__":
    main()
