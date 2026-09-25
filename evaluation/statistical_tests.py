import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_RESAMPLES = 4000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    rng = _rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_items, n_items)
        values[i] = statistic(idx)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), values
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return lo, hi, values


# Above this many discordant pairs the exact test is both unnecessary and
# expensive: the normal approximation agrees to several decimal places, while
# the exact sum costs ~15 s per test at n=20000 -- minutes across a full
# pairwise table.  Phase 4 scales to Tier A, so this path is reached in
# normal use, not only in principle.
EXACT_BINOMIAL_MAX_N = 1000


def binomial_two_sided(n: int, k: int, exact_max: int = EXACT_BINOMIAL_MAX_N) -> float:
    """
    Two-sided binomial test against p = 0.5.

    The division is done on Python integers.  `2.0 ** n` would coerce to float
    first and raise OverflowError for n > 1023, which is a plausible number of
    discordant pairs the moment the prompt set grows past a few thousand rows;
    integer true division is correctly rounded and has no such ceiling.
    """
    if n == 0:
        return 1.0
    k = min(k, n - k)

    if n <= exact_max:
        tail = sum(math.comb(n, i) for i in range(k + 1))
        numerator, denominator = 2 * tail, 1 << n
        return 1.0 if numerator >= denominator else numerator / denominator

    # Normal approximation with a continuity correction.
    z = (abs(k - n / 2.0) - 0.5) / math.sqrt(n / 4.0)
    return min(1.0, 2.0 * _normal_sf(max(z, 0.0)))


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Paired inputs must have the same length")
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = binomial_two_sided(n, min(n10, n01))
    return {
        "a_only_correct": n10,
        "b_only_correct": n01,
        "discordant": n,
        "p_value": p,
        "p_method": "exact" if n <= EXACT_BINOMIAL_MAX_N else "normal_approximation",
        "odds_ratio": (n10 + 0.5) / (n01 + 0.5),
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _structural_components(scores: np.ndarray, m: int, n: int):
    k = scores.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        pos = scores[r, :m]
        neg = scores[r, m:]
        tx[r] = _midrank(pos)
        ty[r] = _midrank(neg)
        tz[r] = _midrank(scores[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    s = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, s


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def delong_test(
    y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float]
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    order = np.argsort(-y, kind="mergesort")
    y, a, b = y[order], a[order], b[order]
    m = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    if m == 0 or n == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "p_value": float("nan")}

    aucs, s = _structural_components(np.vstack([a, b]), m, n)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        p = 1.0 if diff == 0 else 0.0
        z = 0.0
        se = 0.0
    else:
        se = math.sqrt(var)
        z = diff / se
        p = 2 * _normal_sf(abs(z))
    # `se` is the standard error of the PAIRED difference, not of either AUC.
    # It is what an equivalence test needs: a confidence interval on the gap
    # itself.  Marginal CIs on two AUCs, which ignore that both models scored
    # the same prompts, are far wider and overlap almost regardless of the
    # truth -- so "the CIs overlap" is not evidence that the gap is small.
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": diff,
        "se": float(se),
        "z": float(z),
        "p_value": float(p),
    }


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[dict]:
    pairs = sorted(enumerate(p_values), key=lambda kv: kv[1])
    n = len(pairs)
    out = [None] * n
    running = 0.0
    for rank, (idx, p) in enumerate(pairs):
        adjusted = min(1.0, (n - rank) * p)
        running = max(running, adjusted)
        out[idx] = {
            "p_raw": float(p),
            "p_holm": float(running),
            "significant": bool(running < alpha),
        }
    return out


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> List[dict]:
    pairs = sorted(enumerate(p_values), key=lambda kv: kv[1], reverse=True)
    n = len(pairs)
    out = [None] * n
    running = 1.0
    for pos, (idx, p) in enumerate(pairs):
        rank = n - pos
        adjusted = min(running, p * n / rank)
        running = adjusted
        out[idx] = {
            "p_raw": float(p),
            "p_bh": float(adjusted),
            "significant": bool(adjusted < alpha),
        }
    return out
