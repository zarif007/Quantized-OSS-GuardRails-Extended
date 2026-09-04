from typing import Dict, List, Sequence, Tuple

import numpy as np

from evaluation.threshold_analysis import rates_at, roc_curve

DEFAULT_BINS = 15
GOLDEN = (5.0**0.5 - 1) / 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def expected_calibration_error(
    y_true: Sequence[int], p_unsafe: Sequence[float], n_bins: int = DEFAULT_BINS
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_unsafe, dtype=np.float64)
    confidence = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0.5 else (confidence >= lo) & (confidence <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins}


def brier_score(y_true: Sequence[int], p_unsafe: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_unsafe, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    y_true: Sequence[int], p_unsafe: Sequence[float], n_bins: int = DEFAULT_BINS
) -> List[Dict[str, float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_unsafe, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return rows


def nll_at_temperature(margins: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = sigmoid(margins / temperature)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(
    margins: Sequence[float], y_true: Sequence[int], lo: float = 0.01, hi: float = 100.0
) -> Dict[str, float]:
    m = np.asarray(margins, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    keep = np.isfinite(m)
    m, y = m[keep], y[keep]

    phi = GOLDEN
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(200):
        if nll_at_temperature(m, y, c) < nll_at_temperature(m, y, d):
            b = d
        else:
            a = c
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if abs(b - a) < 1e-6:
            break
    t = (a + b) / 2
    return {
        "temperature": float(t),
        "nll_before": nll_at_temperature(m, y, 1.0),
        "nll_after": nll_at_temperature(m, y, t),
    }


def apply_temperature(margins: Sequence[float], temperature: float) -> np.ndarray:
    return sigmoid(np.asarray(margins, dtype=np.float64) / temperature)


def threshold_for_fpr(
    y_true: Sequence[int], scores: Sequence[float], target_fpr: float
) -> float:
    fpr, _, thr = roc_curve(y_true, scores)
    eligible = np.where(fpr <= target_fpr)[0]
    if eligible.size == 0:
        return float("inf")
    return float(thr[eligible[-1]])


def threshold_for_max_f1(y_true: Sequence[int], scores: Sequence[float]) -> float:
    grid = np.unique(np.asarray(scores, dtype=np.float64))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = rates_at(y_true, scores, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def split_calibration_test(
    y_true: Sequence[int], seed: int = 42, calibration_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    cal_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * calibration_fraction))
        cal_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.sort(np.array(cal_idx)), np.sort(np.array(test_idx))


def recalibrate(
    y_true: Sequence[int],
    scores: Sequence[float],
    target_fpr: float,
    seed: int = 42,
    calibration_fraction: float = 0.5,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    cal, test = split_calibration_test(y, seed=seed, calibration_fraction=calibration_fraction)

    threshold = threshold_for_fpr(y[cal], s[cal], target_fpr)
    tuned = rates_at(y[test], s[test], threshold)
    default = rates_at(y[test], s[test], 0.5)
    return {
        "target_fpr": float(target_fpr),
        "fitted_threshold": float(threshold),
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "tuned_tpr": tuned["tpr"],
        "tuned_fpr": tuned["fpr"],
        "tuned_f1": tuned["f1"],
        "tuned_balanced_accuracy": tuned["balanced_accuracy"],
        "default_tpr": default["tpr"],
        "default_fpr": default["fpr"],
        "default_f1": default["f1"],
        "default_balanced_accuracy": default["balanced_accuracy"],
    }
