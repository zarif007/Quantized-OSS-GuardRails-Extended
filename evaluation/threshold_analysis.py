from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE_RATES = [0.01, 0.05, 0.10, 0.30, 0.50, 0.62]
FPR_TARGETS = [0.01, 0.05, 0.10]


def to_labels(ground_truth: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if g == "unsafe" else 0 for g in ground_truth], dtype=int)


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) - 1 and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auprc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    area = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        area += p * (r - prev_recall)
        prev_recall = r
    return float(area)


def roc_curve(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])
    thr = np.concatenate([[np.inf], s, [-np.inf]])
    return fpr, tpr, thr


def tpr_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, scores)
    eligible = np.where(fpr <= target_fpr)[0]
    if eligible.size == 0:
        return {"tpr": 0.0, "fpr": 0.0, "threshold": float("inf")}
    i = eligible[-1]
    return {"tpr": float(tpr[i]), "fpr": float(fpr[i]), "threshold": float(thr[i])}


def confusion_at(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=np.float64) >= threshold).astype(int)
    return {
        "tp": int(np.sum((y == 1) & (pred == 1))),
        "fn": int(np.sum((y == 1) & (pred == 0))),
        "fp": int(np.sum((y == 0) & (pred == 1))),
        "tn": int(np.sum((y == 0) & (pred == 0))),
    }


def rates_at(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    c = confusion_at(y_true, scores, threshold)
    tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
    fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "threshold": float(threshold),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(1 - tpr),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float((tpr + (1 - fpr)) / 2),
        "flag_rate": float((c["tp"] + c["fp"]) / max(sum(c.values()), 1)),
        **c,
    }


def threshold_sweep(
    y_true: Sequence[int], scores: Sequence[float], grid: Sequence[float] = None
) -> List[Dict[str, float]]:
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 4)
    return [rates_at(y_true, scores, t) for t in grid]


def accuracy_at_base_rate(tpr: float, fpr: float, base_rate: float) -> float:
    return float(base_rate * tpr + (1 - base_rate) * (1 - fpr))


def base_rate_sweep(tpr: float, fpr: float, base_rates: Sequence[float] = None) -> Dict[float, float]:
    rates = base_rates if base_rates is not None else BASE_RATES
    return {float(pi): accuracy_at_base_rate(tpr, fpr, pi) for pi in rates}


def log_diagnostic_odds_ratio(tp: int, fn: int, fp: int, tn: int) -> float:
    return float(np.log(((tp + 0.5) / (fn + 0.5)) / ((fp + 0.5) / (tn + 0.5))))


def margin_summary(margins: Sequence[float]) -> Dict[str, float]:
    m = np.asarray(margins, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": int(m.size),
        "mean": float(np.mean(m)),
        "std": float(np.std(m, ddof=1)) if m.size > 1 else 0.0,
        "median": float(np.median(m)),
        "iqr": float(np.quantile(m, 0.75) - np.quantile(m, 0.25)),
    }
