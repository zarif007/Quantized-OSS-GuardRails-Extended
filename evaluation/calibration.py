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


def fit_platt(margins: Sequence[float], y_true: Sequence[int],
              max_iter: int = 100, tol: float = 1e-9) -> Dict[str, float]:
    """
    Fit p = sigmoid(a*m + b) by Newton/IRLS.  No scipy dependency.

    Two parameters, and they mean different things:

      a  scale     how sharply confidence rises with the margin.  a < 1 means
                   the model is overconfident, and 1/a is the temperature.
      b  intercept where the calibrated boundary sits.

    Only the ratio moves decisions.  `sigmoid(a*m + b) >= 0.5` reduces to
    `m >= -b/a`, so -b/a is the estimated systematic shift of the score
    distribution against the model's own fixed boundary at m = 0.
    """
    m = np.asarray(margins, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    keep = np.isfinite(m)
    m, y = m[keep], y[keep]
    if len(m) == 0 or len(np.unique(y)) < 2:
        return {"scale": float("nan"), "intercept": float("nan"),
                "location_shift": float("nan"), "temperature": float("nan"),
                "nll_before": float("nan"), "nll_after": float("nan"),
                "converged": False}

    # Fit on standardized margins.  Guard scores are well separated, so the
    # raw scale spans orders of magnitude, p saturates, and an unstandardized
    # Newton step wanders into a negative slope -- a calibration map that says
    # a higher unsafe-margin means less unsafe.  Standardizing conditions the
    # problem; the parameters are mapped back afterwards.
    centre, spread = float(np.mean(m)), float(np.std(m))
    if not np.isfinite(spread) or spread == 0:
        spread = 1.0
    z = (m - centre) / spread
    X = np.column_stack([z, np.ones_like(z)])

    def _nll(weights):
        q = np.clip(sigmoid(X @ weights), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    w = np.array([1.0, 0.0])
    converged = False
    current = _nll(w)
    for _ in range(max_iter):
        p = sigmoid(X @ w)
        sw = np.clip(p * (1 - p), 1e-8, None)
        grad = X.T @ (y - p)
        hess = X.T @ (X * sw[:, None]) + 1e-6 * np.eye(2)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break

        alpha, improved = 1.0, False
        for _ in range(40):
            candidate = w + alpha * step
            value = _nll(candidate)
            if value <= current:
                w, current, improved = candidate, value, True
                break
            alpha *= 0.5
        if not improved:
            # The line search could not improve: stuck, not converged.  These
            # are different states and only one of them is a usable fit.
            break
        if np.max(np.abs(alpha * step)) < tol:
            converged = True
            break

    # Back to the original margin scale: a*m + b == a_z*z + b_z.
    w = np.array([w[0] / spread, w[1] - w[0] * centre / spread])

    a, b = float(w[0]), float(w[1])
    # Evaluate with the back-transformed parameters against the ORIGINAL
    # margins.  `X` holds standardized margins, so `X @ w` after the transform
    # mixes the two scales and reports a fitted model as worse than the raw one.
    p_fit = np.clip(sigmoid(a * m + b), 1e-12, 1 - 1e-12)
    p_raw = np.clip(sigmoid(m), 1e-12, 1 - 1e-12)
    # A calibration map must be increasing in the margin.  A non-positive
    # slope is a failed fit, not a finding, and must not reach a table.
    valid = converged and a > 0
    if not valid:
        return {"scale": float("nan"), "intercept": float("nan"),
                "location_shift": float("nan"), "temperature": float("nan"),
                "nll_before": float("nan"), "nll_after": float("nan"),
                "converged": False}

    return {
        "scale": a,
        "intercept": b,
        # Where the model's scores sit relative to its own boundary.  This is
        # the quantity that changes decisions; the scale does not.
        "location_shift": float(-b / a) if a != 0 else float("nan"),
        "temperature": float(1.0 / a) if a != 0 else float("nan"),
        "nll_before": float(-np.mean(y * np.log(p_raw) + (1 - y) * np.log(1 - p_raw))),
        "nll_after": float(-np.mean(y * np.log(p_fit) + (1 - y) * np.log(1 - p_fit))),
        "converged": converged,
    }


def calibration_decomposition(
    y_true: Sequence[int],
    margins: Sequence[float],
    seed: int = 42,
    calibration_fraction: float = 0.5,
) -> Dict[str, float]:
    """
    Split a precision's shift into a scale part and a location part.

    This replaces a temperature-only repair, which was vacuous: for any
    temperature T > 0, `sigmoid(m/T) >= 0.5` is exactly `m >= 0`, so rescaling
    confidence cannot change a single decision at a fixed 0.5 threshold.  A
    change in safety rate at a fixed threshold is therefore *never* explained
    by temperature alone -- it must be a movement of the scores relative to
    the boundary.

    That is the point of the decomposition rather than an obstacle to it: the
    scale part is what inflates ECE, the location part is what moves the
    safety rate, and separating them says which mechanism is doing the work.
    """
    y = np.asarray(y_true, dtype=int)
    m = np.asarray(margins, dtype=np.float64)
    cal, test = split_calibration_test(y, seed=seed, calibration_fraction=calibration_fraction)

    fit = fit_platt(m[cal], y[cal])
    raw = rates_at(y[test], sigmoid(m[test]), 0.5)
    scaled = sigmoid(fit["scale"] * m[test] + fit["intercept"])
    repaired = rates_at(y[test], scaled, 0.5)

    return {
        "scale": fit["scale"],
        "intercept": fit["intercept"],
        "location_shift": fit["location_shift"],
        "temperature": fit["temperature"],
        "converged": fit["converged"],
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "default_tpr": raw["tpr"],
        "default_fpr": raw["fpr"],
        "default_flag_rate": raw["flag_rate"],
        "repaired_tpr": repaired["tpr"],
        "repaired_fpr": repaired["fpr"],
        "repaired_flag_rate": repaired["flag_rate"],
        "ece_before": expected_calibration_error(y[test], sigmoid(m[test]))["ece"],
        "ece_after": expected_calibration_error(y[test], scaled)["ece"],
    }
