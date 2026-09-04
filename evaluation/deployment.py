from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.threshold_analysis import auroc, rates_at, to_labels, tpr_at_fpr


def _median_latency(group: pd.DataFrame) -> float:
    """
    Median, not mean: these runs share a cloud host, where one scheduling
    hiccup moves a mean and leaves a median alone.
    """
    if "latency_sec" not in group.columns:
        return float("nan")
    return float(np.nanmedian(group["latency_sec"].values))


def safety_per_gb(df: pd.DataFrame, memory_col: str = "total_memory_mb",
                  target_fpr: float = 0.05) -> pd.DataFrame:
    """
    Safety per unit memory at a fixed operating point.

    `memory_col` defaults to the measured footprint (VRAM + host on CUDA).
    Pass `weights_mb` for the hardware-independent view, which is the right
    x-axis when the question is "what does this quantization cost to ship"
    rather than "what did it cost on this pod".
    """
    rows = []
    for model, group in df.groupby("model"):
        y = to_labels(group["ground_truth"].values)
        scores = group["p_unsafe"].values if "p_unsafe" in group.columns else \
            (group["prediction"] == "unsafe").astype(float).values
        point = tpr_at_fpr(y, scores, target_fpr)
        memory_gb = float(group[memory_col].iloc[0]) / 1024.0 if memory_col in group.columns else float("nan")
        weights_gb = float(group["weights_mb"].iloc[0]) / 1024.0 if "weights_mb" in group.columns else float("nan")
        latency = _median_latency(group)
        rows.append({
            "model": model,
            "tpr_at_target_fpr": point["tpr"],
            "target_fpr": target_fpr,
            "memory_gb": memory_gb,
            "weights_gb": weights_gb,
            "backend": group["backend"].iloc[0] if "backend" in group.columns else None,
            "gpu_name": group["gpu_name"].iloc[0] if "gpu_name" in group.columns else None,
            "latency_median_sec": latency,
            "latency_sec": latency,  # legacy alias
            "auroc": auroc(y, scores),
            "safety_per_gb": point["tpr"] / memory_gb if memory_gb and memory_gb > 0 else float("nan"),
            "ges_tf": point["tpr"] / (latency * memory_gb)
            if latency and memory_gb and latency > 0 and memory_gb > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("safety_per_gb", ascending=False)


def pareto_front(df: pd.DataFrame, x: str = "memory_gb", y: str = "tpr_at_target_fpr") -> pd.DataFrame:
    work = df.dropna(subset=[x, y]).sort_values(x).reset_index(drop=True)
    keep, best = [], -np.inf
    for _, row in work.iterrows():
        if row[y] > best:
            keep.append(True)
            best = row[y]
        else:
            keep.append(False)
    work["on_pareto_front"] = keep
    return work


def iso_memory_comparison(df: pd.DataFrame, budgets_gb: List[float],
                          target_fpr: float = 0.05) -> pd.DataFrame:
    table = safety_per_gb(df, target_fpr=target_fpr)
    rows = []
    for budget in budgets_gb:
        eligible = table[table["memory_gb"] <= budget]
        if eligible.empty:
            rows.append({"budget_gb": budget, "best_model": None, "tpr_at_target_fpr": float("nan")})
            continue
        best = eligible.loc[eligible["tpr_at_target_fpr"].idxmax()]
        rows.append({
            "budget_gb": budget,
            "best_model": best["model"],
            "tpr_at_target_fpr": best["tpr_at_target_fpr"],
            "memory_gb": best["memory_gb"],
            "n_eligible": len(eligible),
        })
    return pd.DataFrame(rows)


def uncertainty_cascade(
    combined: pd.DataFrame,
    fast_model: str,
    reference_model: str,
    band_widths: List[float] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    band_widths = band_widths or [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    fast = combined[combined["model"] == fast_model].set_index("prompt_id")
    ref = combined[combined["model"] == reference_model].set_index("prompt_id")
    shared = fast.index.intersection(ref.index)
    if len(shared) == 0:
        return pd.DataFrame()
    fast, ref = fast.loc[shared], ref.loc[shared]

    y = to_labels(fast["ground_truth"].values)
    p_fast = fast["p_unsafe"].values
    p_ref = ref["p_unsafe"].values
    distance = np.abs(p_fast - threshold)

    fast_latency = _median_latency(fast)
    ref_latency = _median_latency(ref)
    fast_mem = float(fast["total_memory_mb"].iloc[0]) / 1024 if "total_memory_mb" in fast.columns else np.nan
    ref_mem = float(ref["total_memory_mb"].iloc[0]) / 1024 if "total_memory_mb" in ref.columns else np.nan

    baseline_fast = rates_at(y, p_fast, threshold)
    baseline_ref = rates_at(y, p_ref, threshold)

    rows = []
    for band in band_widths:
        escalate = distance < band
        blended = np.where(escalate, p_ref, p_fast)
        point = rates_at(y, blended, threshold)
        frac = float(escalate.mean())
        rows.append({
            "fast_model": fast_model,
            "reference_model": reference_model,
            "band_width": band,
            "fraction_escalated": frac,
            "tpr": point["tpr"],
            "fpr": point["fpr"],
            "balanced_accuracy": point["balanced_accuracy"],
            "auroc": auroc(y, blended),
            "tpr_gain_vs_fast": point["tpr"] - baseline_fast["tpr"],
            "fpr_change_vs_fast": point["fpr"] - baseline_fast["fpr"],
            "tpr_gap_to_reference": baseline_ref["tpr"] - point["tpr"],
            "expected_latency_sec": fast_latency + frac * ref_latency,
            "mean_latency_sec": fast_latency + frac * ref_latency,  # legacy alias
            "peak_memory_gb": max(fast_mem, ref_mem) if frac > 0 else fast_mem,
        })
    return pd.DataFrame(rows)


def ensemble_at_iso_memory(combined: pd.DataFrame, members: List[str],
                           threshold: float = 0.5) -> Dict[str, float]:
    frames = []
    for model in members:
        sub = combined[combined["model"] == model].set_index("prompt_id")
        frames.append(sub["p_unsafe"].rename(model))
    matrix = pd.concat(frames, axis=1).dropna()
    if matrix.empty:
        return {}
    truth = combined.drop_duplicates("prompt_id").set_index("prompt_id")["ground_truth"]
    y = to_labels(truth.reindex(matrix.index).values)

    mean_scores = matrix.mean(axis=1).values
    max_scores = matrix.max(axis=1).values
    total_gb = sum(
        float(combined[combined["model"] == m]["total_memory_mb"].iloc[0]) / 1024
        for m in members
        if "total_memory_mb" in combined.columns
    )
    return {
        "members": "+".join(members),
        "n": len(matrix),
        "total_memory_gb": total_gb,
        "auroc_mean_vote": auroc(y, mean_scores),
        "auroc_max_vote": auroc(y, max_scores),
        "tpr_mean_vote": rates_at(y, mean_scores, threshold)["tpr"],
        "fpr_mean_vote": rates_at(y, mean_scores, threshold)["fpr"],
        "tpr_max_vote": rates_at(y, max_scores, threshold)["tpr"],
        "fpr_max_vote": rates_at(y, max_scores, threshold)["fpr"],
    }
