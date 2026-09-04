from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.statistical_tests import mcnemar_exact


def agreement_matrix(combined: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    frames = {}
    for model in models:
        sub = combined[combined["model"] == model].set_index("prompt_id")
        frames[model] = sub["prediction"]
    matrix = pd.DataFrame(frames).dropna()
    truth = (
        combined.drop_duplicates("prompt_id").set_index("prompt_id")["ground_truth"]
    )
    matrix.insert(0, "ground_truth", truth.reindex(matrix.index))
    matrix["n_unsafe_votes"] = (matrix[models] == "unsafe").sum(axis=1)
    matrix["unanimous"] = matrix["n_unsafe_votes"].isin([0, len(models)])
    return matrix.reset_index()


def flip_table(combined: pd.DataFrame, reference: str, target: str) -> pd.DataFrame:
    ref = combined[combined["model"] == reference].set_index("prompt_id")
    tgt = combined[combined["model"] == target].set_index("prompt_id")
    shared = ref.index.intersection(tgt.index)
    ref, tgt = ref.loc[shared], tgt.loc[shared]

    rows = pd.DataFrame(
        {
            "prompt_id": shared,
            "ground_truth": ref["ground_truth"].values,
            "ref_prediction": ref["prediction"].values,
            "target_prediction": tgt["prediction"].values,
        }
    )
    if "p_unsafe" in ref.columns:
        rows["ref_p_unsafe"] = ref["p_unsafe"].values
        rows["target_p_unsafe"] = tgt["p_unsafe"].values
        rows["delta_p"] = rows["target_p_unsafe"] - rows["ref_p_unsafe"]
        rows["ref_distance_from_threshold"] = (rows["ref_p_unsafe"] - 0.5).abs()
    if "margin" in ref.columns:
        rows["ref_margin"] = ref["margin"].values
        rows["target_margin"] = tgt["margin"].values
        rows["delta_margin"] = rows["target_margin"] - rows["ref_margin"]

    rows["flipped"] = rows["ref_prediction"] != rows["target_prediction"]
    rows["direction"] = np.where(
        ~rows["flipped"],
        "none",
        np.where(
            (rows["ref_prediction"] == "safe") & (rows["target_prediction"] == "unsafe"),
            "safe_to_unsafe",
            "unsafe_to_safe",
        ),
    )
    rows["reference"] = reference
    rows["target"] = target
    return rows


def flip_summary(flips: pd.DataFrame) -> Dict[str, float]:
    total = len(flips)
    out = {
        "reference": flips["reference"].iloc[0] if total else None,
        "target": flips["target"].iloc[0] if total else None,
        "n": total,
        "n_flipped": int(flips["flipped"].sum()),
        "flip_rate": float(flips["flipped"].mean()) if total else 0.0,
        "safe_to_unsafe": int((flips["direction"] == "safe_to_unsafe").sum()),
        "unsafe_to_safe": int((flips["direction"] == "unsafe_to_safe").sum()),
    }
    if "delta_margin" in flips.columns:
        out["mean_delta_margin"] = float(flips["delta_margin"].mean())
        out["mean_delta_margin_flipped"] = float(
            flips.loc[flips["flipped"], "delta_margin"].mean()
        ) if out["n_flipped"] else float("nan")
    correct_ref = flips["ref_prediction"] == flips["ground_truth"]
    correct_tgt = flips["target_prediction"] == flips["ground_truth"]
    out.update(mcnemar_exact(correct_tgt.values, correct_ref.values))
    return out


def flip_rate_by_distance(flips: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if "ref_distance_from_threshold" not in flips.columns:
        return pd.DataFrame()
    df = flips.dropna(subset=["ref_distance_from_threshold"]).copy()
    if df.empty:
        return pd.DataFrame()
    edges = np.linspace(0, 0.5, n_bins + 1)
    df["bin"] = pd.cut(df["ref_distance_from_threshold"], bins=edges, include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("flipped", "size"),
        n_flipped=("flipped", "sum"),
        flip_rate=("flipped", "mean"),
        mean_distance=("ref_distance_from_threshold", "mean"),
    ).reset_index()
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def borderline_subset(
    combined: pd.DataFrame, models: List[str], margin_window: float = 1.0
) -> pd.DataFrame:
    matrix = agreement_matrix(combined, models)
    disputed = set(matrix.loc[~matrix["unanimous"], "prompt_id"])

    low_margin = set()
    if "margin" in combined.columns:
        agg = combined.groupby("prompt_id")["margin"].apply(lambda s: s.abs().min())
        low_margin = set(agg[agg <= margin_window].index)

    ids = disputed | low_margin
    out = combined[combined["prompt_id"].isin(ids)].copy()
    out["borderline_reason"] = np.where(
        out["prompt_id"].isin(disputed & low_margin),
        "disputed+low_margin",
        np.where(out["prompt_id"].isin(disputed), "disputed", "low_margin"),
    )
    return out


def flip_report(combined: pd.DataFrame, models: List[str], reference: Optional[str] = None):
    ref = reference or models[0]
    summaries, all_flips, distance_tables = [], [], []
    for target in models:
        if target == ref:
            continue
        flips = flip_table(combined, ref, target)
        if flips.empty:
            continue
        all_flips.append(flips)
        summaries.append(flip_summary(flips))
        dist = flip_rate_by_distance(flips)
        if not dist.empty:
            dist["reference"] = ref
            dist["target"] = target
            distance_tables.append(dist)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_flips, ignore_index=True) if all_flips else pd.DataFrame(),
        pd.concat(distance_tables, ignore_index=True) if distance_tables else pd.DataFrame(),
    )
