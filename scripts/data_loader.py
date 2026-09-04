import os
from typing import List, Optional

import pandas as pd

from scripts.datasets_registry import DATASETS, get_spec

NORMALIZED_DIR = "datasets/normalized"
REQUIRED_COLUMNS = {"prompt", "ground_truth"}

LEGACY_PATHS = {
    "harmbench": "datasets/harmbench/harmbench.csv",
    "xstest": "datasets/xstest/xstest.csv",
}


def _candidate_paths(name: str) -> List[str]:
    paths = [os.path.join(NORMALIZED_DIR, f"{name}.csv")]
    if name in LEGACY_PATHS:
        paths.append(LEGACY_PATHS[name])
    if name in DATASETS and DATASETS[name].local_csv:
        paths.append(DATASETS[name].local_csv)
    return paths


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset '{name}' is missing columns: {sorted(missing)}")
    bad = set(df["ground_truth"].dropna().unique()) - {"safe", "unsafe"}
    if bad:
        raise ValueError(f"Dataset '{name}' has unexpected labels: {sorted(bad)}")
    df = df.dropna(subset=["prompt", "ground_truth"]).reset_index(drop=True)
    if "category" not in df.columns:
        df["category"] = None
    if "language" not in df.columns:
        df["language"] = "en"
    if "level" not in df.columns:
        df["level"] = "prompt"
    return df


def get_dataset(name: str, subset_size: Optional[int] = None) -> pd.DataFrame:
    key = name.lower()
    for path in _candidate_paths(key):
        if os.path.exists(path):
            df = _validate(pd.read_csv(path), key)
            df["dataset"] = key
            if subset_size:
                df = stratified_subset(df, subset_size)
            return df.reset_index(drop=True)

    raise FileNotFoundError(
        f"Dataset '{name}' not found. Looked in: {_candidate_paths(key)}\n"
        f"Run: python scripts/download_datasets.py --datasets {key}"
    )


def get_datasets(names: List[str], subset_size: Optional[int] = None) -> pd.DataFrame:
    return pd.concat([get_dataset(n, subset_size) for n in names], ignore_index=True)


def get_harmbench(subset_size=None) -> pd.DataFrame:
    return get_dataset("harmbench", subset_size)


def get_xstest(subset_size=None) -> pd.DataFrame:
    return get_dataset("xstest", subset_size)


def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby("ground_truth"):
        take = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(take, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def base_rate(df: pd.DataFrame) -> float:
    return float((df["ground_truth"] == "unsafe").mean())


def describe(df: pd.DataFrame) -> str:
    counts = df["ground_truth"].value_counts().to_dict()
    return f"n={len(df)} {counts} harmful_base_rate={base_rate(df):.3f}"
