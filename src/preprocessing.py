from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import DATA_PATH, MONTH_ORDER, RANDOM_SEED, TARGET


def load_online_shoppers(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(bool).astype(int)
    df["Weekend"] = df["Weekend"].astype(bool).map({True: "Weekend", False: "Weekday"})
    df["MonthIndex"] = df["Month"].map(MONTH_ORDER)
    if df["MonthIndex"].isna().any():
        unknown = sorted(df.loc[df["MonthIndex"].isna(), "Month"].unique())
        raise ValueError(f"Unknown Month labels: {unknown}")
    return df


def temporal_split(df: pd.DataFrame):
    """Train on early/mid-year sessions and evaluate on late-year deployment months."""
    train_pool = df[df["MonthIndex"] <= 9].copy()
    test = df[df["MonthIndex"] >= 10].copy()
    train, valid = train_test_split(
        train_pool,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=train_pool[TARGET],
    )
    return train, valid, test


def random_reference_split(df: pd.DataFrame):
    train_valid, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df[TARGET],
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=train_valid[TARGET],
    )
    return train, valid, test


def adjacent_temporal_split(df: pd.DataFrame):
    """Use September as model-selection validation, then test on Oct-Dec."""
    train = df[df["MonthIndex"] <= 8].copy()
    valid = df[df["MonthIndex"] == 9].copy()
    test = df[df["MonthIndex"] >= 10].copy()
    return train, valid, test


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        c for c in x.columns if x[c].dtype == "object" or str(x[c].dtype) == "category"
    ]
    numeric = [c for c in x.columns if c not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("cat", encoder, categorical), ("num", "passthrough", numeric)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
