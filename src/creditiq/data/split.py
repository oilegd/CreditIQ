"""Reproducible out-of-time (preferred) or stratified random splits.

Splits: train / validation / calibration / test.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLIT_ORDER = ("train", "validation", "calibration", "test")
SPLIT_COLUMN = "split"


def assign_out_of_time_split(df: pd.DataFrame, split_cfg: Mapping[str, Any]) -> pd.Series:
    """Label each row by issue month using inclusive ``YYYY-MM`` boundaries."""
    date_col = split_cfg["date_column"]
    month = pd.to_datetime(df[date_col]).dt.to_period("M")
    train_end = pd.Period(split_cfg["train_end"], freq="M")
    val_end = pd.Period(split_cfg["validation_end"], freq="M")
    cal_end = pd.Period(split_cfg["calibration_end"], freq="M")
    if not train_end < val_end < cal_end:
        raise ValueError(
            "Split boundaries must satisfy train_end < validation_end < calibration_end"
        )
    labels = np.select(
        [month <= train_end, month <= val_end, month <= cal_end],
        ["train", "validation", "calibration"],
        default="test",
    )
    labels = pd.Series(labels, index=df.index, name=SPLIT_COLUMN, dtype="object")
    labels = labels.mask(month.isna().to_numpy())
    return labels


def assign_random_split(
    df: pd.DataFrame, target: str, split_cfg: Mapping[str, Any], seed: int
) -> pd.Series:
    """Stratified random split with the configured proportions."""
    rs = split_cfg["random_split"]
    idx = df.index.to_numpy()
    y = df[target].to_numpy()
    rest_idx, test_idx = train_test_split(
        idx, test_size=rs["test_size"], stratify=y, random_state=seed
    )
    rest_y = df.loc[rest_idx, target].to_numpy()
    holdout_frac = (rs["validation_size"] + rs["calibration_size"]) / (1 - rs["test_size"])
    train_idx, hold_idx = train_test_split(
        rest_idx, test_size=holdout_frac, stratify=rest_y, random_state=seed
    )
    hold_y = df.loc[hold_idx, target].to_numpy()
    cal_frac = rs["calibration_size"] / (rs["validation_size"] + rs["calibration_size"])
    val_idx, cal_idx = train_test_split(
        hold_idx, test_size=cal_frac, stratify=hold_y, random_state=seed
    )
    labels = pd.Series(pd.NA, index=df.index, name=SPLIT_COLUMN, dtype="object")
    for name, ids in zip(SPLIT_ORDER, (train_idx, val_idx, cal_idx, test_idx), strict=True):
        labels.loc[ids] = name
    return labels


def assign_split(
    df: pd.DataFrame, target: str, split_cfg: Mapping[str, Any], seed: int
) -> pd.Series:
    strategy = split_cfg.get("strategy", "out_of_time")
    if strategy == "out_of_time":
        return assign_out_of_time_split(df, split_cfg)
    if strategy == "random":
        return assign_random_split(df, target, split_cfg, seed)
    raise ValueError(f"Unknown split strategy: {strategy}")


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{split_name: frame}`` using the ``split`` column."""
    if SPLIT_COLUMN not in df.columns:
        raise KeyError("DataFrame has no 'split' column; run scripts/prepare_data.py first.")
    labels = df[SPLIT_COLUMN].astype("string")
    frames = {name: df.loc[labels == name].reset_index(drop=True) for name in SPLIT_ORDER}
    empty = [k for k, v in frames.items() if v.empty]
    if empty:
        raise ValueError(f"Empty split(s): {empty}. Check split boundaries in config.yaml.")
    return frames


def split_summary(df: pd.DataFrame, target: str, date_col: str) -> pd.DataFrame:
    """Rows, adverse rate and date coverage per split."""
    grouped = df.groupby(df[SPLIT_COLUMN].astype("string"), observed=True)
    out = grouped.agg(
        n_loans=(target, "size"),
        n_adverse=(target, "sum"),
        adverse_rate=(target, "mean"),
        first_issue=(date_col, "min"),
        last_issue=(date_col, "max"),
    )
    return out.reindex([s for s in SPLIT_ORDER if s in out.index])


def assert_disjoint(frames: Mapping[str, pd.DataFrame], key_columns: list[str]) -> None:
    """Fail if any loan (by key columns) appears in more than one split."""
    parts = []
    for name, frame in frames.items():
        cols = [c for c in key_columns if c in frame.columns]
        part = frame[cols].copy()
        part["_split"] = name
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    cols = [c for c in combined.columns if c != "_split"]
    n_splits = combined.groupby(cols, dropna=False, observed=True)["_split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} loan key(s) appear in more than one split")
