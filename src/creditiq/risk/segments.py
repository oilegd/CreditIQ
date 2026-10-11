"""Segment-level performance analysis (descriptive; not a legal fairness assessment)."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

MIN_SEGMENT_SIZE = 200


def segment_labels(df: pd.DataFrame, seg_cfg: Mapping[str, Any]) -> pd.Series:
    """Bin a numeric column (``bins``) or use a categorical column as-is."""
    col = seg_cfg["column"]
    if col == "credit_history_years":
        values = (
            (pd.to_datetime(df["issue_d"]).dt.year - pd.to_datetime(df["earliest_cr_line"]).dt.year)
            * 12
            + (
                pd.to_datetime(df["issue_d"]).dt.month
                - pd.to_datetime(df["earliest_cr_line"]).dt.month
            )
        ) / 12.0
    else:
        values = df[col]
    if "bins" in seg_cfg:
        bins = [float(b) for b in seg_cfg["bins"]]
        labels = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            labels.append(f">= {lo:,.0f}" if np.isinf(hi) else f"{lo:,.0f} - <{hi:,.0f}")
        return (
            pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, right=False)
            .astype("string")
            .fillna("Missing")
        )
    return values.astype("string").fillna("Missing")


def segment_performance(
    df: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    segments_cfg: Mapping[str, Mapping[str, Any]],
    min_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    """Per segment: size, prevalence, mean prediction, calibration gap, AUC, recall, FPR.

    Segments smaller than ``min_size`` get NaN AUC to avoid over-interpreting noise.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    flag = p >= threshold
    rows = []
    for seg_name, seg_cfg in segments_cfg.items():
        labels = segment_labels(df, seg_cfg).to_numpy()
        for value in pd.unique(labels):
            m = labels == value
            n = int(m.sum())
            ys, ps, fs = y[m], p[m], flag[m]
            pos, neg = ys == 1, ys == 0
            rows.append(
                {
                    "segment": seg_name,
                    "segment_label": seg_cfg.get("label", seg_name),
                    "group": str(value),
                    "n": n,
                    "observed_adverse_rate": float(ys.mean()) if n else np.nan,
                    "mean_predicted": float(ps.mean()) if n else np.nan,
                    "calibration_gap": float(ps.mean() - ys.mean()) if n else np.nan,
                    "roc_auc": float(roc_auc_score(ys, ps))
                    if n >= min_size and 0 < ys.sum() < n
                    else np.nan,
                    "recall": float(fs[pos].mean()) if pos.any() else np.nan,
                    "false_positive_rate": float(fs[neg].mean()) if neg.any() else np.nan,
                    "flagged_rate": float(fs.mean()) if n else np.nan,
                    "small_sample": n < min_size,
                }
            )
    out = pd.DataFrame(rows)
    out["_order"] = out["group"].map(_natural_key)
    return out.sort_values(["segment", "_order"]).drop(columns="_order").reset_index(drop=True)


def _natural_key(label: str) -> tuple[int, float, str]:
    """Order bins numerically ('< 1 year' < '5 - <10' < '>= 20'); 'Missing' last."""
    if label == "Missing":
        return (2, 0.0, label)
    if label.startswith("<"):
        return (0, -1.0, label)
    match = re.search(r"\d[\d,]*\.?\d*", label)
    if match:
        return (0, float(match.group().replace(",", "")), label)
    return (1, 0.0, label)
