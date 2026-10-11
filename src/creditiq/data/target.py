"""Target construction: map loan statuses to an adverse-outcome label; drop unresolved loans."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from creditiq.data.clean import CleaningLog

logger = logging.getLogger(__name__)


class UnknownStatusError(ValueError):
    """Raised when the data contains a loan status not covered by the target definition."""


def status_summary(df: pd.DataFrame, target_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Count loans per status and the target class each status maps to."""
    status_col = target_cfg["status_column"]
    counts = df[status_col].astype("string").value_counts(dropna=False)
    mapping = {s: "positive (adverse)" for s in target_cfg["positive_statuses"]}
    mapping |= {s: "negative" for s in target_cfg["negative_statuses"]}
    mapping |= {s: "excluded (unresolved)" for s in target_cfg["excluded_statuses"]}
    out = counts.rename("n_loans").to_frame()
    out["share_pct"] = (out["n_loans"] / out["n_loans"].sum() * 100).round(2)
    out["target_class"] = [mapping.get(str(s), "UNMAPPED") for s in out.index]
    out.index.name = "loan_status"
    return out


def build_target(
    df: pd.DataFrame, target_cfg: Mapping[str, Any], log: CleaningLog | None = None
) -> tuple[pd.DataFrame, CleaningLog]:
    """Return resolved loans only, with a binary ``target_cfg['name']`` column.

    * positive: adverse resolved outcome (e.g. Charged Off / Default)
    * negative: Fully Paid
    * excluded: unresolved at the data snapshot (Current, Late, In Grace Period)

    Any status not listed in the configuration raises :class:`UnknownStatusError`
    rather than being silently assigned to a class.
    """
    log = log or CleaningLog()
    status_col = target_cfg["status_column"]
    if status_col not in df.columns:
        raise KeyError(f"Status column '{status_col}' not found.")
    positive = set(target_cfg["positive_statuses"])
    negative = set(target_cfg["negative_statuses"])
    excluded = set(target_cfg["excluded_statuses"])
    overlap = (positive & negative) | (positive & excluded) | (negative & excluded)
    if overlap:
        raise ValueError(f"Statuses assigned to more than one class: {sorted(overlap)}")

    status = df[status_col].astype("string").str.strip()
    known = positive | negative | excluded
    unknown = sorted(set(status.dropna().unique()) - known)
    if unknown:
        raise UnknownStatusError(
            f"Unmapped loan_status values {unknown}. Add them to target.* in config.yaml."
        )

    n0 = len(df)
    keep = status.isin(positive | negative)
    out = df.loc[keep].copy()
    out[target_cfg["name"]] = status[keep].isin(positive).astype("int8").to_numpy()
    log.record_rows(
        "exclude_unresolved_status",
        n0,
        len(out),
        "unresolved at snapshot: " + ", ".join(sorted(excluded)),
    )
    logger.info(
        "Target '%s': %s resolved loans, adverse rate %.4f",
        target_cfg["name"],
        f"{len(out):,}",
        out[target_cfg["name"]].mean() if len(out) else float("nan"),
    )
    return out.reset_index(drop=True), log
