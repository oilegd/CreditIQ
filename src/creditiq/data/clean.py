"""Cleaning: type coercion, value normalisation, invalid-value handling and row-removal tracking."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditiq.data.validate import duplicate_mask

logger = logging.getLogger(__name__)

EMP_LENGTH_CANONICAL = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]
HOME_OWNERSHIP_MAP = {"NONE": "OTHER", "ANY": "OTHER"}


@dataclass
class CleaningLog:
    """Audit trail of row removals and value-level fixes."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    value_fixes: dict[str, int] = field(default_factory=dict)

    def record_rows(self, step: str, before: int, after: int, reason: str) -> None:
        self.steps.append(
            {
                "step": step,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "reason": reason,
            }
        )
        logger.info("%s: removed %s rows (%s)", step, f"{before - after:,}", reason)

    def record_fix(self, name: str, count: int) -> None:
        if count:
            self.value_fixes[name] = self.value_fixes.get(name, 0) + int(count)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            s = df[col].astype("string").str.strip()
            df[col] = s.mask(s == "")
    return df


def parse_dates(df: pd.DataFrame, columns: Sequence[str], log: CleaningLog) -> pd.DataFrame:
    """Convert date columns to month-start timestamps. Supports Excel dates and 'Dec-2011' text."""
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            raw = s.astype("string")
            parsed = pd.to_datetime(raw, format="%b-%Y", errors="coerce")
            fallback = parsed.isna() & raw.notna()
            if fallback.any():
                parsed = parsed.where(
                    ~fallback, pd.to_datetime(raw.where(fallback), errors="coerce")
                )
            log.record_fix(
                f"{col}: unparseable dates set to NaT", int((parsed.isna() & raw.notna()).sum())
            )
            s = parsed
        df[col] = s.dt.to_period("M").dt.to_timestamp()
    return df


def parse_percent(series: pd.Series) -> pd.Series:
    """Return a float fraction. Numeric input is assumed to already be a fraction;
    strings such as ``"12.4%"`` are divided by 100."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip()
    has_pct = text.str.endswith("%").fillna(False)
    numbers = pd.to_numeric(text.str.rstrip("%").str.strip(), errors="coerce").astype("float64")
    return numbers.where(~has_pct, numbers / 100.0)


def normalize_term(series: pd.Series) -> pd.Series:
    """``' 36 months'`` / ``36`` -> ``'36 months'``."""
    months = series.astype("string").str.extract(r"(\d+)", expand=False)
    return (months + " months").astype("string")


def normalize_emp_length(series: pd.Series) -> pd.Series:
    """Map employment-length variants onto the canonical Lending Club labels; others -> NA."""

    def _canon(value: object) -> object:
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        text = str(value).strip().lower()
        if text in {"", "n/a", "na", "nan", "none"}:
            return pd.NA
        if text.startswith("<"):
            return "< 1 year"
        match = re.search(r"(\d+)", text)
        if not match:
            return pd.NA
        years = int(match.group(1))
        if years >= 10:
            return "10+ years"
        if years == 0:
            return "< 1 year"
        return "1 year" if years == 1 else f"{years} years"

    return series.map(_canon).astype("string")


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str], log: CleaningLog) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue
        before_missing = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        log.record_fix(
            f"{col}: non-numeric values set to NaN", int(df[col].isna().sum()) - before_missing
        )
    return df


def apply_valid_ranges(
    df: pd.DataFrame, valid_ranges: Mapping[str, Sequence[float]], log: CleaningLog
) -> pd.DataFrame:
    """Set implausible values to NaN (keeps rows; models handle missingness explicitly)."""
    for col, (low, high) in valid_ranges.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        low, high = float(low), float(high)
        bad = (df[col] < low) | (df[col] > high)
        n_bad = int(bad.sum())
        if n_bad:
            df[col] = df[col].astype("float64").mask(bad)
            log.record_fix(f"{col}: outside [{low}, {high}] set to NaN", n_bad)
    return df


def optimize_dtypes(df: pd.DataFrame, max_categories: int = 60) -> pd.DataFrame:
    """Downcast integers and store low-cardinality strings as categoricals."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_string_dtype(s) and s.nunique(dropna=True) <= max_categories:
            df[col] = s.astype("category")
    return df


def clean_loans(
    df: pd.DataFrame, data_cfg: Mapping[str, Any], log: CleaningLog | None = None
) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean a raw Lending Club frame. Returns a new frame and the audit log."""
    log = log or CleaningLog()
    df = df.copy()
    df = strip_strings(df)

    n0 = len(df)
    df = df.dropna(subset=[c for c in ("loan_status", "issue_d") if c in df.columns])
    log.record_rows("drop_missing_key_fields", n0, len(df), "missing loan_status or issue_d")

    df = parse_dates(df, data_cfg.get("date_columns", []), log)
    for col in data_cfg.get("percent_columns", []):
        if col in df.columns:
            df[col] = parse_percent(df[col])
    if "term" in df.columns:
        df["term"] = normalize_term(df["term"])
    if "emp_length" in df.columns:
        df["emp_length"] = normalize_emp_length(df["emp_length"])
    if "home_ownership" in df.columns:
        upper = df["home_ownership"].astype("string").str.upper()
        log.record_fix(
            "home_ownership: NONE/ANY mapped to OTHER", int(upper.isin(HOME_OWNERSHIP_MAP).sum())
        )
        df["home_ownership"] = upper.replace(HOME_OWNERSHIP_MAP)

    numeric_cols = list(data_cfg.get("valid_ranges", {}).keys())
    df = coerce_numeric(df, numeric_cols, log)
    df = apply_valid_ranges(df, data_cfg.get("valid_ranges", {}), log)

    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        future = df["earliest_cr_line"] > df["issue_d"]
        log.record_fix("earliest_cr_line after issue_d set to NaT", int(future.sum()))
        df["earliest_cr_line"] = df["earliest_cr_line"].mask(future)

    n1 = len(df)
    mask, method = duplicate_mask(
        df, data_cfg.get("id_column"), data_cfg.get("duplicate_fingerprint")
    )
    df = df.loc[~mask]
    log.record_rows("drop_duplicates", n1, len(df), f"duplicate loans ({method})")

    drop_cols = [c for c in data_cfg.get("processed_drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped free-text/identifying columns from processed data: %s", drop_cols)
    df = optimize_dtypes(df).reset_index(drop=True)
    return df, log
