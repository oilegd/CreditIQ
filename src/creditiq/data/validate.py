"""Schema, type, range and duplicate validation plus data-quality summaries."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when required columns are missing or have incompatible types."""


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_required: list[str] = field(default_factory=list)
    type_errors: dict[str, str] = field(default_factory=dict)
    range_violations: dict[str, int] = field(default_factory=dict)
    n_duplicates: int = 0
    duplicate_method: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.type_errors

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        return out


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Fail loudly if any required column is absent (never invent data)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Missing required columns: {missing}. Available columns: {sorted(df.columns)}"
        )


def check_types(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    date_columns: Sequence[str],
) -> dict[str, str]:
    """Return ``{column: problem}`` for columns whose dtype is incompatible with expectations."""
    errors: dict[str, str] = {}
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors[col] = f"expected numeric, found {df[col].dtype}"
    for col in date_columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            errors[col] = f"expected datetime, found {df[col].dtype}"
    return errors


def count_range_violations(
    df: pd.DataFrame, valid_ranges: Mapping[str, Sequence[float]]
) -> dict[str, int]:
    """Count non-null values falling outside ``[low, high]`` per column."""
    out: dict[str, int] = {}
    for col, (low, high) in valid_ranges.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        low, high = float(low), float(high)
        out[col] = int(((s < low) | (s > high)).sum())
    return out


def duplicate_mask(
    df: pd.DataFrame, id_column: str | None, fingerprint: Sequence[str] | None
) -> tuple[pd.Series, str]:
    """Boolean mask marking 2nd+ occurrences of a loan, and the method used."""
    if id_column and id_column in df.columns:
        return df.duplicated(subset=[id_column], keep="first"), f"id column '{id_column}'"
    cols = [c for c in (fingerprint or []) if c in df.columns]
    if cols:
        return df.duplicated(subset=cols, keep="first"), f"fingerprint of {len(cols)} columns"
    return df.duplicated(keep="first"), "full-row comparison"


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missingness, cardinality and numeric range."""
    rows = []
    for col in df.columns:
        s = df[col]
        row: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(float(s.isna().mean() * 100), 2) if len(s) else np.nan,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row.update(min=s.min(), median=s.median(), max=s.max())
        elif pd.api.types.is_datetime64_any_dtype(s):
            row.update(min=s.min(), median=pd.NaT, max=s.max())
        else:
            top = s.value_counts(dropna=True).head(1)
            row.update(min=None, median=None, max=None)
            row["top_value"] = str(top.index[0]) if len(top) else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    numeric_columns: Sequence[str],
    date_columns: Sequence[str],
    valid_ranges: Mapping[str, Sequence[float]],
    id_column: str | None = None,
    fingerprint: Sequence[str] | None = None,
) -> ValidationReport:
    """Run all checks and return a report (does not mutate ``df``)."""
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])
    report.missing_required = [c for c in required_columns if c not in df.columns]
    report.type_errors = check_types(df, numeric_columns, date_columns)
    report.range_violations = count_range_violations(df, valid_ranges)
    mask, method = duplicate_mask(df, id_column, fingerprint)
    report.n_duplicates = int(mask.sum())
    report.duplicate_method = method
    if id_column is None or id_column not in df.columns:
        report.warnings.append(
            "No loan identifier available; duplicates detected via " + method + "."
        )
    for col, n in report.range_violations.items():
        if n:
            report.warnings.append(f"{col}: {n} value(s) outside plausible range")
    return report


def assert_valid(report: ValidationReport) -> None:
    """Raise :class:`SchemaError` when a report contains blocking problems."""
    if report.missing_required:
        raise SchemaError(f"Missing required columns: {report.missing_required}")
    if report.type_errors:
        raise SchemaError(f"Column type errors: {report.type_errors}")
