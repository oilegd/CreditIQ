"""Feature-availability audit and hard leakage guards for training and inference."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

CATEGORIES = (
    "pre_underwriting",
    "at_origination",
    "outcome_derived",
    "sensitive",
    "uncertain",
    "identifier_metadata",
)
NEVER_ALLOWED_CATEGORIES = frozenset({"outcome_derived", "sensitive", "identifier_metadata"})

# Defence in depth: known Lending Club post-origination / outcome fields. These are
# rejected even if someone mis-classifies them in config.yaml.
KNOWN_LEAKAGE_COLUMNS = frozenset(
    {
        "loan_status",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "out_prncp",
        "out_prncp_inv",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "pymnt_plan",
        "funded_amnt_inv",
        "hardship_flag",
        "debt_settlement_flag",
        "settlement_status",
        "is_adverse",
        "split",
    }
)


class LeakageError(ValueError):
    """Raised when a forbidden column would enter training or inference."""


def forbidden_columns(catalog: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Columns that may never be model inputs, from the catalog plus the hard-coded list."""
    from_catalog = {
        col for col, meta in catalog.items() if meta["category"] in NEVER_ALLOWED_CATEGORIES
    }
    return from_catalog | set(KNOWN_LEAKAGE_COLUMNS)


def validate_catalog(catalog: Mapping[str, Mapping[str, Any]]) -> None:
    bad = {c: m.get("category") for c, m in catalog.items() if m.get("category") not in CATEGORIES}
    if bad:
        raise ValueError(f"Invalid feature_catalog categories: {bad}")
    missing_rationale = [c for c, m in catalog.items() if not m.get("rationale")]
    if missing_rationale:
        raise ValueError(f"feature_catalog entries without rationale: {missing_rationale}")


def uncatalogued_columns(columns: Iterable[str], catalog: Mapping[str, Any]) -> list[str]:
    """Source columns that have no availability classification."""
    return sorted(c for c in columns if c not in catalog)


def check_feature_inputs(
    input_columns: Iterable[str],
    catalog: Mapping[str, Mapping[str, Any]],
    allowed_categories: Iterable[str],
    uncertain_allowlist: Iterable[str] = (),
    reference_columns: Iterable[str] = (),
) -> None:
    """Validate a feature set's raw inputs against the availability audit.

    ``reference_columns`` (e.g. ``issue_d``) may be used only as a date reference for
    derived features; they are never passed to a model directly.
    """
    allowed = set(allowed_categories)
    allowlist = set(uncertain_allowlist)
    forbidden = forbidden_columns(catalog)
    problems: list[str] = []
    for col in input_columns:
        if col in reference_columns:
            continue
        if col in forbidden:
            problems.append(f"{col}: forbidden (leakage/sensitive/identifier)")
            continue
        meta = catalog.get(col)
        if meta is None:
            problems.append(f"{col}: not classified in feature_catalog")
            continue
        category = meta["category"]
        if category == "uncertain" and col in allowlist:
            logger.debug("Using uncertain-availability column %s (allow-listed)", col)
            continue
        if category not in allowed:
            problems.append(f"{col}: category '{category}' not allowed for this feature set")
    if problems:
        raise LeakageError("Feature-set leakage check failed:\n  " + "\n  ".join(problems))


def assert_no_leakage(columns: Iterable[str], catalog: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise if any of ``columns`` is a known outcome-derived/sensitive/identifier field."""
    hits = sorted(set(columns) & forbidden_columns(catalog))
    if hits:
        raise LeakageError(f"Forbidden columns present in model input: {hits}")


def audit_table(
    catalog: Mapping[str, Mapping[str, Any]],
    feature_sets: Mapping[str, Iterable[str]],
    source_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per catalogued column with category, rationale and feature-set membership."""
    sets = {name: set(cols) for name, cols in feature_sets.items()}
    present = set(source_columns) if source_columns is not None else None
    rows = []
    for col, meta in catalog.items():
        row = {
            "column": col,
            "category": meta["category"],
            "rationale": meta["rationale"],
        }
        if present is not None:
            row["in_source"] = col in present
        for name, cols in sets.items():
            row[f"used_in_{name}"] = col in cols
        rows.append(row)
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return (
        pd.DataFrame(rows)
        .assign(_o=lambda d: d["category"].map(order))
        .sort_values(["_o", "column"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def univariate_auc_screen(
    df: pd.DataFrame, target: str, columns: Iterable[str], min_non_null: int = 100
) -> pd.DataFrame:
    """Empirical leakage screen: |AUC - 0.5| of each numeric/date column alone vs. the target.

    Outcome-derived fields typically separate classes almost perfectly; this table makes
    that visible and supports the audit's exclusion decisions. It is descriptive only.
    """
    y = df[target].to_numpy()
    rows = []
    for col in columns:
        if col not in df.columns or col == target:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            x = s.astype("int64").astype("float64").where(s.notna())
        elif pd.api.types.is_numeric_dtype(s):
            x = s.astype("float64")
        else:
            continue
        mask = x.notna().to_numpy()
        if mask.sum() < min_non_null or len(np.unique(y[mask])) < 2:
            continue
        auc = roc_auc_score(y[mask], x.to_numpy()[mask])
        rows.append(
            {
                "column": col,
                "n_non_null": int(mask.sum()),
                "auc": float(auc),
                "separation": float(abs(auc - 0.5) * 2),
            }
        )
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)
