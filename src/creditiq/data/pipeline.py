"""End-to-end data preparation: validate -> clean -> target -> split."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from creditiq.config import Config
from creditiq.data.clean import CleaningLog, clean_loans
from creditiq.data.split import SPLIT_COLUMN, assign_split, split_summary
from creditiq.data.target import build_target, status_summary
from creditiq.data.validate import (
    ValidationReport,
    assert_valid,
    require_columns,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    data: pd.DataFrame
    cleaned_all: pd.DataFrame
    cleaning_log: CleaningLog
    raw_validation: ValidationReport
    clean_validation: ValidationReport
    status_counts: pd.DataFrame
    splits: pd.DataFrame


def _validate(df: pd.DataFrame, cfg: Config, numeric_ok: bool) -> ValidationReport:
    data_cfg = cfg["data"]
    numeric_cols = list(data_cfg["valid_ranges"].keys()) if numeric_ok else []
    date_cols = data_cfg["date_columns"] if numeric_ok else []
    return validate_dataframe(
        df,
        required_columns=data_cfg["required_columns"],
        numeric_columns=numeric_cols,
        date_columns=date_cols,
        valid_ranges=data_cfg["valid_ranges"],
        id_column=data_cfg.get("id_column"),
        fingerprint=data_cfg.get("duplicate_fingerprint"),
    )


def prepare_dataset(raw: pd.DataFrame, cfg: Config) -> PreparedData:
    """Run the full preparation pipeline on a raw (column-normalised) frame."""
    require_columns(raw, cfg["data"]["required_columns"])
    raw_report = _validate(raw, cfg, numeric_ok=False)
    status_counts = status_summary(raw, cfg["target"])

    cleaned, log = clean_loans(raw, cfg["data"])
    clean_report = _validate(cleaned, cfg, numeric_ok=True)
    assert_valid(clean_report)

    resolved, log = build_target(cleaned, cfg["target"], log)
    target = cfg["target"]["name"]
    resolved[SPLIT_COLUMN] = assign_split(resolved, target, cfg["split"], cfg.seed)
    n0 = len(resolved)
    resolved = resolved.loc[resolved[SPLIT_COLUMN].notna()].reset_index(drop=True)
    log.record_rows("drop_unsplittable", n0, len(resolved), "missing split date")
    resolved[SPLIT_COLUMN] = resolved[SPLIT_COLUMN].astype("category")
    splits = split_summary(resolved, target, cfg["split"]["date_column"])
    logger.info("Split summary:\n%s", splits.to_string())
    return PreparedData(
        data=resolved,
        cleaned_all=cleaned,
        cleaning_log=log,
        raw_validation=raw_report,
        clean_validation=clean_report,
        status_counts=status_counts,
        splits=splits,
    )
