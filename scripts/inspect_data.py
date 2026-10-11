"""Inspect the raw Lending Club source before running the pipeline.

Usage (from the repository root):
    python scripts/inspect_data.py
    python scripts/inspect_data.py --raw-path "data/raw/12_13_14 Excel.xlsx" --sheet 0
    python scripts/inspect_data.py --synthetic        # synthetic demo data only
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from creditiq.cli import base_parser, config_from_args, ensure_synthetic_source
from creditiq.data.load import detect_format, file_digest, list_excel_sheets, load_raw
from creditiq.data.target import status_summary
from creditiq.data.validate import data_quality_summary
from creditiq.features.build_features import resolve_feature_spec
from creditiq.features.leakage_checks import uncatalogued_columns, validate_catalog
from creditiq.models.artifacts import write_json
from creditiq.reporting.markdown import write_data_dictionary

logger = logging.getLogger("inspect_data")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--raw-path", type=Path, default=None, help="Override paths.raw_file.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index.")
    parser.add_argument("--no-cache", action="store_true", help="Re-read the source file.")
    parser.add_argument("--synthetic-rows", type=int, default=20000)
    args = parser.parse_args()
    cfg = config_from_args(args)

    path = args.raw_path or cfg.path("raw_file")
    if args.synthetic:
        ensure_synthetic_source(path, args.synthetic_rows, cfg.seed)
    sheet = args.sheet if args.sheet is not None else cfg["data"]["excel_sheet"]
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    fmt = detect_format(path) if path.exists() else "missing"
    sheets = list_excel_sheets(path) if fmt == "excel" else ["n/a"]
    logger.info("Source: %s | format=%s | sheets=%s", path, fmt, sheets)
    df = load_raw(path, cfg.path("interim_dir"), sheet=sheet, use_cache=not args.no_cache)

    catalog = cfg["feature_catalog"]
    validate_catalog(catalog)
    quality = data_quality_summary(df)
    missing_required = [c for c in cfg["data"]["required_columns"] if c not in df.columns]
    unclassified = uncatalogued_columns(df.columns, catalog)
    status = status_summary(df, cfg["target"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(f"\nRows x columns: {df.shape[0]:,} x {df.shape[1]}")
    print("\nColumn profile:\n", quality[["dtype", "pct_missing", "n_unique"]].to_string())
    print("\nLoan status -> target mapping:\n", status.to_string())
    if missing_required:
        logger.error("MISSING REQUIRED COLUMNS: %s", missing_required)
    if unclassified:
        logger.warning("Columns without a feature-availability classification: %s", unclassified)
    if (status["target_class"] == "UNMAPPED").any():
        logger.error("Unmapped loan_status values present - update target.* in config.yaml")

    feature_sets = {
        name: resolve_feature_spec(cfg.raw, name).model_input_columns
        for name in cfg["features"]["sets"]
    }
    source = {
        "file": path.name,
        "format": fmt,
        "sheets": ", ".join(map(str, sheets)),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "sha256": file_digest(path, "sha256"),
        "md5": file_digest(path, "md5"),
    }
    reports_dir = cfg.path("reports_dir")
    write_data_dictionary(
        reports_dir / "data_dictionary.md", quality, catalog, feature_sets, source
    )
    write_json(
        {
            "source": source,
            "missing_required_columns": missing_required,
            "unclassified_columns": unclassified,
            "status_counts": status.reset_index().to_dict("records"),
            "column_profile": quality.reset_index().astype(str).to_dict("records"),
        },
        cfg.path("metrics_dir") / "data_profile.json",
    )
    logger.info("Wrote %s and data_profile.json", reports_dir / "data_dictionary.md")


if __name__ == "__main__":
    main()
