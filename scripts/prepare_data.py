"""Clean, validate, build the target, split, audit leakage and run SQL analytics.

Usage (from the repository root):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --synthetic
"""

from __future__ import annotations

import logging
from pathlib import Path

from creditiq.analytics.duckdb_queries import connect, run_all
from creditiq.cli import (
    base_parser,
    clean_all_path,
    config_from_args,
    ensure_synthetic_source,
    processed_path,
)
from creditiq.data.load import load_raw
from creditiq.data.pipeline import prepare_dataset
from creditiq.data.split import SPLIT_COLUMN
from creditiq.features.build_features import resolve_feature_spec
from creditiq.features.leakage_checks import audit_table, univariate_auc_screen
from creditiq.models.artifacts import write_json
from creditiq.reporting.markdown import write_leakage_audit

logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--synthetic-rows", type=int, default=20000)
    args = parser.parse_args()
    cfg = config_from_args(args)

    path = args.raw_path or cfg.path("raw_file")
    if args.synthetic:
        ensure_synthetic_source(path, args.synthetic_rows, cfg.seed)
    raw = load_raw(path, cfg.path("interim_dir"), sheet=cfg["data"]["excel_sheet"])
    prepared = prepare_dataset(raw, cfg)
    target = cfg["target"]["name"]

    out_clean = clean_all_path(cfg)
    out_model = processed_path(cfg)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    prepared.cleaned_all.to_parquet(out_clean, index=False)
    prepared.data.to_parquet(out_model, index=False)
    logger.info(
        "Wrote %s (%s rows) and %s (%s rows)",
        out_clean,
        f"{len(prepared.cleaned_all):,}",
        out_model,
        f"{len(prepared.data):,}",
    )

    metrics_dir = cfg.path("metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    catalog = cfg["feature_catalog"]
    train = prepared.data[prepared.data[SPLIT_COLUMN].astype("string") == "train"]
    screen_cols = [c for c in prepared.data.columns if c not in (target, SPLIT_COLUMN)]
    screen = univariate_auc_screen(train, target, screen_cols)
    screen["category"] = screen["column"].map(lambda c: catalog.get(c, {}).get("category", "?"))
    screen.to_csv(metrics_dir / "leakage_screen.csv", index=False)

    specs = {name: resolve_feature_spec(cfg.raw, name) for name in cfg["features"]["sets"]}
    audit = audit_table(
        catalog, {n: s.model_input_columns for n, s in specs.items()}, source_columns=raw.columns
    )
    audit.to_csv(metrics_dir / "leakage_audit_table.csv", index=False)
    write_leakage_audit(
        cfg.path("reports_dir") / "leakage_audit.md",
        audit,
        screen,
        {n: s.description for n, s in specs.items()},
    )

    con = connect({"loans_all": out_clean, "loans": out_model})
    sql_dir = metrics_dir / "sql"
    sql_dir.mkdir(parents=True, exist_ok=True)
    sql_results = run_all(cfg.path("sql_dir"), con)
    for name, table in sql_results.items():
        if name == "03_feature_queries.engineered_features":
            continue  # row-level output; not persisted
        table.to_csv(sql_dir / f"{name}.csv", index=False)
    con.close()

    write_json(
        {
            "source_rows": int(len(raw)),
            "cleaned_rows_all_statuses": int(len(prepared.cleaned_all)),
            "modelling_rows": int(len(prepared.data)),
            "row_removals": prepared.cleaning_log.steps,
            "value_fixes": prepared.cleaning_log.value_fixes,
            "raw_validation": prepared.raw_validation.to_dict(),
            "clean_validation": prepared.clean_validation.to_dict(),
            "status_counts": prepared.status_counts.reset_index().to_dict("records"),
            "splits": prepared.splits.reset_index().astype(str).to_dict("records"),
        },
        metrics_dir / "data_preparation.json",
    )
    print("\nRow removals:\n", prepared.cleaning_log.to_frame().to_string(index=False))
    print("\nValue fixes:", prepared.cleaning_log.value_fixes)
    print("\nSplits:\n", prepared.splits.to_string())
    print(
        "\nTop of empirical leakage screen (train split):\n", screen.head(12).to_string(index=False)
    )


if __name__ == "__main__":
    main()
