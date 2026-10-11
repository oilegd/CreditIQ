"""Shared command-line helpers for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from creditiq.config import Config, load_config, setup_logging

logger = logging.getLogger(__name__)

SYNTHETIC_PATH_OVERRIDES = {
    "raw_file": "data/synthetic/synthetic_loans.csv",
    "interim_dir": "data/synthetic/interim",
    "processed_dir": "data/synthetic/processed",
    "models_dir": "data/synthetic/models",
    "reports_dir": "data/synthetic/reports",
    "figures_dir": "data/synthetic/reports/figures",
    "metrics_dir": "data/synthetic/reports/metrics",
}

PROCESSED_FILE = "loans_model.parquet"
CLEAN_ALL_FILE = "loans_clean_all_statuses.parquet"


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use SYNTHETIC data; all outputs go to data/synthetic/ (never real artifacts).",
    )
    parser.add_argument("--log-level", default=None, help="Logging level (default INFO).")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    if getattr(args, "synthetic", False):
        cfg = cfg.with_overrides({"paths": SYNTHETIC_PATH_OVERRIDES})
        logger.warning("SYNTHETIC MODE: outputs are written under data/synthetic/ only.")
    return cfg


def feature_sets_from_arg(cfg: Config, value: str | None) -> list[str]:
    if value in (None, ""):
        return [cfg["features"]["default_set"]]
    if value == "all":
        return list(cfg["features"]["sets"].keys())
    return [value]


def processed_path(cfg: Config) -> Path:
    return cfg.path("processed_dir") / PROCESSED_FILE


def clean_all_path(cfg: Config) -> Path:
    return cfg.path("interim_dir") / CLEAN_ALL_FILE


def require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}\n-> {hint}")
    return path


def ensure_synthetic_source(path: Path, n: int, seed: int) -> None:
    """Write a SYNTHETIC CSV source (only used with ``--synthetic``) if it does not exist."""
    from creditiq.data.synthetic import make_synthetic_loans

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_loans(n, seed).to_csv(path, index=False)
        logger.warning("Wrote SYNTHETIC source file %s (%s rows)", path, f"{n:,}")
