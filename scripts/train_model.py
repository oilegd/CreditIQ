"""Train Dummy, Logistic Regression and LightGBM; calibrate and save artifacts.

Usage (from the repository root):
    python scripts/train_model.py                       # default (applicant) feature set
    python scripts/train_model.py --feature-set origination
    python scripts/train_model.py --feature-set all
    python scripts/train_model.py --synthetic
"""

from __future__ import annotations

import logging

import pandas as pd

from creditiq.cli import (
    base_parser,
    config_from_args,
    feature_sets_from_arg,
    processed_path,
    require_file,
)
from creditiq.models.train import save_training_result, train_all

logger = logging.getLogger("train_model")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--feature-set", default=None, help="applicant | origination | all")
    args = parser.parse_args()
    cfg = config_from_args(args)

    path = require_file(processed_path(cfg), "Run: python scripts/prepare_data.py")
    processed = pd.read_parquet(path)
    for fs in feature_sets_from_arg(cfg, args.feature_set):
        result = train_all(processed, cfg, fs)
        out_dir = save_training_result(result, cfg.path("models_dir"), cfg)
        cols = ["roc_auc", "pr_auc", "brier", "log_loss", "fit_seconds"]
        print(f"\n[{fs}] validation metrics (selection only; test untouched):")
        print(result.validation_metrics[cols].to_string())
        print(f"[{fs}] selected model: {result.selected_model} -> {out_dir}")


if __name__ == "__main__":
    main()
