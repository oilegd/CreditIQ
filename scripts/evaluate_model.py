"""Evaluate saved models on the untouched out-of-time test split.

Usage (from the repository root):
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --feature-set all
    python scripts/evaluate_model.py --synthetic
"""

from __future__ import annotations

import logging

import pandas as pd

from creditiq.cli import (
    base_parser,
    clean_all_path,
    config_from_args,
    feature_sets_from_arg,
    processed_path,
    require_file,
)
from creditiq.models.evaluation_run import run_evaluation, save_evaluation

logger = logging.getLogger("evaluate_model")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--feature-set", default=None, help="applicant | origination | all")
    parser.add_argument("--bootstrap-reps", type=int, default=None, help="Override bootstrap reps.")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if args.bootstrap_reps is not None:
        cfg = cfg.with_overrides({"evaluation": {"bootstrap_reps": args.bootstrap_reps}})

    processed = pd.read_parquet(
        require_file(processed_path(cfg), "Run: python scripts/prepare_data.py")
    )
    clean_all = pd.read_parquet(
        require_file(clean_all_path(cfg), "Run: python scripts/prepare_data.py")
    )
    for fs in feature_sets_from_arg(cfg, args.feature_set):
        outputs = run_evaluation(processed, cfg, fs, clean_all=clean_all)
        save_evaluation(outputs, cfg.path("metrics_dir"))
        if "unresolved_bounds" in outputs.tables:
            print(f"\n[{fs}] Sensitivity to excluded unresolved loans:")
            print(
                outputs.tables["unresolved_bounds"][
                    ["model", "scenario", "n", "prevalence", "roc_auc", "brier", "mean_predicted"]
                ].to_string(index=False)
            )
        tm = outputs.tables["test_metrics"]
        cols = [
            "model",
            "variant",
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "ece",
            "precision",
            "recall",
            "f1",
        ]
        print(
            f"\n[{fs}] OUT-OF-TIME TEST metrics (n={outputs.summary['test_n']:,}, "
            f"prevalence={outputs.summary['test_prevalence']:.4f}):"
        )
        print(tm[cols].to_string(index=False))
        ci = outputs.tables["bootstrap_ci"]
        print(ci.to_string(index=False))
        if "auc_diff_lightgbm_minus_logreg" in outputs.summary:
            print("AUC diff LightGBM - LogReg:", outputs.summary["auc_diff_lightgbm_minus_logreg"])
        if "random_split_benchmark" in outputs.tables:
            print(outputs.tables["random_split_benchmark"].to_string(index=False))


if __name__ == "__main__":
    main()
