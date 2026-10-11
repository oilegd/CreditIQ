"""Load saved training/evaluation outputs for reports, notebooks and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from creditiq.models.artifacts import METADATA_FILE, read_json

EVAL_TABLES = (
    "test_metrics",
    "bootstrap_ci",
    "calibration_tables",
    "roc_curves",
    "pr_curves",
    "sensitivity_36_month",
    "metrics_by_quarter",
    "segment_performance",
    "decision_scenarios",
    "decision_sensitivity",
    "risk_buckets",
    "decision_curve",
    "random_split_benchmark",
    "score_distribution",
    "unresolved_bounds",
    "shap_global_importance",
    "explainer_agreement",
)


def results_available(models_dir: Path, metrics_dir: Path, feature_set: str) -> bool:
    return (models_dir / feature_set / METADATA_FILE).is_file() and (
        metrics_dir / feature_set / "evaluation_summary.json"
    ).is_file()


def load_results(models_dir: Path, metrics_dir: Path, feature_set: str) -> dict[str, Any]:
    """Return metadata, summaries and all CSV tables for one feature set.

    Every CSV under ``reports/metrics/<feature_set>/`` is loaded under its file stem;
    expected tables that are missing default to an empty DataFrame.
    """
    mdir = models_dir / feature_set
    edir = metrics_dir / feature_set
    out: dict[str, Any] = {
        "metadata": read_json(mdir / METADATA_FILE),
        "summary": read_json(edir / "evaluation_summary.json"),
        "validation_metrics": pd.read_csv(mdir / "validation_metrics.csv"),
    }
    sel = out["metadata"]["selected_model"]
    cal_path = mdir / f"calibration_selection_{sel}.csv"
    out["calibration_selection"] = pd.read_csv(cal_path) if cal_path.is_file() else pd.DataFrame()
    for name in EVAL_TABLES:
        out[name] = pd.DataFrame()
    for path in sorted(edir.glob("*.csv")):
        out[path.stem] = pd.read_csv(path)
    return out


def load_sql_results(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    sql_dir = metrics_dir / "sql"
    if not sql_dir.is_dir():
        return {}
    return {p.stem: pd.read_csv(p) for p in sorted(sql_dir.glob("*.csv"))}


def tda_cloud_path(processed_dir: Path, feature_set: str, split: str) -> Path:
    """Local (git-ignored) point-cloud sample used by the TDA page."""
    return processed_dir / "tda" / f"{feature_set}_{split}.parquet"
