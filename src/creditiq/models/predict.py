"""Standalone inference from saved artifacts (no training data required)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from creditiq.models.artifacts import (
    BUNDLE_SUFFIX,
    METADATA_FILE,
    ModelBundle,
    load_bundle,
    read_json,
)
from creditiq.risk.decision_analysis import assign_risk_bucket

logger = logging.getLogger(__name__)


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when trained artifacts are missing (run scripts/train_model.py)."""


def artifact_dir(models_dir: Path, feature_set: str) -> Path:
    return Path(models_dir) / feature_set


def load_metadata(models_dir: Path, feature_set: str) -> dict:
    path = artifact_dir(models_dir, feature_set) / METADATA_FILE
    if not path.is_file():
        raise ArtifactsNotFoundError(
            f"No trained artifacts at {path.parent}. Run: python scripts/train_model.py "
            f"--feature-set {feature_set}"
        )
    return read_json(path)


def load_model(models_dir: Path, feature_set: str, model_name: str | None = None) -> ModelBundle:
    """Load a bundle; defaults to the model selected on validation data."""
    meta = load_metadata(models_dir, feature_set)
    name = model_name or meta["selected_model"]
    path = artifact_dir(models_dir, feature_set) / f"{name}{BUNDLE_SUFFIX}"
    if not path.is_file():
        raise ArtifactsNotFoundError(f"Model artifact not found: {path}")
    return load_bundle(path)


def available_models(models_dir: Path, feature_set: str) -> list[str]:
    d = artifact_dir(models_dir, feature_set)
    return sorted(p.stem for p in d.glob(f"*{BUNDLE_SUFFIX}")) if d.is_dir() else []


def score_applications(
    bundle: ModelBundle, applications: pd.DataFrame, risk_buckets: list[dict]
) -> pd.DataFrame:
    """Return raw and calibrated adverse-outcome probabilities plus risk bucket per row."""
    p_raw = bundle.predict_raw(applications)
    p_cal = bundle.calibrator.transform(p_raw)
    out = pd.DataFrame(
        {
            "pd_raw": p_raw,
            "pd_calibrated": p_cal,
            "risk_bucket": assign_risk_bucket(p_cal, risk_buckets),
            "flagged_at_threshold": p_cal >= bundle.threshold,
        },
        index=applications.index,
    )
    if not ((out["pd_calibrated"] >= 0) & (out["pd_calibrated"] <= 1)).all():
        raise ValueError("Calibrated probabilities outside [0, 1]")
    return out
