"""Serializable model bundle, input schema capture and artifact persistence."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from creditiq.features.build_features import FeatureSpec
from creditiq.models.calibrate import Calibrator

logger = logging.getLogger(__name__)

BUNDLE_SUFFIX = ".joblib"
METADATA_FILE = "metadata.json"


class SchemaMismatchError(ValueError):
    """Raised when inference input does not match the training input schema."""


QUANTILE_LEVELS = np.linspace(0, 1, 21)


def capture_input_schema(train: pd.DataFrame, spec: FeatureSpec) -> dict[str, dict[str, Any]]:
    """Record the raw input contract (types, observed categories, numeric ranges) from train.

    Also stores aggregate distribution summaries (5%-step quantiles, category frequencies)
    used to sample perturbations for LIME without shipping any training rows.
    """
    schema: dict[str, dict[str, Any]] = {}
    for col in spec.numeric:
        s = pd.to_numeric(train[col], errors="coerce")
        schema[col] = {
            "kind": "numeric",
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "median": float(s.median()),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "pct_missing": float(s.isna().mean() * 100),
            "integer_valued": bool(np.allclose(s.dropna() % 1, 0)) if s.notna().any() else False,
            "quantiles": [float(v) for v in s.quantile(QUANTILE_LEVELS)],
        }
    for col in spec.categorical:
        counts = train[col].astype("string").value_counts(dropna=True)
        schema[col] = {
            "kind": "categorical",
            "categories": [str(c) for c in counts.index],
            "frequencies": [float(v) for v in (counts / counts.sum())],
            "mode": str(counts.index[0]) if len(counts) else None,
            "pct_missing": float(train[col].isna().mean() * 100),
        }
    for col in [*spec.date, spec.reference_date_column] if spec.date else []:
        s = pd.to_datetime(train[col])
        schema[col] = {
            "kind": "date",
            "min": str(s.min().date()),
            "median": str(s.quantile(0.5).date()),
            "max": str(s.max().date()),
            "quantiles": [str(d.date()) for d in s.quantile(QUANTILE_LEVELS)],
            "reference_only": col == spec.reference_date_column,
        }
    return schema


def example_application(schema: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """An illustrative, NON-REAL application built from training medians/modes."""
    row: dict[str, Any] = {}
    for col, meta in schema.items():
        if meta["kind"] == "numeric":
            row[col] = meta["median"]
        elif meta["kind"] == "categorical":
            row[col] = meta["mode"]
        else:
            row[col] = pd.Timestamp(meta["median"])
    return pd.DataFrame([row])


def coerce_to_schema(
    df: pd.DataFrame, schema: dict[str, dict[str, Any]], strict: bool = False
) -> pd.DataFrame:
    """Return a frame with exactly the training input columns, coerced to training types.

    Missing required columns raise :class:`SchemaMismatchError`. Extra columns are
    dropped (or rejected when ``strict``) so outcome fields can never reach the model.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise SchemaMismatchError(f"Inference input is missing columns: {missing}")
    extra = [c for c in df.columns if c not in schema]
    if extra and strict:
        raise SchemaMismatchError(f"Unexpected columns in inference input: {extra}")
    out = pd.DataFrame(index=df.index)
    for col, meta in schema.items():
        if meta["kind"] == "numeric":
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif meta["kind"] == "categorical":
            out[col] = df[col].astype("string")
        else:
            out[col] = pd.to_datetime(df[col], errors="coerce")
    return out


@dataclass
class ModelBundle:
    """Everything needed for standalone inference: pipeline, calibrator, threshold, schema."""

    name: str
    feature_set: str
    pipeline: Pipeline
    calibrator: Calibrator
    threshold: float
    input_schema: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return coerce_to_schema(X, self.input_schema)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model probability of an adverse outcome."""
        return self.pipeline.predict_proba(self._prepare(X))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of an adverse outcome."""
        return self.calibrator.transform(self.predict_raw(X))


def save_bundle(bundle: ModelBundle, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bundle.name}{BUNDLE_SUFFIX}"
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved %s -> %s", bundle.name, path)
    return path


def load_bundle(path: Path) -> ModelBundle:
    obj = joblib.load(path)
    if not isinstance(obj, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return obj


def environment_info() -> dict[str, str]:
    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp | datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
