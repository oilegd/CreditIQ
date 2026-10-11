"""Point clouds for topological analysis.

Each loan is represented by the engineered numeric features of the applicant model,
median-imputed and robustly standardised ((x - median) / IQR, clipped to +/-5) using
statistics from the TRAINING split only. Calibrated predictions and observed outcomes
are attached for colouring and diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditiq.features.build_features import (
    FeatureSpec,
    engineer_features,
    engineered_column_groups,
)
from creditiq.models.artifacts import ModelBundle

FEATURE_PREFIX = "f__"
DESCRIPTIVE = ["loan_amnt", "annual_inc", "dti", "revol_util", "purpose", "home_ownership", "term"]


def robust_scaler(train: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    nums, _ = engineered_column_groups(spec)
    f = engineer_features(train, spec)[nums]
    q1, med, q3 = f.quantile(0.25), f.median(), f.quantile(0.75)
    iqr = (q3 - q1).where((q3 - q1) > 0, f.std().where(f.std() > 0, 1.0))
    return pd.DataFrame({"median": med, "scale": iqr})


def build_point_cloud(
    rows: pd.DataFrame,
    spec: FeatureSpec,
    scaler: pd.DataFrame,
    bundle: ModelBundle,
    target: str,
    sample_size: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample ``rows`` and return standardised features + y, pd and descriptive columns."""
    if len(rows) > sample_size:
        rows = rows.sample(sample_size, random_state=seed)
    rows = rows.reset_index(drop=True)
    nums, _ = engineered_column_groups(spec)
    f = engineer_features(rows, spec)[nums]
    z = ((f.fillna(scaler["median"]) - scaler["median"]) / scaler["scale"]).clip(-5, 5)
    out = z.add_prefix(FEATURE_PREFIX)
    out["y"] = rows[target].to_numpy().astype(int)
    out["pd"] = bundle.predict_proba(rows)
    out["share_60m"] = rows["term"].astype(str).str.startswith("60").astype(float).to_numpy()
    for col in DESCRIPTIVE:
        if col in rows.columns:
            out[col] = (
                rows[col].astype(str).to_numpy()
                if col in ("purpose", "home_ownership", "term")
                else pd.to_numeric(rows[col], errors="coerce").to_numpy()
            )
    out["issue_quarter"] = pd.to_datetime(rows["issue_d"]).dt.to_period("Q").astype(str).to_numpy()
    return out


def feature_columns(cloud: pd.DataFrame) -> list[str]:
    return [c for c in cloud.columns if c.startswith(FEATURE_PREFIX)]


def feature_matrix(cloud: pd.DataFrame) -> np.ndarray:
    return cloud[feature_columns(cloud)].to_numpy(dtype="float64")


def pretty(name: str) -> str:
    return name.removeprefix(FEATURE_PREFIX)
