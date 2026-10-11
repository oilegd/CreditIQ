"""SHAP explanations (global and local) and permutation importance.

SHAP values describe how the *model* uses each feature for a prediction, in
log-odds space before calibration. They are not causal effects. Correlated
features (e.g. ``loan_amnt`` and ``loan_to_income``) can share or swap credit,
so individual attributions should be read together with their correlates.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from creditiq.features.build_features import FeatureSpec, feature_raw_sources
from creditiq.models.artifacts import ModelBundle, coerce_to_schema

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {"lightgbm", "logistic_regression"}


@dataclass
class ShapResult:
    values: np.ndarray
    base_value: float
    data: pd.DataFrame  # model-space (transformed) features

    @property
    def feature_names(self) -> list[str]:
        return list(self.data.columns)

    def to_explanation(self) -> shap.Explanation:
        return shap.Explanation(
            values=self.values,
            base_values=np.full(len(self.values), self.base_value),
            data=self.data.to_numpy(dtype="float64"),
            feature_names=self.feature_names,
        )


def _prep_and_model(bundle: ModelBundle):
    return bundle.pipeline.named_steps["prep"], bundle.pipeline.named_steps["model"]


def transform_inputs(bundle: ModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    prep, _ = _prep_and_model(bundle)
    return prep.transform(coerce_to_schema(X, bundle.input_schema))


def _binary_values(raw) -> np.ndarray:
    if isinstance(raw, list):
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    return arr[..., 1] if arr.ndim == 3 else arr


def _scalar_base(base) -> float:
    arr = np.atleast_1d(np.asarray(base, dtype="float64"))
    return float(arr[-1])


def build_explainer(bundle: ModelBundle, background: pd.DataFrame | None = None):
    """Pick an explainer suited to the model family."""
    if bundle.name not in SUPPORTED_MODELS:
        raise ValueError(f"SHAP explanations are not supported for model '{bundle.name}'")
    _, model = _prep_and_model(bundle)
    if bundle.name == "lightgbm":
        return shap.TreeExplainer(model)
    if background is None:
        raise ValueError("A background sample is required for the linear explainer.")
    bg_arr = transform_inputs(bundle, background).to_numpy(dtype="float64")
    return shap.LinearExplainer(model, shap.maskers.Independent(bg_arr, max_samples=len(bg_arr)))


def _linear_shap_from_means(bundle: ModelBundle, data: pd.DataFrame) -> ShapResult:
    """Exact linear SHAP (independent features): coef * (x - training mean), no training rows."""
    _, model = _prep_and_model(bundle)
    means = bundle.metadata["transformed_feature_means"]
    mu = np.array([means[c] for c in data.columns], dtype="float64")
    coef = model.coef_.ravel()
    values = (data.to_numpy(dtype="float64") - mu) * coef
    base = float(model.intercept_[0] + coef @ mu)
    return ShapResult(values=values, base_value=base, data=data)


def compute_shap(
    bundle: ModelBundle,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
    explainer=None,
) -> ShapResult:
    """SHAP values (log-odds contributions) for rows of raw inputs ``X``.

    ``background`` may be omitted: LightGBM uses path-dependent TreeSHAP, and logistic
    regression uses the training feature means stored in the bundle at fit time.
    """
    data = transform_inputs(bundle, X)
    if (
        explainer is None
        and background is None
        and bundle.name == "logistic_regression"
        and "transformed_feature_means" in bundle.metadata
    ):
        return _linear_shap_from_means(bundle, data)
    explainer = explainer or build_explainer(bundle, background)
    with warnings.catch_warnings():
        # shap warns that LightGBM binary output is a list; _binary_values handles both shapes.
        warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
        raw = explainer.shap_values(
            data if bundle.name == "lightgbm" else data.to_numpy(dtype="float64")
        )
    return ShapResult(
        values=_binary_values(raw), base_value=_scalar_base(explainer.expected_value), data=data
    )


def global_importance(result: ShapResult) -> pd.DataFrame:
    """Mean |SHAP| per feature, descending."""
    imp = np.abs(result.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": result.feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_instance(result: ShapResult, row: int = 0, top_k: int | None = None) -> pd.DataFrame:
    """Local contributions for one row, sorted by absolute impact."""
    contrib = pd.DataFrame(
        {
            "feature": result.feature_names,
            "model_input_value": result.data.iloc[row].to_numpy(),
            "shap_log_odds": result.values[row],
        }
    )
    contrib["direction"] = np.where(contrib["shap_log_odds"] > 0, "raises risk", "lowers risk")
    contrib = contrib.reindex(contrib["shap_log_odds"].abs().sort_values(ascending=False).index)
    contrib = contrib.reset_index(drop=True)
    return contrib.head(top_k) if top_k else contrib


def signed_importance(result: ShapResult) -> pd.DataFrame:
    """Per feature: mean |SHAP|, mean SHAP and the Spearman correlation between the
    feature value and its SHAP value (sign = typical direction of the effect)."""
    rows = []
    for j, f in enumerate(result.feature_names):
        x = result.data.iloc[:, j].to_numpy(dtype="float64")
        s = result.values[:, j]
        mask = np.isfinite(x)
        corr = (
            float(pd.Series(x[mask]).corr(pd.Series(s[mask]), method="spearman"))
            if mask.sum() > 10 and np.unique(x[mask]).size > 1 and np.unique(s[mask]).size > 1
            else float("nan")
        )
        rows.append(
            {
                "feature": f,
                "mean_abs_shap": float(np.abs(s).mean()),
                "mean_shap": float(s.mean()),
                "value_shap_spearman": corr,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def dependence_table(
    result: ShapResult, features: list[str] | None = None, n_bins: int = 20
) -> pd.DataFrame:
    """Binned SHAP dependence: per feature-value quantile bin, mean and spread of SHAP.

    Aggregated (no individual rows), suitable for storage and interactive plotting.
    """
    features = features or result.feature_names
    out = []
    for f in features:
        j = result.feature_names.index(f)
        x = result.data.iloc[:, j].to_numpy(dtype="float64")
        s = result.values[:, j]
        mask = np.isfinite(x)
        if mask.sum() < n_bins:
            continue
        frame = pd.DataFrame({"x": x[mask], "s": s[mask]})
        n_unique = frame["x"].nunique()
        if n_unique <= n_bins:
            frame["bin"] = frame["x"]
        else:
            frame["bin"] = pd.qcut(frame["x"], q=n_bins, duplicates="drop").cat.codes
        g = frame.groupby("bin").agg(
            feature_value=("x", "mean"),
            mean_shap=("s", "mean"),
            p10_shap=("s", lambda v: v.quantile(0.1)),
            p90_shap=("s", lambda v: v.quantile(0.9)),
            n=("s", "size"),
        )
        out.append(g.reset_index(drop=True).assign(feature=f))
        if (~mask).sum() >= 20:
            out.append(
                pd.DataFrame(
                    {
                        "feature_value": [np.nan],
                        "mean_shap": [float(s[~mask].mean())],
                        "p10_shap": [float(np.quantile(s[~mask], 0.1))],
                        "p90_shap": [float(np.quantile(s[~mask], 0.9))],
                        "n": [int((~mask).sum())],
                        "feature": [f],
                    }
                )
            )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def shap_by_raw_input(result: ShapResult, spec: FeatureSpec) -> pd.DataFrame:
    """Aggregate mean |SHAP| from model features back to raw inputs.

    Features derived from several inputs (e.g. ``loan_to_income``) have their attribution
    split equally among their sources - an explicit approximation for comparison with
    raw-input PFI and LIME.
    """
    acc: dict[str, float] = {}
    imp = np.abs(result.values).mean(axis=0)
    for f, v in zip(result.feature_names, imp, strict=True):
        sources = feature_raw_sources(f, spec)
        for src in sources:
            acc[src] = acc.get(src, 0.0) + float(v) / len(sources)
    return (
        pd.DataFrame({"feature": list(acc), "mean_abs_shap": list(acc.values())})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
