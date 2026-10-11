"""LIME-style local surrogate explanations for tabular credit applications.

Follows the tabular LIME recipe of Ribeiro, Singh & Guestrin (2016, "Why should I trust
you?") with quartile discretisation, as in the reference ``lime`` package:

1. Perturbations are sampled feature-by-feature from the *training* marginal
   distributions. Only aggregate summaries saved in the model's input schema are used
   (5%-step quantiles, category frequencies, missing rates), so no training rows are needed.
2. Each sample is mapped to a binary interpretable vector: 1 if a feature falls in the
   same quartile bin / category / missing-state as the explained applicant.
3. Samples are weighted by an exponential kernel on distance in that binary space and a
   weighted ridge regression is fitted to the model's (calibrated) probability.
4. Coefficients estimate how much "being in the applicant's bin" moves the probability.

LIME is a local linear approximation that depends on the sampling distribution and seed;
``lime_stability`` quantifies that variability, and ``LimeResult.r2`` reports fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from creditiq.models.artifacts import QUANTILE_LEVELS, ModelBundle, coerce_to_schema

QUARTILE_IDX = (5, 10, 15)  # positions of the 25/50/75% quantiles in QUANTILE_LEVELS


@dataclass
class LimeResult:
    explanation: pd.DataFrame  # feature, condition, weight (probability units)
    intercept: float
    model_prediction: float
    surrogate_prediction: float
    r2: float
    n_samples: int
    kernel_width: float


def _perturbable(schema: dict) -> list[str]:
    return [c for c, m in schema.items() if not m.get("reference_only")]


def _numeric_bin(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    out = np.searchsorted(cuts, values, side="left").astype(float)
    out[np.isnan(values)] = -1  # missing is its own bin
    return out


def _sample_numeric(meta: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    q = np.asarray(meta["quantiles"], dtype=float)
    values = np.interp(rng.uniform(size=n), QUANTILE_LEVELS, q)
    if meta.get("integer_valued"):
        values = np.round(values)
    values[rng.uniform(size=n) < meta["pct_missing"] / 100.0] = np.nan
    return values


def _condition(col: str, meta: dict, value: object) -> str:
    if meta["kind"] == "categorical":
        return f"{col} is missing" if pd.isna(value) else f"{col} = {value}"
    if meta["kind"] == "numeric":
        if pd.isna(value):
            return f"{col} is missing"
        cuts = [meta["quantiles"][i] for i in QUARTILE_IDX]
        b = int(np.searchsorted(cuts, float(value), side="left"))
        lo = f"{cuts[b - 1]:,.4g} < " if b > 0 else ""
        hi = f" <= {cuts[b]:,.4g}" if b < 3 else ""
        return f"{lo}{col}{hi}"
    cuts = [meta["quantiles"][i] for i in QUARTILE_IDX]
    v = pd.Timestamp(value).date().isoformat()
    b = int(np.searchsorted(cuts, v, side="left"))
    lo = f"{cuts[b - 1]} < " if b > 0 else ""
    hi = f" <= {cuts[b]}" if b < 3 else ""
    return f"{lo}{col}{hi}"


def explain_lime(
    bundle: ModelBundle,
    instance: pd.DataFrame,
    n_samples: int = 3000,
    seed: int = 0,
    kernel_width: float | None = None,
    alpha: float = 1.0,
    calibrated: bool = True,
) -> LimeResult:
    """Explain one application (single-row raw input frame) with a LIME local surrogate."""
    if len(instance) != 1:
        raise ValueError("explain_lime expects exactly one row")
    schema = bundle.input_schema
    inst = coerce_to_schema(instance, schema).iloc[0]
    cols = _perturbable(schema)
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    samples = pd.DataFrame(index=range(n))
    z = np.zeros((n, len(cols)))
    for j, col in enumerate(cols):
        meta = schema[col]
        if meta["kind"] == "numeric":
            vals = _sample_numeric(meta, n, rng)
            cuts = np.asarray([meta["quantiles"][i] for i in QUARTILE_IDX])
            x0 = np.array([np.nan if pd.isna(inst[col]) else float(inst[col])])
            z[:, j] = _numeric_bin(vals, cuts) == _numeric_bin(x0, cuts)[0]
            samples[col] = vals
        elif meta["kind"] == "categorical":
            cats = np.asarray(meta["categories"], dtype=object)
            vals = rng.choice(
                cats, size=n, p=np.asarray(meta["frequencies"]) / np.sum(meta["frequencies"])
            )
            vals[rng.uniform(size=n) < meta["pct_missing"] / 100.0] = None
            same = (
                pd.isna(pd.Series(vals)).to_numpy()
                if pd.isna(inst[col])
                else (vals == str(inst[col]))
            )
            z[:, j] = same
            samples[col] = vals
        else:
            ords = np.array([pd.Timestamp(d).toordinal() for d in meta["quantiles"]], dtype=float)
            sampled = np.interp(rng.uniform(size=n), QUANTILE_LEVELS, ords).round().astype(int)
            dates = pd.to_datetime([pd.Timestamp.fromordinal(int(o)) for o in sampled])
            dates = dates.to_period("M").to_timestamp()
            cuts = ords[list(QUARTILE_IDX)]
            x0 = float(pd.Timestamp(inst[col]).toordinal())
            z[:, j] = np.searchsorted(cuts, sampled, "left") == np.searchsorted(cuts, x0, "left")
            samples[col] = dates
    for col, meta in schema.items():
        if meta.get("reference_only"):
            samples[col] = inst[col]  # the reference date is held fixed
    # Row 0 is the instance itself.
    for col in samples.columns:
        samples.at[0, col] = inst[col]
    z[0, :] = 1.0

    predict = bundle.predict_proba if calibrated else bundle.predict_raw
    y = predict(samples[list(schema)])
    width = kernel_width or 0.75 * np.sqrt(len(cols))
    dist = np.sqrt((1.0 - z).sum(axis=1))
    w = np.sqrt(np.exp(-(dist**2) / width**2))
    ridge = Ridge(alpha=alpha).fit(z, y, sample_weight=w)
    pred = ridge.predict(z)
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    expl = pd.DataFrame(
        {
            "feature": cols,
            "condition": [_condition(c, schema[c], inst[c]) for c in cols],
            "weight": ridge.coef_,
        }
    )
    expl = expl.reindex(expl["weight"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )
    return LimeResult(
        explanation=expl,
        intercept=float(ridge.intercept_),
        model_prediction=float(y[0]),
        surrogate_prediction=float(pred[0]),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_samples=n,
        kernel_width=float(width),
    )


def lime_stability(
    bundle: ModelBundle, instance: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2, 3, 4), **kw
) -> pd.DataFrame:
    """Mean, std and sign agreement of LIME weights across random seeds."""
    runs = [
        explain_lime(bundle, instance, seed=s, **kw).explanation.set_index("feature") for s in seeds
    ]
    weights = pd.concat([r["weight"] for r in runs], axis=1)
    out = pd.DataFrame(
        {
            "condition": runs[0]["condition"],
            "mean_weight": weights.mean(axis=1),
            "std_weight": weights.std(axis=1),
            "sign_agreement": (np.sign(weights).eq(np.sign(weights.mean(axis=1)), axis=0)).mean(
                axis=1
            ),
        }
    )
    return out.reindex(out["mean_weight"].abs().sort_values(ascending=False).index).reset_index()


def global_lime_importance(
    bundle: ModelBundle, rows: pd.DataFrame, n_samples: int = 1500, seed: int = 0
) -> pd.DataFrame:
    """Mean |LIME weight| per feature over many explained applications (aggregate view)."""
    acc: dict[str, list[float]] = {}
    fid = []
    for i in range(len(rows)):
        res = explain_lime(bundle, rows.iloc[[i]], n_samples=n_samples, seed=seed + i)
        fid.append(res.r2)
        for f, wgt in zip(res.explanation["feature"], res.explanation["weight"], strict=True):
            acc.setdefault(f, []).append(abs(wgt))
    out = pd.DataFrame(
        {"feature": list(acc), "mean_abs_lime_weight": [float(np.mean(v)) for v in acc.values()]}
    ).sort_values("mean_abs_lime_weight", ascending=False)
    out.attrs["mean_r2"] = float(np.nanmean(fid))
    return out.reset_index(drop=True)
