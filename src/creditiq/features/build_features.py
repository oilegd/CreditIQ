"""Reproducible feature engineering shared by training, evaluation, notebooks and the app.

Design:
* :func:`engineer_features` is a *stateless* row-wise transformation (no statistics
  are learned), so it cannot leak information between splits.
* All learned preprocessing (clipping quantiles, imputation medians, scaling, category
  vocabularies) lives in scikit-learn transformers fitted on the training split only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from creditiq.features.leakage_checks import assert_no_leakage, check_feature_inputs

logger = logging.getLogger(__name__)

# Ordered categoricals converted to numeric ordinals instead of being one-hot encoded.
EMP_LENGTH_YEARS = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}
SUB_GRADES = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
SUB_GRADE_ORDINAL = {sg: float(i) for i, sg in enumerate(SUB_GRADES)}
ORDINAL_CATEGORICALS = {"term", "emp_length", "sub_grade"}
# Heavy-tailed non-negative inputs replaced by log1p versions.
LOG_COLUMNS = {"annual_inc": "log_annual_inc", "revol_bal": "log_revol_bal"}
# Engineered feature -> raw inputs it is derived from (used to map attributions back).
DERIVED_SOURCES = {
    "loan_to_income": ["loan_amnt", "annual_inc"],
    "revol_bal_to_income": ["revol_bal", "annual_inc"],
    "open_to_total_acc": ["open_acc", "total_acc"],
    "installment_to_monthly_income": ["installment", "annual_inc"],
    "has_delinq_history": ["mths_since_last_delinq", "delinq_2yrs"],
    "has_public_record": ["pub_rec", "mths_since_last_record"],
    "has_major_derog": ["mths_since_last_major_derog"],
    "term_months": ["term"],
    "emp_length_years": ["emp_length"],
    "sub_grade_ordinal": ["sub_grade"],
    "credit_history_months": ["earliest_cr_line"],
    **{v: [k] for k, v in LOG_COLUMNS.items()},
}


@dataclass(frozen=True)
class FeatureSpec:
    """Raw inputs for one feature set (resolved from config, including ``extends``)."""

    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    date: tuple[str, ...]
    reference_date_column: str
    allowed_categories: tuple[str, ...]
    uncertain_allowlist: tuple[str, ...] = ()
    description: str = ""
    catalog: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def input_columns(self) -> list[str]:
        """Raw columns required at inference time (order is stable)."""
        cols = [*self.numeric, *self.categorical, *self.date]
        if self.date and self.reference_date_column not in cols:
            cols.append(self.reference_date_column)
        return cols

    @property
    def model_input_columns(self) -> list[str]:
        """Raw inputs that carry information into features (excludes the reference date)."""
        return [c for c in self.input_columns if c != self.reference_date_column]

    @property
    def nominal_categoricals(self) -> list[str]:
        return [c for c in self.categorical if c not in ORDINAL_CATEGORICALS]

    def check(self) -> None:
        check_feature_inputs(
            self.input_columns,
            self.catalog,
            self.allowed_categories,
            self.uncertain_allowlist,
            reference_columns=[self.reference_date_column],
        )


def resolve_feature_spec(cfg: Mapping[str, Any], name: str | None = None) -> FeatureSpec:
    """Build a :class:`FeatureSpec` from ``cfg['features']`` resolving ``extends`` chains."""
    feats = cfg["features"]
    name = name or feats["default_set"]
    sets = feats["sets"]
    if name not in sets:
        raise KeyError(f"Unknown feature set '{name}'. Available: {sorted(sets)}")

    def _collect(set_name: str, seen: tuple[str, ...] = ()) -> dict[str, list[str]]:
        if set_name in seen:
            raise ValueError(f"Circular 'extends' in feature sets: {seen + (set_name,)}")
        entry = sets[set_name]
        base = (
            _collect(entry["extends"], seen + (set_name,))
            if entry.get("extends")
            else {"numeric": [], "categorical": [], "date": []}
        )
        for key in ("numeric", "categorical", "date"):
            for col in entry.get(key) or []:
                if col not in base[key]:
                    base[key].append(col)
        return base

    cols = _collect(name)
    entry = sets[name]
    spec = FeatureSpec(
        name=name,
        numeric=tuple(cols["numeric"]),
        categorical=tuple(cols["categorical"]),
        date=tuple(cols["date"]),
        reference_date_column=feats["reference_date_column"],
        allowed_categories=tuple(entry.get("allowed_categories", ["pre_underwriting"])),
        uncertain_allowlist=tuple(entry.get("uncertain_allowlist") or []),
        description=" ".join(str(entry.get("description", "")).split()),
        catalog=cfg["feature_catalog"],
    )
    spec.check()
    return spec


def _months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)
    months = (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)
    return months.astype("float64")


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def _as_str(series: pd.Series) -> pd.Series:
    """Categorical/str input -> object dtype with ``np.nan`` for missing (sklearn friendly)."""
    s = series.astype("string").str.strip()
    return s.astype(object).where(s.notna(), np.nan)


def engineer_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Derive the model feature frame from raw inputs (stateless; safe on any split)."""
    missing = [c for c in spec.input_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing input columns for feature set '{spec.name}': {missing}")
    out = pd.DataFrame(index=df.index)
    for col in spec.numeric:
        values = pd.to_numeric(df[col], errors="coerce").astype("float64")
        if col in LOG_COLUMNS:
            out[LOG_COLUMNS[col]] = np.log1p(values.clip(lower=0))
        else:
            out[col] = values

    num = {c: pd.to_numeric(df[c], errors="coerce") for c in spec.numeric}
    if "loan_amnt" in num and "annual_inc" in num:
        out["loan_to_income"] = _safe_ratio(num["loan_amnt"], num["annual_inc"])
    if "revol_bal" in num and "annual_inc" in num:
        out["revol_bal_to_income"] = _safe_ratio(num["revol_bal"], num["annual_inc"])
    if "open_acc" in num and "total_acc" in num:
        out["open_to_total_acc"] = _safe_ratio(num["open_acc"], num["total_acc"])
    if "installment" in num and "annual_inc" in num:
        out["installment_to_monthly_income"] = _safe_ratio(
            num["installment"], num["annual_inc"] / 12.0
        )
    if "mths_since_last_delinq" in num:
        delinq = num["mths_since_last_delinq"].notna()
        if "delinq_2yrs" in num:
            delinq |= num["delinq_2yrs"].fillna(0) > 0
        out["has_delinq_history"] = delinq.astype("float64")
    if "pub_rec" in num:
        record = num["pub_rec"].fillna(0) > 0
        if "mths_since_last_record" in num:
            record |= num["mths_since_last_record"].notna()
        out["has_public_record"] = record.astype("float64")
    if "mths_since_last_major_derog" in num:
        out["has_major_derog"] = num["mths_since_last_major_derog"].notna().astype("float64")

    if "term" in spec.categorical:
        out["term_months"] = pd.to_numeric(
            df["term"].astype("string").str.extract(r"(\d+)", expand=False), errors="coerce"
        ).astype("float64")
    if "emp_length" in spec.categorical:
        out["emp_length_years"] = _as_str(df["emp_length"]).map(EMP_LENGTH_YEARS).astype("float64")
    if "sub_grade" in spec.categorical:
        out["sub_grade_ordinal"] = _as_str(df["sub_grade"]).map(SUB_GRADE_ORDINAL).astype("float64")
    if "earliest_cr_line" in spec.date:
        hist = _months_between(df[spec.reference_date_column], df["earliest_cr_line"])
        out["credit_history_months"] = hist.where(hist >= 0)

    for col in spec.nominal_categoricals:
        out[col] = _as_str(df[col])
    return out


def feature_raw_sources(feature: str, spec: FeatureSpec) -> list[str]:
    """Raw input column(s) a model-space feature name derives from.

    Handles engineered features, log transforms, missing-value indicators
    (``missingindicator_<col>``) and one-hot columns (``<categorical>_<level>``).
    """
    if feature.startswith("missingindicator_"):
        feature = feature.removeprefix("missingindicator_")
    if feature in DERIVED_SOURCES:
        return list(DERIVED_SOURCES[feature])
    if feature in spec.input_columns:
        return [feature]
    for cat in spec.nominal_categoricals:
        if feature.startswith(f"{cat}_"):
            return [cat]
    return [feature]


def engineered_column_groups(spec: FeatureSpec) -> tuple[list[str], list[str]]:
    """(numeric feature names, nominal categorical feature names) produced for ``spec``."""
    probe = pd.DataFrame({c: pd.Series(dtype="object") for c in spec.input_columns})
    frame = engineer_features(probe, spec)
    cats = list(spec.nominal_categoricals)
    nums = [c for c in frame.columns if c not in cats]
    return nums, cats


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Pipeline step wrapping :func:`engineer_features` with an inference-schema guard."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureBuilder:
        self.spec.check()
        self.input_columns_ = list(self.spec.input_columns)
        assert_no_leakage(
            [c for c in self.input_columns_ if c != self.spec.reference_date_column],
            self.spec.catalog,
        )
        frame = engineer_features(X.iloc[:5], self.spec)
        self.feature_names_out_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame of raw inputs.")
        frame = engineer_features(X, self.spec)
        return frame[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at quantiles learned on the training data only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> QuantileClipper:
        arr = np.asarray(X, dtype="float64")
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype="float64")
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.arange(self.n_features_in_).astype(str))


def make_preprocessor(model_type: str, spec: FeatureSpec, model_cfg: Mapping[str, Any]) -> Pipeline:
    """Feature engineering + model-appropriate encoding as one sklearn Pipeline.

    * ``linear``: winsorise -> median-impute (+missing indicators) -> standardise;
      categoricals -> constant-impute -> one-hot (unknown/rare -> infrequent bucket).
    * ``tree``: numerics passed through (LightGBM handles NaN natively);
      categoricals -> ordinal codes with unknown/missing = -1 (treated as missing).
    """
    nums, cats = engineered_column_groups(spec)
    if model_type == "linear":
        q_low, q_high = model_cfg.get("clip_quantiles", [0.005, 0.995])
        numeric = Pipeline(
            [
                ("clip", QuantileClipper(q_low, q_high)),
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=50,
                        sparse_output=False,
                    ),
                ),
            ]
        )
    elif model_type == "tree":
        numeric = "passthrough"
        categorical = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")
    columns = ColumnTransformer(
        [("num", numeric, nums), ("cat", categorical, cats)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("features", FeatureBuilder(spec)), ("encode", columns)])
    pipe.set_output(transform="pandas")
    return pipe
