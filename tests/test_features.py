"""Feature engineering, missing values, unknown categories and train-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditiq.analytics.duckdb_queries import connect, parse_sql_file, run_all
from creditiq.data.split import split_frames
from creditiq.features.build_features import (
    FeatureBuilder,
    QuantileClipper,
    engineer_features,
    engineered_column_groups,
    make_preprocessor,
    resolve_feature_spec,
)


@pytest.fixture(scope="module")
def spec(cfg):
    return resolve_feature_spec(cfg.raw, "applicant")


def _row(**overrides) -> pd.DataFrame:
    base = {
        "loan_amnt": 10000,
        "annual_inc": 50000,
        "dti": 15.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "mths_since_last_delinq": np.nan,
        "mths_since_last_record": np.nan,
        "mths_since_last_major_derog": np.nan,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 5000,
        "revol_util": 0.5,
        "total_acc": 20,
        "collections_12_mths_ex_med": 0,
        "term": "36 months",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "purpose": "credit_card",
        "earliest_cr_line": pd.Timestamp("2004-03-01"),
        "issue_d": pd.Timestamp("2014-03-01"),
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_engineered_values(spec):
    f = engineer_features(_row(), spec).iloc[0]
    assert f["loan_to_income"] == pytest.approx(0.2)
    assert f["revol_bal_to_income"] == pytest.approx(0.1)
    assert f["open_to_total_acc"] == pytest.approx(0.5)
    assert f["log_annual_inc"] == pytest.approx(np.log1p(50000))
    assert f["term_months"] == 36
    assert f["emp_length_years"] == 5
    assert f["credit_history_months"] == 120
    assert f["has_delinq_history"] == 0 and f["has_public_record"] == 0
    assert "annual_inc" not in f.index  # replaced by its log


def test_indicator_features(spec):
    f = engineer_features(
        _row(delinq_2yrs=2, mths_since_last_delinq=7, pub_rec=1, mths_since_last_major_derog=30),
        spec,
    ).iloc[0]
    assert f["has_delinq_history"] == 1
    assert f["has_public_record"] == 1
    assert f["has_major_derog"] == 1


def test_invalid_values_become_nan(spec):
    f = engineer_features(
        _row(
            annual_inc=0, total_acc=0, emp_length=None, earliest_cr_line=pd.Timestamp("2015-01-01")
        ),
        spec,
    ).iloc[0]
    assert np.isnan(f["loan_to_income"])  # division by zero income
    assert np.isnan(f["open_to_total_acc"])
    assert np.isnan(f["emp_length_years"])
    assert np.isnan(f["credit_history_months"])  # credit line after reference date


def test_engineer_features_missing_input_raises(spec):
    with pytest.raises(KeyError, match="dti"):
        engineer_features(_row().drop(columns=["dti"]), spec)


def test_engineer_features_is_stateless(prepared, spec):
    data = prepared.data
    full = engineer_features(data, spec)
    part = engineer_features(data.iloc[:100], spec)
    pd.testing.assert_frame_equal(full.iloc[:100], part)


def test_quantile_clipper_learns_only_from_fit_data():
    train = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    test = pd.DataFrame({"x": [-1e9, 1e9]})
    clip = QuantileClipper(0.01, 0.99).fit(train)
    out = clip.transform(test)
    assert out[0, 0] == pytest.approx(np.quantile(train["x"], 0.01))
    assert out[1, 0] == pytest.approx(np.quantile(train["x"], 0.99))


@pytest.mark.parametrize("model_type", ["linear", "tree"])
def test_preprocessor_handles_missing_and_unknown_categories(prepared, spec, cfg, model_type):
    frames = split_frames(prepared.data)
    pre = make_preprocessor(model_type, spec, cfg["model"]["logistic_regression"])
    pre.fit(frames["train"])
    odd = _row(home_ownership="SPACESHIP", purpose=None, dti=np.nan, revol_util=np.nan)
    out = pre.transform(odd)
    assert out.shape[0] == 1
    if model_type == "linear":
        assert not out.isna().any().any()  # imputed
        onehot = [c for c in out.columns if c.startswith("home_ownership_")]
        assert out[onehot].sum(axis=1).iloc[0] in (0, 1)
    else:
        assert out["home_ownership"].iloc[0] == -1  # unknown -> missing code
        assert np.isnan(out["dti"].iloc[0])  # LightGBM handles NaN natively


def test_preprocessing_is_fit_on_train_only(prepared, spec, cfg):
    """Imputation medians must come from the training data, not the evaluation data."""
    frames = split_frames(prepared.data)
    train = frames["train"].copy()
    test = frames["test"].copy()
    test["dti"] = 1000.0  # extreme shift only in test
    pre = make_preprocessor("linear", spec, cfg["model"]["logistic_regression"]).fit(train)
    impute = pre.named_steps["encode"].named_transformers_["num"].named_steps["impute"]
    nums, _ = engineered_column_groups(spec)
    expected = (
        engineer_features(train, spec)[nums]["dti"]
        .clip(*np.nanquantile(engineer_features(train, spec)["dti"], [0.005, 0.995]))
        .median()
    )
    assert impute.statistics_[nums.index("dti")] == pytest.approx(expected)
    clipper = pre.named_steps["encode"].named_transformers_["num"].named_steps["clip"]
    assert clipper.upper_[nums.index("dti")] < 100


def test_feature_builder_output_columns_are_stable(prepared, spec):
    fb = FeatureBuilder(spec).fit(prepared.data)
    shuffled = prepared.data.sample(frac=1.0, axis=1, random_state=0)
    out = fb.transform(shuffled)
    assert list(out.columns) == fb.feature_names_out_


def test_sql_features_match_python(prepared, spec, cfg):
    con = connect({"loans_all": prepared.cleaned_all, "loans": prepared.data})
    sql = con.execute(
        parse_sql_file(cfg.path("sql_dir") / "03_feature_queries.sql")["engineered_features"]
    ).df()
    py = engineer_features(prepared.data, spec)
    for col in sql.columns:
        np.testing.assert_allclose(
            sql[col].astype(float).to_numpy(),
            py[col].astype(float).to_numpy(),
            equal_nan=True,
            err_msg=col,
        )


def test_all_sql_files_execute(prepared, cfg):
    con = connect({"loans_all": prepared.cleaned_all, "loans": prepared.data})
    results = run_all(cfg.path("sql_dir"), con)
    assert len(results) >= 15
    counts = results["01_data_quality.row_counts"].iloc[0]
    assert counts["n_resolved_loans"] == len(prepared.data)
    assert all(not df.empty for df in results.values())
