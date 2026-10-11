"""Inference from saved artifacts, schema consistency, explanations and decision maths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditiq.data.split import split_frames
from creditiq.explainability.shap_analysis import compute_shap, explain_instance, global_importance
from creditiq.models.artifacts import SchemaMismatchError
from creditiq.models.evaluation_run import decision_curve, run_evaluation, save_evaluation
from creditiq.models.predict import (
    ArtifactsNotFoundError,
    available_models,
    load_model,
    score_applications,
)
from creditiq.risk.decision_analysis import (
    DecisionAssumptions,
    assign_risk_bucket,
    breakeven_pd,
    bucket_summary,
    expected_loss,
    sensitivity_analysis,
    simulate_threshold,
    threshold_scenarios,
)

BUCKETS = [
    {"name": "A", "upper": 0.1},
    {"name": "B", "upper": 0.3},
    {"name": "C", "upper": 1.000001},
]


# ----------------------------------------------------------------- inference
def test_load_selected_model_and_score(trained, prepared, cfg):
    result, models_dir = trained
    bundle = load_model(models_dir, "applicant")
    assert bundle.name == result.selected_model
    test = split_frames(prepared.data)["test"]
    scored = score_applications(bundle, test, cfg["decision"]["risk_buckets"])
    assert scored["pd_calibrated"].between(0, 1).all()
    assert scored["pd_raw"].between(0, 1).all()
    assert scored["risk_bucket"].notna().all()
    assert len(scored) == len(test)


def test_inference_needs_only_raw_application_fields(trained):
    """Standalone inference: a single hand-built application, no training data."""
    _, models_dir = trained
    bundle = load_model(models_dir, "applicant", "lightgbm")
    app = {}
    for col, meta in bundle.input_schema.items():
        if meta["kind"] == "numeric":
            app[col] = meta["median"]
        elif meta["kind"] == "categorical":
            app[col] = meta["mode"]
        else:
            app[col] = meta["median"]
    p = bundle.predict_proba(pd.DataFrame([app]))
    assert p.shape == (1,) and 0 < p[0] < 1


def test_inference_schema_matches_training(trained):
    result, _ = trained
    bundle = result.bundles["logistic_regression"]
    assert list(bundle.input_schema) == result.metadata["input_columns"]


def test_missing_column_raises(trained, prepared):
    result, _ = trained
    bundle = result.bundles["lightgbm"]
    with pytest.raises(SchemaMismatchError, match="dti"):
        bundle.predict_proba(prepared.data.head(3).drop(columns=["dti"]))


def test_unknown_categories_and_missing_values_are_scored(trained, prepared):
    result, _ = trained
    rows = prepared.data.head(4).copy()
    rows["purpose"] = rows["purpose"].astype(str)
    rows.loc[rows.index[0], "purpose"] = "time_travel"
    rows.loc[rows.index[1], "home_ownership"] = None
    rows.loc[rows.index[2], "dti"] = np.nan
    rows.loc[rows.index[3], "emp_length"] = None
    for name in ("lightgbm", "logistic_regression"):
        p = result.bundles[name].predict_proba(rows)
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_missing_artifacts_give_clear_error(tmp_path):
    with pytest.raises(ArtifactsNotFoundError, match="train_model.py"):
        load_model(tmp_path, "applicant")
    assert available_models(tmp_path, "applicant") == []


# ----------------------------------------------------------------- explanations
@pytest.mark.parametrize("name", ["lightgbm", "logistic_regression"])
def test_shap_values_are_additive(trained, prepared, name):
    result, _ = trained
    bundle = result.bundles[name]
    frames = split_frames(prepared.data)
    sample = frames["test"].head(25)
    res = compute_shap(bundle, sample, frames["train"].head(200))
    raw = bundle.predict_raw(sample)
    log_odds = np.log(raw / (1 - raw))
    np.testing.assert_allclose(res.base_value + res.values.sum(axis=1), log_odds, atol=1e-6)
    imp = global_importance(res)
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    local = explain_instance(res, 0, top_k=5)
    assert len(local) == 5 and set(local["direction"]) <= {"raises risk", "lowers risk"}


# ----------------------------------------------------------------- decision maths
@pytest.mark.parametrize("name", ["lightgbm", "logistic_regression"])
def test_shap_without_training_data(trained, prepared, name):
    """The app explains predictions from artifacts alone (no training rows)."""
    result, _ = trained
    bundle = result.bundles[name]
    frames = split_frames(prepared.data)
    sample = frames["test"].head(10)
    res = compute_shap(bundle, sample)  # no background
    raw = bundle.predict_raw(sample)
    np.testing.assert_allclose(
        res.base_value + res.values.sum(axis=1), np.log(raw / (1 - raw)), atol=1e-6
    )
    if name == "logistic_regression":
        full = compute_shap(bundle, sample, frames["train"])  # LinearExplainer, full train
        np.testing.assert_allclose(res.values, full.values, atol=1e-6)


# ----------------------------------------------------------------- decision maths
def test_expected_loss_formula():
    el = expected_loss(np.array([0.1, 0.5]), np.array([10000, 2000]), lgd=0.8)
    np.testing.assert_allclose(el, [800.0, 800.0])


def test_risk_buckets():
    out = assign_risk_bucket(np.array([0.0, 0.0999, 0.1, 0.29, 0.3, 1.0]), BUCKETS)
    assert out.tolist() == ["A", "A", "B", "B", "C", "C"]
    with pytest.raises(ValueError):
        assign_risk_bucket(np.array([0.5]), [{"name": "x", "upper": 0.4}])
    with pytest.raises(ValueError):
        assign_risk_bucket(
            np.array([0.1]), [{"name": "x", "upper": 0.4}, {"name": "y", "upper": 0.2}]
        )


def test_simulate_threshold_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.40, 0.10, 0.60])
    amount = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    a = DecisionAssumptions(lgd=0.5, ead_fraction=1.0, profit_rate_good_loan=0.1)
    r = simulate_threshold(y, p, amount, threshold=0.2, assumptions=a)
    # approved: loans 0 (good) and 2 (bad)
    assert r["approval_rate"] == 0.5
    assert r["false_approval_cost"] == pytest.approx(0.5 * 3000)
    assert r["false_rejection_opportunity_cost"] == pytest.approx(0.1 * 2000)
    assert r["profit_from_good_approvals"] == pytest.approx(0.1 * 1000)
    assert r["net_value"] == pytest.approx(100 - 1500)
    assert r["expected_loss_approved"] == pytest.approx(0.05 * 0.5 * 1000 + 0.10 * 0.5 * 3000)
    assert r["observed_bad_rate_approved"] == pytest.approx(0.5)


def test_threshold_scenarios_and_sensitivity():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    amount = rng.uniform(1000, 30000, 2000)
    a = DecisionAssumptions()
    scen = threshold_scenarios(y, p, amount, [0.1, 0.2, 0.3], a)
    assert scen.iloc[0]["scenario"] == "approve all" and scen.iloc[0]["approval_rate"] == 1.0
    assert scen["approval_rate"].iloc[1:].is_monotonic_increasing
    sens = sensitivity_analysis(y, p, amount, [0.05, 0.1, 0.2, 0.3, 0.5], [0.5, 0.9], [0.05, 0.2])
    assert len(sens) == 4
    low_loss = sens[(sens["lgd"] == 0.5) & (sens["profit_rate_good_loan"] == 0.2)][
        "best_threshold"
    ].iloc[0]
    high_loss = sens[(sens["lgd"] == 0.9) & (sens["profit_rate_good_loan"] == 0.05)][
        "best_threshold"
    ].iloc[0]
    assert low_loss >= high_loss  # costlier defaults -> stricter cut-off


def test_breakeven_and_assumption_validation():
    assert breakeven_pd(DecisionAssumptions(lgd=0.9, profit_rate_good_loan=0.1)) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError):
        DecisionAssumptions(lgd=1.5)


def test_bucket_summary_and_decision_curve():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 0.9, 1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    amount = np.full(1000, 1000.0)
    bs = bucket_summary(y, p, amount, BUCKETS, DecisionAssumptions())
    assert bs["n"].sum() == 1000 and bs["share"].sum() == pytest.approx(1.0)
    curve = decision_curve(y, p, amount)
    row = curve.iloc[-1]
    assert row["volume_approved_good"] + row["volume_rejected_good"] == pytest.approx(
        (y == 0).sum() * 1000
    )
    assert curve["n_approved"].is_monotonic_increasing


# ----------------------------------------------------------------- end-to-end evaluation
def test_segment_groups_are_naturally_ordered():
    from creditiq.risk.segments import segment_performance

    df = pd.DataFrame({"loan_amnt": [1000, 7000, 12000, 30000] * 60})
    y = np.tile([0, 1, 0, 1], 60)
    p = np.full(len(df), 0.3)
    seg = segment_performance(
        df, y, p, 0.5, {"amt": {"column": "loan_amnt", "bins": [0, 5000, 10000, 25000, np.inf]}}
    )
    assert seg["group"].tolist() == [
        "0 - <5,000",
        "5,000 - <10,000",
        "10,000 - <25,000",
        ">= 25,000",
    ]


# ----------------------------------------------------------------- end-to-end evaluation
def test_run_evaluation_end_to_end(trained, prepared, cfg, tmp_path):
    _, models_dir = trained
    local_cfg = cfg.with_overrides(
        {"paths": {"models_dir": str(models_dir), "metrics_dir": str(tmp_path)}}
    )
    out = run_evaluation(prepared.data, local_cfg, "applicant", clean_all=prepared.cleaned_all)
    tm = out.tables["test_metrics"]
    assert set(tm["model"]) == {"dummy", "logistic_regression", "lightgbm"}
    assert out.summary["test_n"] == len(split_frames(prepared.data)["test"])
    assert not out.tables["segment_performance"].empty
    assert "auc_diff_lightgbm_minus_logreg" in out.summary
    bounds = out.tables["unresolved_bounds"]
    lg = bounds[bounds["model"] == "lightgbm"].set_index("scenario")
    reported = lg.loc["reported: unresolved excluded"]
    repaid = lg.loc["all unresolved repaid"]
    assert reported["n_unresolved_added"] == 0 and repaid["n_unresolved_added"] > 0
    assert repaid["prevalence"] < reported["prevalence"]  # adding survivors lowers the rate
    assert reported["roc_auc"] == pytest.approx(
        tm[(tm["model"] == "lightgbm") & tm["variant"].str.startswith("calibrated")][
            "roc_auc"
        ].iloc[0]
    )
    scores = out.tables["score_distribution"]
    assert scores["n"].sum() == out.summary["test_n"]
    saved = save_evaluation(out, tmp_path)
    assert (saved / "evaluation_summary.json").is_file()
    assert (saved / "decision_curve.csv").is_file()


def test_bias_paragraph_labels_scenarios_as_assumed(trained, prepared, cfg, tmp_path):
    """Regression: hypothetical outcome assignments must not be worded as observed results."""
    from creditiq.reporting.markdown import bias_paragraph

    _, models_dir = trained
    local_cfg = cfg.with_overrides({"paths": {"models_dir": str(models_dir)}})
    out = run_evaluation(prepared.data, local_cfg, "applicant", clean_all=prepared.cleaned_all)
    text = bias_paragraph(
        {
            "metadata": {"selected_model": out.selected_model},
            "unresolved_bounds": out.tables["unresolved_bounds"],
            "sensitivity_36_month": out.tables["sensitivity_36_month"],
        }
    )
    assert "assumed" in text and "not observed outcomes" in text
    assert "realistic" not in text
    assert "scenario rate" in text


def test_threshold_metrics_from_bins_match_exact(trained, prepared):
    from creditiq.models.evaluate import threshold_metrics
    from creditiq.models.evaluation_run import score_distribution
    from creditiq.reporting.interactive import threshold_metrics_from_bins

    result, _ = trained
    test = split_frames(prepared.data)["test"]
    p = result.bundles["lightgbm"].predict_proba(test)
    y = test["is_adverse"].to_numpy()
    bins = score_distribution(y, p, test["loan_amnt"].to_numpy(dtype=float))
    for t in (0.1, 0.2, 0.35):  # thresholds on bin edges are exact
        a = threshold_metrics_from_bins(bins, t)
        b = threshold_metrics(y, p, t)
        assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (b["tp"], b["fp"], b["fn"], b["tn"])
