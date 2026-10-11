"""LIME, permutation importance (single/grouped), SHAP summaries and the explainability suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditiq.data.split import split_frames
from creditiq.explainability.lime_tabular import (
    explain_lime,
    global_lime_importance,
    lime_stability,
)
from creditiq.explainability.permutation import permutation_importance
from creditiq.explainability.shap_analysis import (
    compute_shap,
    dependence_table,
    shap_by_raw_input,
    signed_importance,
)
from creditiq.explainability.suite import explainer_agreement, run_explainability
from creditiq.features.build_features import feature_raw_sources, resolve_feature_spec
from creditiq.models.artifacts import example_application
from creditiq.models.train import select_candidate


@pytest.fixture(scope="module")
def frames(prepared):
    return split_frames(prepared.data)


@pytest.fixture(scope="module")
def lgbm(trained):
    return trained[0].bundles["lightgbm"]


# ----------------------------------------------------------------- LIME
def test_lime_is_deterministic_and_uses_only_artifacts(lgbm, frames):
    row = frames["test"].iloc[[0]]
    a = explain_lime(lgbm, row, n_samples=800, seed=3)
    b = explain_lime(lgbm, row, n_samples=800, seed=3)
    pd.testing.assert_frame_equal(a.explanation, b.explanation)
    assert a.n_samples == 800
    assert "issue_d" not in set(a.explanation["feature"])  # reference date held fixed
    assert a.model_prediction == pytest.approx(lgbm.predict_proba(row)[0])
    assert a.r2 <= 1.0


@pytest.mark.parametrize(("term", "sign"), [("36 months", -1), ("60 months", 1)])
def test_lime_recovers_the_term_effect(lgbm, term, sign):
    """Synthetic data: 60-month loans are riskier, so the term condition must carry that sign."""
    app = example_application(lgbm.input_schema).assign(term=term)
    expl = explain_lime(lgbm, app, n_samples=3000, seed=0).explanation.set_index("feature")
    assert np.sign(expl.loc["term", "weight"]) == sign
    assert expl.loc["term", "condition"] == f"term = {term}"


def test_lime_stability_and_global(lgbm, frames):
    stab = lime_stability(lgbm, frames["test"].iloc[[1]], seeds=(0, 1, 2), n_samples=600)
    assert stab["sign_agreement"].between(0, 1).all()
    glob = global_lime_importance(lgbm, frames["test"].head(4), n_samples=500)
    assert (glob["mean_abs_lime_weight"] >= 0).all()
    assert "mean_r2" in glob.attrs


def test_lime_rejects_multiple_rows(lgbm, frames):
    with pytest.raises(ValueError):
        explain_lime(lgbm, frames["test"].head(2))


# ----------------------------------------------------------------- PFI
def test_permutation_importance_single_and_grouped(lgbm, frames):
    test = frames["test"]
    y = test["is_adverse"].to_numpy()
    single = permutation_importance(lgbm, test, y, n_repeats=2, seed=0)
    assert {"roc_auc_drop_mean", "log_loss_drop_mean", "brier_drop_mean"} <= set(single.columns)
    assert "issue_d" not in set(single["feature"])
    top = single.iloc[0]["feature"]
    assert top in {"term", "dti", "annual_inc", "inq_last_6mths", "revol_util"}
    again = permutation_importance(lgbm, test, y, n_repeats=2, seed=0)
    pd.testing.assert_frame_equal(single, again)
    grouped = permutation_importance(
        lgbm,
        test,
        y,
        groups={"size": ["annual_inc", "loan_amnt"], "ghost": ["not_a_column"]},
        n_repeats=2,
    )
    assert list(grouped["feature"]) == ["size"]  # missing columns dropped
    assert grouped.iloc[0]["n_columns"] == 2
    assert set(grouped.attrs["baseline"]) == {"roc_auc", "log_loss", "brier"}


# ----------------------------------------------------------------- SHAP summaries
def test_shap_summaries(trained, frames, cfg):
    spec = resolve_feature_spec(cfg.raw, "applicant")
    for name in ("lightgbm", "logistic_regression"):
        res = compute_shap(trained[0].bundles[name], frames["test"].head(200))
        signed = signed_importance(res)
        assert signed["mean_abs_shap"].is_monotonic_decreasing
        dep = dependence_table(res, ["dti"], n_bins=10)
        assert len(dep) <= 11 and dep["n"].sum() <= 200  # aggregated bins, not rows
        raw = shap_by_raw_input(res, spec)
        assert raw["mean_abs_shap"].sum() == pytest.approx(np.abs(res.values).mean(axis=0).sum())


def test_feature_raw_sources(cfg):
    spec = resolve_feature_spec(cfg.raw, "applicant")
    assert feature_raw_sources("loan_to_income", spec) == ["loan_amnt", "annual_inc"]
    assert feature_raw_sources("log_annual_inc", spec) == ["annual_inc"]
    assert feature_raw_sources("missingindicator_mths_since_last_delinq", spec) == [
        "mths_since_last_delinq"
    ]
    assert feature_raw_sources("purpose_credit_card", spec) == ["purpose"]
    assert feature_raw_sources("dti", spec) == ["dti"]


def test_explainer_agreement_ranks():
    shap_raw = pd.DataFrame({"feature": ["a", "b", "c"], "mean_abs_shap": [3, 2, 1]})
    pfi = pd.DataFrame({"feature": ["a", "b", "c"], "roc_auc_drop_mean": [0.3, 0.1, 0.2]})
    lime = pd.DataFrame({"feature": ["a", "b"], "mean_abs_lime_weight": [0.5, 0.4]})
    agr = explainer_agreement(shap_raw, pfi, lime).set_index("feature")
    assert agr.loc["a", ["rank_shap", "rank_pfi", "rank_lime"]].tolist() == [1, 1, 1]
    assert agr.loc["c", "rank_lime"] == 3  # absent in LIME -> importance 0


def test_run_explainability_suite(trained, prepared, cfg):
    _, models_dir = trained
    small = cfg.with_overrides(
        {
            "paths": {"models_dir": str(models_dir)},
            "evaluation": {
                "shap_sample_size": 150,
                "permutation_sample_size": 200,
                "permutation_repeats": 2,
            },
            "explainability": {"lime_global_rows": 3, "lime_samples": 1000},
        }
    )
    tables, objects = run_explainability(small, "applicant", prepared.data)
    for m in ("lightgbm", "logistic_regression"):
        for t in (
            "shap_signed",
            "shap_dependence",
            "shap_raw",
            "pfi_single",
            "pfi_grouped",
            "lime_global",
        ):
            assert not tables[f"{t}_{m}"].empty, (t, m)
    assert set(tables["explainer_agreement"]["model"]) == {"lightgbm", "logistic_regression"}
    assert "shap_global_importance" in tables and "shap_lightgbm" in objects


# ----------------------------------------------------------------- model selection tie-break
def test_select_candidate_breaks_near_ties_on_log_loss():
    rows = [
        {"val_roc_auc": 0.70000, "val_log_loss": 0.40},
        {"val_roc_auc": 0.70003, "val_log_loss": 0.60},  # tiny AUC edge, much worse probabilities
        {"val_roc_auc": 0.69000, "val_log_loss": 0.30},  # outside tolerance
    ]
    assert select_candidate(rows, "roc_auc", tolerance=0.0005) == 0
    assert select_candidate(rows, "roc_auc", tolerance=0.0) == 1
