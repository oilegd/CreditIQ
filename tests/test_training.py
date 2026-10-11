"""Training pipeline, calibration, metrics and artifact serialisation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from creditiq.data.split import split_frames
from creditiq.models.artifacts import METADATA_FILE, load_bundle
from creditiq.models.calibrate import Calibrator, calibration_table, select_calibration
from creditiq.models.evaluate import (
    bootstrap_ci,
    choose_threshold,
    evaluate_predictions,
    threshold_metrics,
)
from creditiq.models.train import MODEL_NAMES, train_all


def test_training_produces_all_models_and_artifacts(trained):
    result, models_dir = trained
    assert set(result.bundles) == set(MODEL_NAMES)
    assert result.selected_model in {"logistic_regression", "lightgbm"}
    out = models_dir / "applicant"
    for name in MODEL_NAMES:
        assert (out / f"{name}.joblib").is_file()
    meta = json.loads((out / METADATA_FILE).read_text())
    assert meta["selected_model"] == result.selected_model
    assert meta["split"]["test"]["n"] > 0
    assert "environment" in meta and "config" in meta


def test_models_beat_dummy_on_validation(trained):
    result, _ = trained
    vm = result.validation_metrics
    assert vm.loc["dummy", "roc_auc"] == pytest.approx(0.5)
    assert vm.loc["lightgbm", "roc_auc"] > 0.6
    assert vm.loc["logistic_regression", "roc_auc"] > 0.6


def test_dummy_predicts_training_prior(trained, prepared):
    result, _ = trained
    frames = split_frames(prepared.data)
    p = result.bundles["dummy"].predict_raw(frames["test"])
    assert np.allclose(p, frames["train"]["is_adverse"].mean())


def test_serialized_bundle_reproduces_predictions(trained, prepared):
    result, models_dir = trained
    test = split_frames(prepared.data)["test"]
    for name in ("lightgbm", "logistic_regression"):
        loaded = load_bundle(models_dir / "applicant" / f"{name}.joblib")
        np.testing.assert_allclose(
            loaded.predict_proba(test), result.bundles[name].predict_proba(test)
        )


def test_training_is_reproducible_and_never_uses_test_labels(prepared, cfg, trained):
    """Retrain with scrambled TEST labels: every artifact must be identical."""
    result, _ = trained
    data = prepared.data.copy()
    is_test = data["split"].astype(str) == "test"
    data.loc[is_test, "is_adverse"] = 1 - data.loc[is_test, "is_adverse"]
    again = train_all(data, cfg, "applicant")
    calib = split_frames(prepared.data)["calibration"]
    for name in ("lightgbm", "logistic_regression"):
        np.testing.assert_allclose(
            again.bundles[name].predict_proba(calib), result.bundles[name].predict_proba(calib)
        )
        assert again.bundles[name].threshold == pytest.approx(result.bundles[name].threshold)
        assert again.bundles[name].calibrator.method == result.bundles[name].calibrator.method


# ----------------------------------------------------------------- calibration
def _miscalibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.6, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    raw = np.clip(true_p * 1.6, 0, 0.99)  # systematically over-predicts
    return raw, y


def test_calibration_improves_brier_out_of_sample():
    raw, y = _miscalibrated()
    cal = Calibrator("isotonic").fit(raw[:2000], y[:2000])
    assert brier_score_loss(y[2000:], cal.transform(raw[2000:])) < brier_score_loss(
        y[2000:], raw[2000:]
    )
    sig = Calibrator("sigmoid").fit(raw[:2000], y[:2000])
    assert brier_score_loss(y[2000:], sig.transform(raw[2000:])) < brier_score_loss(
        y[2000:], raw[2000:]
    )


def test_calibrators_are_monotone_and_bounded():
    raw, y = _miscalibrated()
    grid = np.linspace(0, 1, 101)
    for method in ("sigmoid", "isotonic", "none"):
        out = Calibrator(method).fit(raw, y).transform(grid)
        assert np.all(np.diff(out) >= -1e-12)
        assert out.min() > 0 and out.max() < 1


def test_select_calibration_prefers_recalibration_when_needed():
    raw, y = _miscalibrated()
    best, table = select_calibration(raw, y, cv_folds=5, metric="brier", seed=0)
    assert best.method in {"sigmoid", "isotonic"}
    assert table["selected"].sum() == 1
    assert set(table["method"]) == {"none", "sigmoid", "isotonic"}


def test_calibration_table_bins():
    raw, y = _miscalibrated()
    t = calibration_table(y, raw, n_bins=10)
    assert len(t) == 10 and t["n"].sum() == len(y)


def test_unknown_calibration_method():
    with pytest.raises(ValueError):
        Calibrator("magic")


# ----------------------------------------------------------------- metrics
def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    m = evaluate_predictions(y, p, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["prevalence"] == pytest.approx(y.mean())
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 500


def test_threshold_metrics_hand_computed():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.6])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(0.5)


def test_choose_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, p)
    assert 0.3 < t <= 0.7
    assert threshold_metrics(y, p, t)["f1"] == pytest.approx(1.0)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        evaluate_predictions(np.array([0, 1]), np.array([0.2, 1.4]), 0.5)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.2 + rng.uniform(0, 0.8, 800), 0, 1)
    ci = bootstrap_ci(y, p, reps=50, seed=0)
    for vals in ci.values():
        assert vals["lower"] <= vals["point"] <= vals["upper"]
    assert pd.Series([v["reps"] for v in ci.values()]).eq(50).all()
