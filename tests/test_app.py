"""Headless Streamlit app tests (synthetic artifacts only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from creditiq.config import find_project_root
from creditiq.explainability.suite import run_explainability
from creditiq.models.evaluation_run import run_evaluation, save_evaluation
from creditiq.tda.build import build_tda_clouds

APP = str(find_project_root() / "app" / "streamlit_app.py")
PAGES = [
    "🏠 Overview",
    "📊 Portfolio Analysis",
    "📈 Model Performance",
    "🔍 Explainability",
    "🎯 Risk Prediction",
    "💼 Decision Simulation",
    "🕸️ Topology (TDA)",
    "📚 Model Documentation",
]


def _write_config(raw: dict, tmp_path: Path, **paths: str) -> Path:
    raw = dict(raw)
    raw["paths"] = {**raw["paths"], **paths}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _new_app(monkeypatch, config_path: Path) -> AppTest:
    monkeypatch.setenv("CREDITIQ_CONFIG", str(config_path))
    monkeypatch.delenv("CREDITIQ_SYNTHETIC", raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(APP, default_timeout=180)


def test_app_without_artifacts_shows_setup_message(cfg, tmp_path, monkeypatch):
    empty = str(tmp_path / "empty")
    conf = _write_config(
        cfg.raw,
        tmp_path,
        models_dir=empty,
        metrics_dir=empty,
        reports_dir=empty,
        processed_dir=empty,
    )
    at = _new_app(monkeypatch, conf).run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        if page != PAGES[-1]:
            assert any("No " in w.value for w in at.warning), page


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_app_all_pages_with_artifacts(cfg, trained, prepared, tmp_path, monkeypatch):
    _, models_dir = trained
    metrics_dir = tmp_path / "metrics"
    local = cfg.with_overrides(
        {
            "paths": {
                "models_dir": str(models_dir),
                "metrics_dir": str(metrics_dir),
                "processed_dir": str(tmp_path / "processed"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "evaluation": {
                "shap_sample_size": 150,
                "permutation_sample_size": 200,
                "permutation_repeats": 2,
            },
            "explainability": {"lime_global_rows": 3, "lime_samples": 600},
            "tda": {"sample_size": 400, "persistence_sample": 120},
        }
    )
    save_evaluation(
        run_evaluation(prepared.data, local, "applicant", clean_all=prepared.cleaned_all),
        metrics_dir,
    )
    tables, _ = run_explainability(local, "applicant", prepared.data)
    for name, table in tables.items():
        table.to_csv(metrics_dir / "applicant" / f"{name}.csv", index=False)
    build_tda_clouds(local, "applicant", prepared.data)

    at = _new_app(monkeypatch, _write_config(local.raw, tmp_path)).run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        assert not any("No trained" in w.value for w in at.warning), page

    at.sidebar.radio[0].set_value("🕸️ Topology (TDA)").run()
    labels = {m.label: m.value for m in at.metric}
    assert int(labels["Nodes"].replace(",", "")) > 0
    assert "Connected components" in labels

    at.sidebar.radio[0].set_value("🎯 Risk Prediction").run()
    submit = next(b for b in at.button if "Score application" in b.label)
    submit.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    p = float(metrics["Calibrated adverse-outcome probability"].rstrip("%"))
    assert 0.0 <= p <= 100.0
    assert "Local fidelity R²" in metrics  # LIME tab rendered
