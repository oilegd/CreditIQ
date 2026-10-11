"""Leakage prevention: forbidden columns can never enter training or inference."""

from __future__ import annotations

import pandas as pd
import pytest

from creditiq.data.synthetic import make_synthetic_loans
from creditiq.features.build_features import FeatureBuilder, resolve_feature_spec
from creditiq.features.leakage_checks import (
    KNOWN_LEAKAGE_COLUMNS,
    LeakageError,
    assert_no_leakage,
    audit_table,
    check_feature_inputs,
    forbidden_columns,
    uncatalogued_columns,
    univariate_auc_screen,
    validate_catalog,
)
from creditiq.models.artifacts import SchemaMismatchError, coerce_to_schema

PROMPT_LEAKAGE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "last_credit_pull_d",
]
ORIGINATION_ONLY = [
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
    "funded_amnt",
    "funded_amnt_inv",
]


def test_catalog_is_valid_and_covers_every_source_column(cfg):
    catalog = cfg["feature_catalog"]
    validate_catalog(catalog)
    source_cols = make_synthetic_loans(10).columns  # mirrors the real workbook schema
    assert uncatalogued_columns(source_cols, catalog) == []


@pytest.mark.parametrize("col", PROMPT_LEAKAGE_COLUMNS)
def test_known_leakage_columns_are_forbidden(cfg, col):
    assert col in forbidden_columns(cfg["feature_catalog"])
    assert col in KNOWN_LEAKAGE_COLUMNS


@pytest.mark.parametrize("feature_set", ["applicant", "origination"])
def test_feature_sets_contain_no_forbidden_inputs(cfg, feature_set):
    spec = resolve_feature_spec(cfg.raw, feature_set)
    forbidden = forbidden_columns(cfg["feature_catalog"])
    inputs = set(spec.input_columns) - {spec.reference_date_column}
    assert not inputs & forbidden


def test_applicant_set_excludes_origination_fields(cfg):
    spec = resolve_feature_spec(cfg.raw, "applicant")
    assert not set(spec.input_columns) & set(ORIGINATION_ONLY)
    assert "verification_status" not in spec.input_columns


def test_hardcoded_denylist_beats_config_misclassification(cfg):
    catalog = {
        **cfg["feature_catalog"],
        "total_pymnt": {"category": "pre_underwriting", "rationale": "x"},
    }
    with pytest.raises(LeakageError, match="total_pymnt"):
        check_feature_inputs(["loan_amnt", "total_pymnt"], catalog, ["pre_underwriting"])


def test_unclassified_column_rejected(cfg):
    with pytest.raises(LeakageError, match="not classified"):
        check_feature_inputs(
            ["loan_amnt", "mystery_col"], cfg["feature_catalog"], ["pre_underwriting"]
        )


def test_adding_leakage_to_config_fails_at_resolution(cfg):
    bad = cfg.with_overrides(
        {"features": {"sets": {"applicant": {"numeric": ["loan_amnt", "recoveries"]}}}}
    )
    with pytest.raises(LeakageError):
        resolve_feature_spec(bad.raw, "applicant")


def test_origination_column_rejected_from_applicant_set(cfg):
    bad = cfg.with_overrides(
        {"features": {"sets": {"applicant": {"numeric": ["loan_amnt", "int_rate"]}}}}
    )
    with pytest.raises(LeakageError, match="int_rate"):
        resolve_feature_spec(bad.raw, "applicant")


def test_assert_no_leakage(cfg):
    assert_no_leakage(["loan_amnt", "dti"], cfg["feature_catalog"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["loan_amnt", "last_pymnt_amnt"], cfg["feature_catalog"])


def test_model_features_never_include_outcome_columns(prepared, cfg):
    spec = resolve_feature_spec(cfg.raw, "applicant")
    out = FeatureBuilder(spec).fit(prepared.data).transform(prepared.data)
    leaked = set(out.columns) & (
        forbidden_columns(cfg["feature_catalog"]) | {"is_adverse", "split"}
    )
    assert not leaked


def test_inference_drops_extra_columns_and_strict_mode_rejects(trained, prepared):
    result, _ = trained
    bundle = result.bundles["lightgbm"]
    frame = prepared.data.head(5)  # contains outcome columns
    coerced = coerce_to_schema(frame, bundle.input_schema)
    assert set(coerced.columns) == set(bundle.input_schema)
    assert not set(coerced.columns) & set(PROMPT_LEAKAGE_COLUMNS)
    with pytest.raises(SchemaMismatchError):
        coerce_to_schema(frame, bundle.input_schema, strict=True)


def test_predictions_unaffected_by_outcome_columns(trained, prepared):
    """Changing outcome fields must not change predictions (they are never used)."""
    result, _ = trained
    bundle = result.bundles["lightgbm"]
    frame = prepared.data.head(50).copy()
    p1 = bundle.predict_proba(frame)
    for col in PROMPT_LEAKAGE_COLUMNS:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col] * 0 + 12345
    frame["is_adverse"] = 1 - frame["is_adverse"]
    p2 = bundle.predict_proba(frame)
    pd.testing.assert_series_equal(pd.Series(p1), pd.Series(p2))


def test_univariate_screen_flags_outcome_fields(prepared):
    train = prepared.data[prepared.data["split"].astype(str) == "train"]
    screen = univariate_auc_screen(train, "is_adverse", ["recoveries", "dti", "total_rec_late_fee"])
    top = screen.iloc[0]
    assert top["column"] in {"recoveries", "total_rec_late_fee"}
    assert top["separation"] > 0.9


def test_audit_table_lists_membership(cfg):
    specs = {
        n: resolve_feature_spec(cfg.raw, n).model_input_columns
        for n in ("applicant", "origination")
    }
    audit = audit_table(cfg["feature_catalog"], specs)
    row = audit.set_index("column").loc["int_rate"]
    assert not row["used_in_applicant"] and row["used_in_origination"]
    assert audit.set_index("column").loc["recoveries", "category"] == "outcome_derived"
