"""Schema validation, ingestion, cleaning, target construction and split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditiq.data.clean import (
    CleaningLog,
    clean_loans,
    normalize_emp_length,
    normalize_term,
    parse_dates,
    parse_percent,
)
from creditiq.data.load import (
    UnsupportedFormatError,
    detect_format,
    load_raw,
    normalize_column_name,
    read_source,
)
from creditiq.data.split import (
    SPLIT_COLUMN,
    assert_disjoint,
    assign_out_of_time_split,
    assign_random_split,
    split_frames,
)
from creditiq.data.target import UnknownStatusError, build_target
from creditiq.data.validate import (
    SchemaError,
    data_quality_summary,
    require_columns,
    validate_dataframe,
)


# ----------------------------------------------------------------- ingestion
def test_normalize_column_name():
    assert normalize_column_name(" Loan Amnt ") == "loan_amnt"
    assert normalize_column_name("int_rate (%)") == "int_rate"


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("a.csv", "csv"), ("a.csv.gz", "csv"), ("a.xlsx", "excel"), ("a.parquet", "parquet")],
)
def test_detect_format(tmp_path, name, fmt):
    assert detect_format(tmp_path / name) == fmt


def test_detect_format_rejects_unknown(tmp_path):
    with pytest.raises(UnsupportedFormatError):
        detect_format(tmp_path / "loans.json")


@pytest.mark.parametrize("suffix", ["csv", "parquet", "xlsx"])
def test_read_source_roundtrip(tmp_path, raw_synthetic, suffix):
    small = raw_synthetic.head(30).rename(columns={"loan_amnt": "Loan Amnt"})
    path = tmp_path / f"loans.{suffix}"
    if suffix == "csv":
        small.to_csv(path, index=False)
    elif suffix == "parquet":
        small.to_parquet(path, index=False)
    else:
        small.to_excel(path, index=False, sheet_name="2012_13_14")
    df = read_source(path, sheet="2012_13_14" if suffix == "xlsx" else 0)
    assert len(df) == 30
    assert "loan_amnt" in df.columns  # normalised


def test_read_source_detects_lfs_pointer(tmp_path):
    path = tmp_path / "pointer.xlsx"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        read_source(path)


def test_read_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_source(tmp_path / "missing.xlsx")


def test_load_raw_uses_cache(tmp_path, raw_synthetic):
    src = tmp_path / "loans.csv"
    raw_synthetic.head(50).to_csv(src, index=False)
    cache = tmp_path / "cache"
    first = load_raw(src, cache)
    assert (cache / "raw_cache.parquet").is_file()
    second = load_raw(src, cache)
    pd.testing.assert_frame_equal(first, second)
    assert src.is_file()  # source preserved


# ----------------------------------------------------------------- validation
def test_require_columns_fails_meaningfully(raw_copy):
    with pytest.raises(SchemaError, match="loan_status"):
        require_columns(raw_copy.drop(columns=["loan_status"]), ["loan_amnt", "loan_status"])


def test_validate_dataframe_reports_types_ranges_and_duplicates(raw_copy, cfg):
    raw_copy.loc[10, "dti"] = 500.0  # row 10 is not among the duplicated head(3)
    dup = pd.concat([raw_copy, raw_copy.head(3)], ignore_index=True)
    report = validate_dataframe(
        dup,
        required_columns=cfg["data"]["required_columns"],
        numeric_columns=["dti", "loan_amnt"],
        date_columns=[],
        valid_ranges=cfg["data"]["valid_ranges"],
        fingerprint=cfg["data"]["duplicate_fingerprint"],
    )
    assert report.ok
    assert report.range_violations["dti"] == 1
    assert report.n_duplicates == 3
    assert "fingerprint" in report.duplicate_method

    bad = raw_copy.assign(loan_amnt=raw_copy["loan_amnt"].astype(str))
    report2 = validate_dataframe(bad, ["loan_amnt"], ["loan_amnt"], [], {})
    assert not report2.ok and "loan_amnt" in report2.type_errors


def test_data_quality_summary(raw_copy):
    q = data_quality_summary(raw_copy)
    assert q.loc["url", "pct_missing"] == 100.0
    assert q.loc["policy_code", "n_unique"] == 1


# ----------------------------------------------------------------- cleaning
def test_parse_percent_handles_strings_and_fractions():
    assert parse_percent(pd.Series([0.1242, 0.2])).tolist() == [0.1242, 0.2]
    out = parse_percent(pd.Series(["12.5%", " 7% ", None]))
    assert out.iloc[0] == pytest.approx(0.125) and out.iloc[1] == pytest.approx(0.07)
    assert np.isnan(out.iloc[2])


def test_normalize_term_and_emp_length():
    assert normalize_term(pd.Series([" 36 months", "60", 36])).tolist() == [
        "36 months",
        "60 months",
        "36 months",
    ]
    out = normalize_emp_length(
        pd.Series(["10+ years", "< 1 year", "n/a", None, "3 years", "1 year"])
    )
    assert out.iloc[0] == "10+ years" and out.iloc[1] == "< 1 year"
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])
    assert out.iloc[4] == "3 years" and out.iloc[5] == "1 year"


def test_parse_dates_supports_text_months():
    df = pd.DataFrame({"issue_d": ["Dec-2011", "Jan-2012", "garbage"]})
    log = CleaningLog()
    out = parse_dates(df, ["issue_d"], log)
    assert out["issue_d"].iloc[0] == pd.Timestamp("2011-12-01")
    assert pd.isna(out["issue_d"].iloc[2])
    assert sum(log.value_fixes.values()) == 1


def test_clean_loans_tracks_removals_and_fixes(raw_copy, cfg):
    raw_copy.loc[0, "annual_inc"] = -5  # invalid -> NaN
    raw_copy.loc[1, "loan_status"] = None  # missing key -> row removed
    raw_copy.loc[2, "home_ownership"] = "NONE"
    dup = pd.concat([raw_copy, raw_copy.iloc[[5]]], ignore_index=True)
    cleaned, log = clean_loans(dup, cfg["data"])
    steps = {s["step"]: s for s in log.steps}
    assert steps["drop_missing_key_fields"]["rows_removed"] == 1
    assert steps["drop_duplicates"]["rows_removed"] == 1
    assert len(cleaned) == len(dup) - 2
    assert any("annual_inc" in k for k in log.value_fixes)
    assert "OTHER" in set(cleaned["home_ownership"].astype(str))
    for col in cfg["data"]["processed_drop_columns"]:
        assert col not in cleaned.columns


# ----------------------------------------------------------------- target
def test_build_target_excludes_unresolved(prepared, raw_synthetic, cfg):
    data = prepared.data
    statuses = set(data["loan_status"].astype(str))
    assert statuses <= {"Fully Paid", "Charged Off", "Default"}
    assert not statuses & set(cfg["target"]["excluded_statuses"])
    expected = (data["loan_status"].astype(str) != "Fully Paid").astype(int)
    assert (data["is_adverse"] == expected).all()
    n_unresolved = raw_synthetic["loan_status"].isin(cfg["target"]["excluded_statuses"]).sum()
    assert n_unresolved > 0
    step = next(s for s in prepared.cleaning_log.steps if s["step"] == "exclude_unresolved_status")
    assert step["rows_removed"] == n_unresolved


def test_build_target_maps_default_as_positive(cfg):
    df = pd.DataFrame({"loan_status": ["Default", "Fully Paid", "Current"]})
    out, _ = build_target(df, cfg["target"])
    assert out["is_adverse"].tolist() == [1, 0]


def test_build_target_rejects_unknown_status(cfg):
    df = pd.DataFrame({"loan_status": ["Fully Paid", "Does not meet the credit policy"]})
    with pytest.raises(UnknownStatusError):
        build_target(df, cfg["target"])


# ----------------------------------------------------------------- splits
def test_out_of_time_split_boundaries(prepared, cfg):
    frames = split_frames(prepared.data)
    month = {k: pd.to_datetime(v["issue_d"]).dt.to_period("M") for k, v in frames.items()}
    assert month["train"].max() <= pd.Period(cfg["split"]["train_end"], "M")
    assert month["validation"].min() > month["train"].max()
    assert month["calibration"].min() > month["validation"].max()
    assert month["test"].min() > month["calibration"].max()
    assert sum(len(f) for f in frames.values()) == len(prepared.data)


def test_splits_are_disjoint(prepared, cfg):
    frames = split_frames(prepared.data)
    keys = [c for c in cfg["data"]["duplicate_fingerprint"] if c in prepared.data.columns]
    assert_disjoint(frames, keys)
    leaky = {**frames, "test": pd.concat([frames["test"], frames["train"].head(1)])}
    with pytest.raises(AssertionError):
        assert_disjoint(leaky, keys)


def test_out_of_time_split_rejects_bad_boundaries(prepared, cfg):
    bad = {**cfg["split"], "validation_end": "2013-01"}
    with pytest.raises(ValueError):
        assign_out_of_time_split(prepared.data, bad)


def test_random_split_is_stratified_and_reproducible(prepared, cfg):
    a = assign_random_split(prepared.data, "is_adverse", cfg["split"], seed=1)
    b = assign_random_split(prepared.data, "is_adverse", cfg["split"], seed=1)
    assert a.equals(b)
    df = prepared.data.assign(**{SPLIT_COLUMN: a})
    rates = df.groupby(SPLIT_COLUMN)["is_adverse"].mean()
    assert rates.max() - rates.min() < 0.03
