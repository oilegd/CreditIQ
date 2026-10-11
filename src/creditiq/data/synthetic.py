"""Explicitly SYNTHETIC Lending-Club-shaped data for tests, CI and demos.

The generator mimics the *schema* of the Figshare 2012-2014 workbook (same 51
column names and value formats) with a simple latent-risk data-generating process.
It is never used for reported model evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "medical",
    "car",
]
EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "5 years", "8 years", "10+ years"]
GRADES = list("ABCDEFG")
STATES = ["CA", "NY", "TX", "FL", "IL", "GA", "OH", "WA"]


def make_synthetic_loans(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` synthetic loans with the raw workbook schema (column names normalised)."""
    rng = np.random.default_rng(seed)
    issue_months = pd.date_range("2012-01-01", "2014-12-01", freq="MS")
    issue_d = pd.to_datetime(rng.choice(issue_months, size=n))
    term_months = rng.choice([36, 60], size=n, p=[0.72, 0.28])
    annual_inc = np.round(np.exp(rng.normal(11.1, 0.5, size=n)), -2).clip(4000, 2_000_000)
    loan_amnt = (np.round(rng.uniform(1000, 35000, size=n) / 25) * 25).astype(int)
    dti = rng.gamma(4.0, 4.4, size=n).clip(0, 39.99).round(2)
    revol_util = rng.beta(2.5, 2.0, size=n).round(3)
    delinq_2yrs = rng.poisson(0.3, size=n)
    inq_last_6mths = rng.poisson(0.8, size=n).clip(0, 8)
    pub_rec = rng.poisson(0.15, size=n)
    open_acc = rng.poisson(11, size=n).clip(1, None)
    total_acc = open_acc + rng.poisson(14, size=n)
    revol_bal = np.round(rng.lognormal(9.3, 1.0, size=n)).astype(int)
    hist_years = rng.gamma(4.0, 4.0, size=n).clip(1, 50)
    earliest_cr_line = (
        (issue_d - pd.to_timedelta(hist_years * 365.25, unit="D")).to_period("M").to_timestamp()
    )
    emp_length = rng.choice(
        EMP_LENGTHS + [None], size=n, p=[0.08, 0.07, 0.09, 0.08, 0.1, 0.08, 0.45, 0.05]
    )
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=n, p=[0.5, 0.4, 0.095, 0.005])
    purpose = rng.choice(PURPOSES, size=n)

    z = (
        -1.9
        + 0.9 * (term_months == 60)
        + 0.035 * (dti - 17)
        + 0.8 * (revol_util - 0.55)
        + 0.25 * inq_last_6mths
        - 0.45 * (np.log(annual_inc) - 11.1)
        + 0.25 * np.log1p(loan_amnt / annual_inc * 10)
        + 0.2 * (purpose == "small_business")
        - 0.015 * (hist_years - 15)
        + rng.normal(0, 0.6, size=n)
    )
    pd_true = 1 / (1 + np.exp(-z))
    adverse = rng.uniform(size=n) < pd_true

    grade_idx = np.clip(
        np.digitize(z + rng.normal(0, 0.4, n), [-2.6, -2.1, -1.6, -1.1, -0.6, 0.0]), 0, 6
    )
    grade = np.array(GRADES)[grade_idx]
    sub_grade = np.char.add(grade.astype(str), rng.integers(1, 6, size=n).astype(str))
    int_rate = np.round(0.06 + 0.033 * grade_idx + rng.normal(0, 0.004, n), 4).clip(0.05, 0.3)
    r = int_rate / 12
    installment = np.round(loan_amnt * r / (1 - (1 + r) ** (-term_months)), 2)

    status = np.where(adverse, "Charged Off", "Fully Paid").astype(object)
    late_60 = (term_months == 60) & (issue_d >= pd.Timestamp("2014-04-01"))
    unresolved = late_60 & (rng.uniform(size=n) < 0.3)
    status[unresolved] = rng.choice(
        ["Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"],
        size=int(unresolved.sum()),
        p=[0.9, 0.04, 0.02, 0.04],
    )

    paid_frac = np.where(adverse, rng.uniform(0.05, 0.7, n), 1.0)
    total_rec_prncp = np.round(loan_amnt * paid_frac, 2)
    total_rec_int = np.round(total_rec_prncp * int_rate * term_months / 24, 2)
    recoveries = np.where(adverse, np.round((loan_amnt - total_rec_prncp) * 0.1, 2), 0.0)
    out_prncp = np.where(unresolved, np.round(loan_amnt * 0.4, 2), 0.0)

    df = pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "funded_amnt_inv": np.round(loan_amnt * rng.uniform(0.97, 1.0, n), 2),
            "term": [f" {t} months" for t in term_months],
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_title": rng.choice(["Teacher", "Nurse", "Engineer", "Manager", None], size=n),
            "emp_length": emp_length,
            "home_ownership": home,
            "annual_inc": annual_inc,
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], size=n
            ),
            "issue_d": issue_d,
            "loan_status": status,
            "pymnt_plan": "n",
            "url": np.nan,
            "desc": None,
            "purpose": purpose,
            "title": "Synthetic loan",
            "zip_code": rng.choice(["100xx", "945xx", "606xx", "300xx"], size=n),
            "addr_state": rng.choice(STATES, size=n),
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "earliest_cr_line": earliest_cr_line,
            "inq_last_6mths": inq_last_6mths,
            "mths_since_last_delinq": np.where(
                delinq_2yrs > 0,
                rng.integers(1, 24, n),
                np.where(rng.uniform(size=n) < 0.3, rng.integers(24, 120, n), np.nan),
            ),
            "mths_since_last_record": np.where(pub_rec > 0, rng.integers(1, 120, n), np.nan),
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc,
            "initial_list_status": rng.choice(["f", "w"], size=n),
            "out_prncp": out_prncp,
            "out_prncp_inv": out_prncp,
            "total_pymnt": total_rec_prncp + total_rec_int + recoveries,
            "total_pymnt_inv": total_rec_prncp + total_rec_int + recoveries,
            "total_rec_prncp": total_rec_prncp,
            "total_rec_int": total_rec_int,
            "total_rec_late_fee": np.where(adverse, 15.0, 0.0),
            "recoveries": recoveries,
            "collection_recovery_fee": np.round(recoveries * 0.18, 2),
            "last_pymnt_d": issue_d + pd.DateOffset(months=12),
            "last_pymnt_amnt": installment,
            "next_pymnt_d": pd.Series(pd.NaT, index=range(n)).where(
                ~unresolved, pd.Timestamp("2019-04-01")
            ),
            "last_credit_pull_d": pd.Timestamp("2019-03-01"),
            "collections_12_mths_ex_med": rng.poisson(0.02, size=n),
            "mths_since_last_major_derog": np.where(
                rng.uniform(size=n) < 0.2, rng.integers(1, 80, n), np.nan
            ),
            "policy_code": 1,
            "application_type": "Individual",
        }
    )
    for col in ("emp_title", "emp_length", "desc"):
        df[col] = df[col].astype("string")
    return df
