-- Feature engineering in SQL. `engineered_features` mirrors a subset of
-- creditiq.features.build_features.engineer_features; tests assert both agree.

-- name: engineered_features
SELECT
    loan_amnt / NULLIF(annual_inc, 0)                         AS loan_to_income,
    revol_bal / NULLIF(annual_inc, 0)                         AS revol_bal_to_income,
    open_acc / NULLIF(total_acc, 0)                           AS open_to_total_acc,
    LN(1 + GREATEST(annual_inc, 0))                           AS log_annual_inc,
    CAST(REGEXP_EXTRACT(CAST(term AS VARCHAR), '(\d+)', 1) AS DOUBLE) AS term_months,
    CASE
        WHEN emp_length IS NULL THEN NULL
        WHEN CAST(emp_length AS VARCHAR) = '< 1 year' THEN 0
        WHEN CAST(emp_length AS VARCHAR) = '10+ years' THEN 10
        ELSE CAST(REGEXP_EXTRACT(CAST(emp_length AS VARCHAR), '(\d+)', 1) AS DOUBLE)
    END                                                       AS emp_length_years,
    CASE WHEN earliest_cr_line IS NULL THEN NULL ELSE
        (YEAR(issue_d) - YEAR(earliest_cr_line)) * 12 + (MONTH(issue_d) - MONTH(earliest_cr_line))
    END                                                       AS credit_history_months,
    CASE WHEN mths_since_last_delinq IS NOT NULL OR COALESCE(delinq_2yrs, 0) > 0 THEN 1 ELSE 0 END
                                                              AS has_delinq_history,
    CASE WHEN COALESCE(pub_rec, 0) > 0 OR mths_since_last_record IS NOT NULL THEN 1 ELSE 0 END
                                                              AS has_public_record
FROM loans;

-- name: dti_decile_risk
-- Monotonicity check of a key feature on the TRAINING split only.
SELECT
    dti_decile,
    COUNT(*)                  AS n_loans,
    ROUND(MIN(dti), 2)        AS min_dti,
    ROUND(MAX(dti), 2)        AS max_dti,
    ROUND(AVG(is_adverse), 4) AS adverse_rate
FROM (
    SELECT dti, is_adverse, NTILE(10) OVER (ORDER BY dti) AS dti_decile
    FROM loans
    WHERE split = 'train' AND dti IS NOT NULL
)
GROUP BY dti_decile
ORDER BY dti_decile;

-- name: loan_to_income_quintile_risk
SELECT
    lti_quintile,
    COUNT(*)                          AS n_loans,
    ROUND(MIN(loan_to_income), 3)     AS min_loan_to_income,
    ROUND(MAX(loan_to_income), 3)     AS max_loan_to_income,
    ROUND(AVG(is_adverse), 4)         AS adverse_rate
FROM (
    SELECT
        loan_amnt / NULLIF(annual_inc, 0) AS loan_to_income,
        is_adverse,
        NTILE(5) OVER (ORDER BY loan_amnt / NULLIF(annual_inc, 0)) AS lti_quintile
    FROM loans
    WHERE split = 'train' AND annual_inc > 0
)
GROUP BY lti_quintile
ORDER BY lti_quintile;

-- name: credit_history_band_risk
SELECT
    CASE
        WHEN hist_years < 5  THEN '1: < 5y'
        WHEN hist_years < 10 THEN '2: 5-10y'
        WHEN hist_years < 15 THEN '3: 10-15y'
        WHEN hist_years < 20 THEN '4: 15-20y'
        ELSE '5: >= 20y'
    END AS credit_history_band,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate
FROM (
    SELECT
        ((YEAR(issue_d) - YEAR(earliest_cr_line)) * 12 + (MONTH(issue_d) - MONTH(earliest_cr_line))) / 12.0
            AS hist_years,
        is_adverse
    FROM loans
    WHERE split = 'train' AND earliest_cr_line IS NOT NULL
)
GROUP BY credit_history_band
ORDER BY credit_history_band;
