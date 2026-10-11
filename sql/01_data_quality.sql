-- Data-quality checks on the cleaned loan table (all statuses).
-- Views: loans_all (every cleaned loan), loans (resolved modelling population).

-- name: row_counts
SELECT
    (SELECT COUNT(*) FROM loans_all) AS n_all_loans,
    (SELECT COUNT(*) FROM loans)     AS n_resolved_loans,
    (SELECT COUNT(*) FROM loans_all) - (SELECT COUNT(*) FROM loans) AS n_excluded_unresolved;

-- name: status_distribution
SELECT
    loan_status,
    COUNT(*) AS n_loans,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM loans_all
GROUP BY loan_status
ORDER BY n_loans DESC;

-- name: key_field_missingness
SELECT
    COUNT(*) AS n,
    ROUND(100.0 * AVG(CASE WHEN emp_length IS NULL THEN 1 ELSE 0 END), 2)            AS pct_missing_emp_length,
    ROUND(100.0 * AVG(CASE WHEN revol_util IS NULL THEN 1 ELSE 0 END), 2)            AS pct_missing_revol_util,
    ROUND(100.0 * AVG(CASE WHEN mths_since_last_delinq IS NULL THEN 1 ELSE 0 END), 2) AS pct_missing_mths_since_last_delinq,
    ROUND(100.0 * AVG(CASE WHEN mths_since_last_record IS NULL THEN 1 ELSE 0 END), 2) AS pct_missing_mths_since_last_record,
    ROUND(100.0 * AVG(CASE WHEN mths_since_last_major_derog IS NULL THEN 1 ELSE 0 END), 2) AS pct_missing_mths_since_last_major_derog,
    ROUND(100.0 * AVG(CASE WHEN earliest_cr_line IS NULL THEN 1 ELSE 0 END), 2)      AS pct_missing_earliest_cr_line
FROM loans_all;

-- name: range_checks
SELECT
    SUM(CASE WHEN loan_amnt <= 0 THEN 1 ELSE 0 END)                    AS non_positive_loan_amnt,
    SUM(CASE WHEN annual_inc <= 0 THEN 1 ELSE 0 END)                   AS non_positive_annual_inc,
    SUM(CASE WHEN dti < 0 OR dti > 100 THEN 1 ELSE 0 END)              AS dti_out_of_range,
    SUM(CASE WHEN revol_util < 0 OR revol_util > 2 THEN 1 ELSE 0 END)  AS revol_util_out_of_range,
    SUM(CASE WHEN int_rate <= 0 OR int_rate >= 1 THEN 1 ELSE 0 END)    AS int_rate_out_of_range,
    SUM(CASE WHEN open_acc > total_acc THEN 1 ELSE 0 END)              AS open_exceeds_total_acc,
    SUM(CASE WHEN earliest_cr_line > issue_d THEN 1 ELSE 0 END)        AS credit_line_after_issue,
    SUM(CASE WHEN funded_amnt <> loan_amnt THEN 1 ELSE 0 END)          AS funded_differs_from_requested
FROM loans_all;

-- name: issue_month_coverage
SELECT
    issue_d,
    COUNT(*) AS n_loans,
    SUM(CASE WHEN loan_status IN ('Fully Paid', 'Charged Off', 'Default') THEN 1 ELSE 0 END) AS n_resolved
FROM loans_all
GROUP BY issue_d
ORDER BY issue_d;

-- name: duplicate_fingerprints
SELECT COUNT(*) AS n_duplicate_groups
FROM (
    SELECT loan_amnt, term, int_rate, installment, annual_inc, issue_d, dti,
           earliest_cr_line, revol_bal, open_acc, total_acc, COUNT(*) AS c
    FROM loans_all
    GROUP BY ALL
    HAVING COUNT(*) > 1
);
