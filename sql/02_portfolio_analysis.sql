-- Portfolio analytics. Descriptive only: differences between groups are associations,
-- not causal effects, and reflect Lending Club's historical approval policy.
-- Outcome fields (payments, recoveries) are used here for portfolio reporting only,
-- never as model features.

-- name: adverse_rate_by_vintage_term
SELECT
    DATE_TRUNC('quarter', issue_d) AS issue_quarter,
    term,
    COUNT(*)                       AS n_loans,
    SUM(is_adverse)                AS n_adverse,
    ROUND(AVG(is_adverse), 4)      AS adverse_rate
FROM loans
GROUP BY ALL
ORDER BY issue_quarter, term;

-- name: unresolved_share_by_vintage_term
-- Evidence for selection bias: share of loans still unresolved at the snapshot.
SELECT
    DATE_TRUNC('quarter', issue_d) AS issue_quarter,
    term,
    COUNT(*) AS n_loans,
    ROUND(AVG(CASE WHEN loan_status NOT IN ('Fully Paid', 'Charged Off', 'Default') THEN 1 ELSE 0 END), 4)
        AS unresolved_share
FROM loans_all
GROUP BY ALL
ORDER BY issue_quarter, term;

-- name: adverse_rate_by_grade
SELECT
    grade,
    COUNT(*)                    AS n_loans,
    ROUND(AVG(is_adverse), 4)   AS adverse_rate,
    ROUND(AVG(int_rate), 4)     AS mean_int_rate,
    ROUND(AVG(loan_amnt), 0)    AS mean_loan_amnt
FROM loans
GROUP BY grade
ORDER BY grade;

-- name: adverse_rate_by_purpose
SELECT
    purpose,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate,
    ROUND(SUM(loan_amnt) / 1e6, 2) AS volume_musd
FROM loans
GROUP BY purpose
ORDER BY n_loans DESC;

-- name: adverse_rate_by_home_ownership
SELECT
    home_ownership,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate
FROM loans
GROUP BY home_ownership
ORDER BY n_loans DESC;

-- name: adverse_rate_by_income_band
SELECT
    CASE
        WHEN annual_inc < 40000  THEN '1: < 40k'
        WHEN annual_inc < 60000  THEN '2: 40k-60k'
        WHEN annual_inc < 80000  THEN '3: 60k-80k'
        WHEN annual_inc < 120000 THEN '4: 80k-120k'
        ELSE '5: >= 120k'
    END AS income_band,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate,
    ROUND(MEDIAN(dti), 2)     AS median_dti
FROM loans
GROUP BY income_band
ORDER BY income_band;

-- name: adverse_rate_by_state_top15
SELECT
    addr_state,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate
FROM loans
GROUP BY addr_state
ORDER BY n_loans DESC
LIMIT 15;

-- name: realized_loss_charged_off
-- Empirical net principal loss on charged-off loans, from outcome fields.
-- Shown as a descriptive reference for the hypothetical LGD assumption; the
-- decision simulator does NOT estimate LGD from it.
SELECT
    COUNT(*) AS n_charged_off,
    ROUND(SUM(funded_amnt) / 1e6, 2) AS funded_musd,
    ROUND(
        SUM(funded_amnt - total_rec_prncp - (recoveries - collection_recovery_fee)) / SUM(funded_amnt),
        4
    ) AS net_principal_loss_rate_of_funded,
    ROUND(
        SUM(funded_amnt - total_rec_prncp - (recoveries - collection_recovery_fee))
        / NULLIF(SUM(funded_amnt - total_rec_prncp), 0),
        4
    ) AS net_loss_rate_of_outstanding_at_chargeoff
FROM loans
WHERE loan_status = 'Charged Off';

-- name: split_overview
SELECT
    split,
    COUNT(*)                  AS n_loans,
    ROUND(AVG(is_adverse), 4) AS adverse_rate,
    MIN(issue_d)              AS first_issue,
    MAX(issue_d)              AS last_issue
FROM loans
GROUP BY split
ORDER BY first_issue;
