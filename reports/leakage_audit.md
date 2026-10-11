# Feature leakage audit

> **Educational portfolio project.** Not validated or approved for real-world lending decisions. Scores are not regulatory probabilities of default.

## Prediction point

The intended decision point is **underwriting of a new application**. A feature is
eligible only if its value would be known at that moment. Lending Club's export does
not include application timestamps; the issue month (`issue_d`) is used as the closest
available reference date (for credit-history length and out-of-time splitting).

## Categories

| Category | Meaning | Allowed in models? |
| --- | --- | --- |
| pre_underwriting | Supplied by the applicant or the credit bureau before a decision | yes |
| at_origination | Produced by Lending Club's own assessment/pricing (grade, rate, instalment, listing) | origination model only |
| uncertain | Timing not documented in this export | only when explicitly allow-listed |
| outcome_derived | Repayment, recovery or other post-origination information | **never** |
| sensitive | Free text or geography that can identify people or proxy protected traits | **never** |
| identifier_metadata | IDs, URLs, constants, split dates | **never** |

| category | n_columns |
| --- | --- |
| pre_underwriting | 17 |
| outcome_derived | 16 |
| at_origination | 6 |
| sensitive | 5 |
| identifier_metadata | 4 |
| uncertain | 3 |

## Feature sets

* **applicant** - Conservative applicant-information model. Uses only attributes supplied by the applicant or drawn from the credit bureau at application time. Excludes Lending Club's own grading, pricing and verification outputs.
* **origination** - Loan-origination model. Adds Lending Club's sub-grade, interest rate, instalment, income-verification status and listing status, which are only known once Lending Club has assessed (and priced) the application. `grade` is not used separately because it is the first character of `sub_grade`.

The **applicant** model is the primary portfolio model. The **origination** model is
clearly labelled: it consumes Lending Club's own grade and interest rate, which already
encode Lending Club's risk assessment, so it answers a different question ("how much
does the platform's grade add?") and must not be presented as a pre-approval model.

## Column-by-column decisions

| column | category | rationale | in_source | used_in_applicant | used_in_origination |
| --- | --- | --- | --- | --- | --- |
| annual_inc | pre_underwriting | Self-reported annual income. | yes | yes | yes |
| collections_12_mths_ex_med | pre_underwriting | Bureau: collections in last 12 months excluding medical. | yes | yes | yes |
| delinq_2yrs | pre_underwriting | Bureau: 30+ day delinquencies in the past 2 years. | yes | yes | yes |
| dti | pre_underwriting | Debt-to-income from self-reported income and bureau debts (excluding the requested loan). | yes | yes | yes |
| earliest_cr_line | pre_underwriting | Bureau: earliest credit line date; used to derive credit-history length. Correlates with age (a protected attribute) - see model card. | yes | yes | yes |
| emp_length | pre_underwriting | Self-reported employment length. | yes | yes | yes |
| home_ownership | pre_underwriting | Self-reported housing status. | yes | yes | yes |
| inq_last_6mths | pre_underwriting | Bureau: credit inquiries in the last 6 months. | yes | yes | yes |
| mths_since_last_delinq | pre_underwriting | Bureau: months since last delinquency (missing = none on file). | yes | yes | yes |
| mths_since_last_record | pre_underwriting | Bureau: months since last public record (missing = none on file). | yes | yes | yes |
| open_acc | pre_underwriting | Bureau: number of open credit lines. | yes | yes | yes |
| pub_rec | pre_underwriting | Bureau: number of derogatory public records. | yes | yes | yes |
| purpose | pre_underwriting | Loan purpose selected by the applicant. | yes | yes | yes |
| revol_bal | pre_underwriting | Bureau: total revolving balance. | yes | yes | yes |
| revol_util | pre_underwriting | Bureau: revolving utilisation rate (fraction). | yes | yes | yes |
| term | pre_underwriting | Loan term selected by the applicant (36/60 months). | yes | yes | yes |
| total_acc | pre_underwriting | Bureau: total number of credit lines. | yes | yes | yes |
| funded_amnt | at_origination | Amount committed by Lending Club during/after credit review. Differs from loan_amnt in only 65 rows. Not used. | yes | no | no |
| grade | at_origination | Lending Club's internal risk grade - an output of its own credit model. Not used directly: it is the first character of sub_grade. | yes | no | no |
| initial_list_status | at_origination | Whole vs fractional listing, decided by Lending Club when listing. Origination model only. | yes | no | yes |
| installment | at_origination | Deterministic function of amount, rate and term; inherits int_rate's timing. Origination model only. | yes | no | yes |
| int_rate | at_origination | Price assigned by Lending Club from its own grade; a product of underwriting. Origination model only. | yes | no | yes |
| sub_grade | at_origination | Finer Lending Club risk grade - an output of its own credit model. Origination model only. | yes | no | yes |
| collection_recovery_fee | outcome_derived | Post charge-off collection fee - only exists after default. | yes | no | no |
| funded_amnt_inv | outcome_derived | Amount funded by investors after listing; reflects post-decision market behaviour. Never used. | yes | no | no |
| last_credit_pull_d | outcome_derived | Most recent bureau pull by Lending Club, months or years after origination. | yes | no | no |
| last_pymnt_amnt | outcome_derived | Amount of last payment - repayment outcome. | yes | no | no |
| last_pymnt_d | outcome_derived | Date of last payment - repayment outcome. | yes | no | no |
| loan_status | outcome_derived | Outcome used to build the target. Never a feature. | yes | no | no |
| next_pymnt_d | outcome_derived | Next scheduled payment; populated only for active loans (encodes the outcome). | yes | no | no |
| out_prncp | outcome_derived | Outstanding principal at snapshot - repayment outcome. | yes | no | no |
| out_prncp_inv | outcome_derived | Outstanding principal (investor share) at snapshot - repayment outcome. | yes | no | no |
| pymnt_plan | outcome_derived | Flags a hardship payment plan, which only occurs after repayment problems. | yes | no | no |
| recoveries | outcome_derived | Post charge-off recoveries - only exists after default. | yes | no | no |
| total_pymnt | outcome_derived | Payments received to date - repayment outcome. | yes | no | no |
| total_pymnt_inv | outcome_derived | Payments received to date (investor share) - repayment outcome. | yes | no | no |
| total_rec_int | outcome_derived | Interest received to date - repayment outcome. | yes | no | no |
| total_rec_late_fee | outcome_derived | Late fees received - only exists after late payment. | yes | no | no |
| total_rec_prncp | outcome_derived | Principal received to date - repayment outcome. | yes | no | no |
| addr_state | sensitive | US state; geographic proxy. Excluded from models; descriptive portfolio analytics only. | yes | no | no |
| desc | sensitive | Borrower free text (77% missing); may contain personal data and dated post-listing edits. Excluded. | yes | no | no |
| emp_title | sensitive | Free-text job title (~182k unique values); identifying and noisy. Excluded. | yes | no | no |
| title | sensitive | Free-text loan title; redundant with purpose. Excluded. | yes | no | no |
| zip_code | sensitive | 3-digit ZIP prefix; geographic proxy that can encode protected characteristics. Excluded. | yes | no | no |
| loan_amnt | uncertain | Listed loan amount. Lending Club's data dictionary states that if the credit department reduces the amount, the reduction is reflected here, so it is not purely the applicant's request. Allow-listed for the applicant model (no cleaner requested-amount field exists); flagged. | yes | yes | yes |
| mths_since_last_major_derog | uncertain | Bureau: months since a 90+ day rating. Assumed as-of-application like other bureau fields, but refresh timing is undocumented in this export. Used in both models; flagged. | yes | yes | yes |
| verification_status | uncertain | Income-verification outcome produced during Lending Club review; may complete after listing. Origination model only. | yes | no | yes |
| application_type | identifier_metadata | Constant ('Individual') in this export. | yes | no | no |
| issue_d | identifier_metadata | Issue month. Used for out-of-time splitting and as reference date for credit-history length; not a direct feature. | yes | no | no |
| policy_code | identifier_metadata | Constant (=1) in this export. | yes | no | no |
| url | identifier_metadata | Listing URL; 100% empty in this export. | yes | no | no |

## Empirical leakage screen (training split)

Single-column ROC-AUC against the target. `separation` = 2 x |AUC - 0.5|. Outcome-derived
fields separate classes almost perfectly, which confirms they must be excluded. This
screen is descriptive and only supports the documented reasoning above; a low score
does not make a column safe.

| column | n_non_null | auc | separation | category |
| --- | --- | --- | --- | --- |
| recoveries | 188,171 | 0.967 | 0.933 | outcome_derived |
| collection_recovery_fee | 188,171 | 0.865 | 0.730 | outcome_derived |
| total_rec_prncp | 188,171 | 0.160 | 0.680 | outcome_derived |
| last_pymnt_amnt | 188,171 | 0.235 | 0.530 | outcome_derived |
| total_pymnt | 188,171 | 0.282 | 0.435 | outcome_derived |
| total_pymnt_inv | 188,171 | 0.282 | 0.435 | outcome_derived |
| last_pymnt_d | 188,021 | 0.322 | 0.355 | outcome_derived |
| int_rate | 188,171 | 0.663 | 0.327 | at_origination |
| last_credit_pull_d | 188,161 | 0.377 | 0.246 | outcome_derived |
| dti | 188,171 | 0.563 | 0.126 | pre_underwriting |
| annual_inc | 188,171 | 0.444 | 0.112 | pre_underwriting |
| revol_util | 188,046 | 0.549 | 0.097 | pre_underwriting |
| inq_last_6mths | 188,171 | 0.545 | 0.089 | pre_underwriting |
| funded_amnt_inv | 188,171 | 0.544 | 0.087 | outcome_derived |
| loan_amnt | 188,171 | 0.544 | 0.087 | uncertain |
| funded_amnt | 188,171 | 0.544 | 0.087 | at_origination |
| total_rec_late_fee | 188,171 | 0.538 | 0.076 | outcome_derived |
| installment | 188,171 | 0.534 | 0.069 | at_origination |
| earliest_cr_line | 188,171 | 0.526 | 0.052 | pre_underwriting |
| mths_since_last_record | 17,473 | 0.519 | 0.038 | pre_underwriting |
| total_rec_int | 188,171 | 0.515 | 0.030 | outcome_derived |
| open_acc | 188,171 | 0.514 | 0.028 | pre_underwriting |
| total_acc | 188,171 | 0.490 | 0.020 | pre_underwriting |
| mths_since_last_major_derog | 32,514 | 0.508 | 0.017 | uncertain |
| issue_d | 188,171 | 0.494 | 0.012 | identifier_metadata |

## Automated guards

1. `creditiq.features.leakage_checks.check_feature_inputs` rejects any feature-set input
   whose audit category is not allowed for that set, or that is not classified at all.
2. A hard-coded deny-list (`KNOWN_LEAKAGE_COLUMNS`) rejects known Lending Club outcome fields
   even if the YAML catalogue is edited incorrectly.
3. `FeatureBuilder` re-checks at fit time; inference coerces inputs to the saved training
   schema and drops all other columns, so outcome fields cannot reach a model.
4. `tests/test_leakage.py` asserts all of the above.
