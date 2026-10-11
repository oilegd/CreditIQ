# Model card - CreditIQ `applicant` model

> **Educational portfolio project.** Not validated or approved for real-world lending decisions. Scores are not regulatory probabilities of default.

## Model details

* **Model:** `lightgbm` (selected on validation roc_auc) with
  `sigmoid` probability calibration.
* **Version:** 0.1.0; trained 2026-10-11T21:47:51+00:00
  (Python 3.12.13, scikit-learn 1.9.1,
  LightGBM 4.7.0).
* **Inputs:** 20 raw applicant/bureau fields: `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `collections_12_mths_ex_med`, `term`, `emp_length`, `home_ownership`, `purpose`, `earliest_cr_line`, `issue_d`.
  `issue_d` is used only as the reference date for credit-history length.
* **Output:** calibrated probability of an adverse loan outcome, plus a risk bucket.
* **Artifacts:** `models/applicant/` (pipeline, calibrator, threshold, input schema, metadata).

## Intended use

* Demonstrating credit-risk modelling methodology: leakage control, out-of-time validation,
  calibration, explainability and decision analysis.
* Educational exploration of how application-time information relates to historical
  Lending Club loan outcomes.

## Prohibited uses

* Any real lending, pricing, collections, employment, insurance or housing decision.
* Use as a regulatory PD, IFRS 9 / CECL / Basel parameter, or adverse-action reason source.
* Application to populations other than 2012-2014 Lending Club borrowers without
  re-development and validation.

## Training population and outcome definition

* **Population:** Lending Club loans issued 2012-01-01 to 2013-12-01
  (training split, n = 188,171) that had **resolved** by the snapshot. These are
  loans Lending Club *approved and funded*; rejected applicants are absent
  (reject-inference / selection bias).
* **Positive class:** `Charged Off` or `Default`. **Negative:** `Fully Paid`.
  **Excluded:** `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`.
* **Horizon:** loan lifetime as observed at the snapshot. No charge-off date is available,
  so a fixed-horizon PD cannot be constructed. 60-month loans issued in 2014 are partly
  unresolved; excluding them enriches those vintages with charge-offs (see below).

## Metrics (out-of-time test, 2014-07-01 to 2014-12-01)

| Metric | Selected model (calibrated) | Dummy baseline |
| --- | --- | --- |
| ROC-AUC | 0.6983 (95% CI [0.6949, 0.7021]) | 0.5000 |
| PR-AUC | 0.3501 | 0.1912 |
| Brier score | 0.1423 | 0.1558 |
| Log loss | 0.4489 | 0.4921 |
| Expected calibration error | 0.0073 | 0.0336 |
| Prevalence | 0.1912 | 0.1912 |
| ROC-AUC incl. excluded unresolved loans (realistic labels) | 0.6737 | - |
| ROC-AUC on fully resolved 36-month loans | 0.6530 | - |

**Headline caveat.** 9,236 loans issued in the test window were excluded as unresolved (concentrated in 2014 60-month vintages). They are survivors at the snapshot, so excluding them enriches the remaining 60-month loans with charge-offs. Scoring them with the selected model (lightgbm) and adding them back:

* realistic labelling (Current/Grace repaid, Late adverse): ROC-AUC **0.6737** (reported 0.6983); adverse prevalence 0.1800 vs mean calibrated prediction 0.1918; 60-month loans: observed 0.2632 vs predicted 0.2988;
* all unresolved repaid: ROC-AUC 0.6708;
* extreme bound (all unresolved adverse - implausible): ROC-AUC 0.7319.

On fully resolved 36-month loans (no exclusion bias) the ROC-AUC is **0.6530** with mean prediction 0.1419 vs observed 0.1413.

**Interpretation:** the reported test ROC-AUC (0.6983) is optimistic; a realistic estimate is about 0.674. On the reported test set 60-month loans appear under-predicted (observed 0.328 vs predicted 0.300); with realistic labels they are over-predicted (observed 0.263 vs predicted 0.299). The calibrator was fitted on 2014 Q2, which shares the same exclusion bias.

Threshold 0.202 (chosen on the calibration split): precision
0.312, recall 0.590, F1 0.408.
Full results: [evaluation_summary.md](evaluation_summary.md).

Other feature sets:
* `origination` (uses Lending Club grade/rate - origination-time information): selected `lightgbm`, calibrated test ROC-AUC 0.7150.

## Known limitations

* Retrospective outcome label on resolved loans only; not a fixed-horizon PD.
* Selection bias: only approved/funded loans are observed; performance on the
  through-the-door population is unknown.
* Exclusion bias: unresolved loans (2014 60-month vintages) are excluded from the
  calibration and test splits, so the reported test ROC-AUC and the calibration of
  60-month loans are optimistic (quantified above).
* Temporal drift: calibration was fitted on 2014 Q2 data and may not transfer to later
  periods.
* `loan_amnt` may reflect reductions by Lending Club's credit department (per its data
  dictionary); it is allow-listed as the only available amount field and flagged.
* Credit-bureau fields are taken as reported in the export; their exact pull dates
  are undocumented (`mths_since_last_major_derog` flagged as uncertain).
* No loan identifier exists in this export; duplicates were checked by fingerprint.
* SHAP explanations describe model behaviour, not causal effects, and can be unstable
  across correlated features.

## Fairness considerations

* No protected attributes (race, sex, age, etc.) exist in the data; none were inferred.
* Geography (`zip_code`, `addr_state`) and free text were excluded as potential proxies.
* `earliest_cr_line` (credit-history length) correlates with age; it is a standard bureau
  attribute but should be reviewed for disparate impact in any real application.
* Segment analysis (income, loan amount, credit history, employment, home ownership, term)
  is descriptive; it does not establish fairness under any legal standard.

## Data provenance

Sharar, M. Labib (2021). *12_13_14 Excel.xlsx*. figshare. Dataset. https://doi.org/10.6084/m9.figshare.15074250.v1 (licence stated on Figshare: CC0 1.0). Originally published by Lending Club. The Figshare description
says "funded and matured loans", but the file contains loans still `Current`/`Late` at
the snapshot. Upstream Lending Club terms may apply independently of the Figshare
licence. The project code does not need the raw file in version control: place it
locally under `data/raw/` (git-ignored); see the README data-provenance notes.

## Monitoring considerations (if this were operationalised)

* Population stability (PSI) of inputs and scores by vintage.
* Calibration-in-the-large and by risk bucket as outcomes mature.
* Discrimination on matured cohorts; vintage default curves.
* Rate of unseen categories and out-of-range inputs at inference.
* Override and adverse-action reason governance; periodic fairness review.
