# CreditIQ: Explainable Credit Risk Scoring & Decision Intelligence

**Lending Club 2012–2014 data (CC0)**

> **Educational portfolio project.** Nothing in this repository is validated or approved for real-world lending decisions. Model scores are **not** regulatory probabilities of default.

CreditIQ is an end-to-end, reproducible credit-risk system:

Data validation → target definition → feature-availability (leakage) audit → feature engineering → Dummy / Logistic Regression / LightGBM → probability calibration → out-of-time evaluation with uncertainty and bias bounds → SHAP, LIME and permutation importance → topological data analysis (Mapper, persistent homology) → segment analysis → a **hypothetical** decision simulation → an interactive Streamlit dashboard.

Every number in this README was produced by the pipeline on the real dataset; the full, generated results are in [`reports/evaluation_summary.md`](reports/evaluation_summary.md).

**Technology stack:** Python 3.12 · pandas · NumPy · PyArrow · DuckDB (SQL) · scikit-learn · LightGBM · SHAP · LIME-style surrogates (own implementation) · SciPy · NetworkX · ripser · Altair / Matplotlib / Seaborn · Streamlit · pytest · Ruff · GitHub Actions.

**Repository:** https://github.com/oilegd/CreditIQ

**Live Demo:** https://creditiq-riskanalysis.streamlit.app/

---

## 1. Project overview

| Component | Where |
| --- | --- |
| Python package (`creditiq`) | [`src/creditiq/`](src/creditiq) |
| Configuration (paths, target, splits, feature catalogue, models, explainability, TDA, decision assumptions) | [`config/config.yaml`](config/config.yaml) |
| Command-line pipeline | [`scripts/`](scripts) |
| DuckDB SQL analytics | [`sql/`](sql) |
| Notebooks 01–08 (thin layers over the package) | [`notebooks/`](notebooks) |
| Streamlit dashboard (8 pages) | [`app/streamlit_app.py`](app/streamlit_app.py) |
| Reports: data dictionary, leakage audit, model card, evaluation summary | [`reports/`](reports) |
| Tests (synthetic fixtures only) + CI | [`tests/`](tests), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

## 2. Business problem

A lender must decide, at application time, whether (and at what risk) to extend credit.

The model estimates the probability that a loan ends in an **adverse outcome** using only information available at that point.

Explanations show *why* a score is high, topological maps show *where* in the applicant space the model is weak, and a separate simulation shows how scores *could* inform cut-offs under explicit, hypothetical economics.

## 3. Dataset source and licensing notes

- **Source:** Sharar, M. Labib (2021). *12_13_14 Excel.xlsx*. Figshare. Dataset. https://doi.org/10.6084/m9.figshare.15074250.v1
  - One sheet (`2012_13_14`)
  - 423,810 loans × 51 columns
  - Approximately 125 MB
  - Original publisher of the loan records: Lending Club.

- **Stated licence on Figshare:** CC0 1.0. A Figshare licence covers the uploader's contribution; it does **not** necessarily extinguish rights or terms attached by the original publisher (Lending Club). Treat upstream terms as potentially applicable.

- **Verification:** The local file's MD5 `244d12efa7b8c845263bf26b0d49ae7f` equals Figshare's published `computed_md5` (recorded in [`reports/data_dictionary.md`](reports/data_dictionary.md)).

- **Provenance caveat:** The Figshare description says "funded and matured loans", but the file contains 12,536 loans that were still `Current`, `Late` or `In Grace Period` at the snapshot (about March 2019).

- **Repository artifacts:** The original Lending Club workbook is not included in the current repository. Selected trained model bundles, evaluation outputs and reduced TDA Parquet samples were added for the public Streamlit demonstration. The repository does not use Git LFS. The MIT [`LICENSE`](LICENSE) covers only original code.

## 4. Modelling assumptions

| Item | Decision |
| --- | --- |
| Prediction point | Underwriting of a new application. Application dates are absent; `issue_d` (issue month) is the closest proxy. |
| Eligible population | Funded Lending Club loans issued 2012-01 to 2014-12 with a **resolved** status at the snapshot. |
| Positive (adverse) | `Charged Off` (70,829) or `Default` (1). |
| Negative | `Fully Paid` (340,444). |
| Excluded | `Current` (11,925), `In Grace Period` (201), `Late (16-30 days)` (73), `Late (31-120 days)` (337), unresolved, **not** treated as good. |
| Outcome horizon | Loan lifetime as observed at the snapshot. The export has **no charge-off date**, so a fixed-horizon (e.g. 12-month) PD cannot be built; this is a retrospective loan-outcome model, not a production PD. |
| Selection bias | Only approved, funded loans are observed (no rejected applicants). |
| Exclusion bias | All 36-month loans resolved, but 3%–24% of 60-month loans issued in 2014 were unresolved. Excluded loans are *survivors*, so excluding them enriches 2014 60-month loans with charge-offs. The pipeline scores them and reports bounds (section 12). |

## 5. Repository architecture

```text
config/config.yaml        single source of truth

src/creditiq/
  config.py, cli.py       config, project root, logging; shared script helpers
  data/                   load (CSV/Excel/Parquet + cache) · validate · clean · target · split · pipeline · synthetic
  features/               build_features (stateless engineering + train-only preprocessors) · leakage_checks
  models/                 train (bounded grids, tie-break) · calibrate · evaluate · evaluation_run · predict · artifacts
  explainability/         shap_analysis · lime_tabular · permutation (single + grouped) · suite (agreement)
  tda/                    mapper · lenses · persistence · point_cloud · insights · build
  risk/                   decision_analysis (EL, cut-offs, sensitivity, buckets) · segments
  analytics/              duckdb_queries (named-query runner for sql/*.sql)
  reporting/              figures (static) · interactive (Altair) · markdown writers · results loader

scripts/                  inspect_data → prepare_data → train_model → evaluate_model → generate_reports

app/streamlit_app.py      dashboard (reads saved artifacts and local samples only; never trains)
```

## 6. Installation (Windows PowerShell)

Requires Python **3.11 or 3.12**.

Using [uv](https://docs.astral.sh/uv/) (or plain `venv`):

```powershell
# From the repository root
uv venv --python 3.12 .venv

# Or:
# py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

uv pip install -e ".[dev,tda]"

# Or:
# python -m pip install -e ".[dev,tda]"
```

The optional `tda` extra installs `ripser` for H1 (loop) persistence; everything else works without it.

Exact validated versions: [`requirements.txt`](requirements.txt).

macOS/Linux: `source .venv/bin/activate`; all other commands are identical.

## 7. Data placement

Place the workbook at **`data/raw/12_13_14 Excel.xlsx`** (path configurable via `paths.raw_file`), downloaded from the Figshare DOI above.

If a file at that path is a Git LFS *pointer* instead of the real workbook, ingestion fails with an explicit message.

## 8. Data preparation

```powershell
python scripts/inspect_data.py
python scripts/prepare_data.py
```

The first run parses the Excel file (~4–5 minutes) and caches it as Parquet; later runs take seconds.

`prepare_data.py` writes `reports/leakage_audit.md` and aggregate SQL outputs.

## 9. Training

```powershell
python scripts/train_model.py --feature-set all
```

## 10. Evaluation, explanations, TDA and reports

Run each command on its own line:

```powershell
python scripts/evaluate_model.py --feature-set all
python scripts/generate_reports.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

`evaluate_model.py` uses the untouched out-of-time test set (bootstrap CIs, unresolved-loan bounds, segments, decision analysis).

`generate_reports.py` computes SHAP, single and grouped permutation importance and LIME for both models, builds the TDA point clouds and figures, and regenerates all Markdown reports (~4 minutes on a laptop).

Synthetic demo (no real data needed; every output isolated under `data/synthetic/`): run the same scripts with `--synthetic`, e.g. `python scripts/inspect_data.py --synthetic`.

## 11. Streamlit dashboard

**Live application:** https://creditiq-riskanalysis.streamlit.app/

Run locally:

```powershell
streamlit run app/streamlit_app.py
```

| Page | What it offers |
| --- | --- |
| Overview | KPIs vs the dummy baseline, the headline-AUC caveat, out-of-time timeline, data funnel |
| Portfolio Analysis | Interactive vintage curves, unresolved share, segment explorer, risk gradients |
| Model Performance | ROC/PR (legend highlighting), reliability, **threshold explorer** (live confusion matrix), robustness including bias bounds, segments |
| Explainability | SHAP (signed, raw-aggregated, dependence), single + grouped PFI by metric, global LIME with fidelity, SHAP-PFI-LIME agreement |
| Risk Prediction | Grouped input form → calibrated PD, risk bucket, SHAP, LIME (with stability check), what-if (ICE) curves |
| Decision Simulation | Adjustable LGD/EAD/margin/cut-off, net value and approval curves, break-even PD, bucket calibration |
| Topology (TDA) | Interactive Mapper graph (lens, resolution, overlap, scale; zoom/pan/click-to-inspect), PCA view, persistence diagrams and barcodes, evidence-based improvement ideas, stability across resolutions |
| Model Documentation | Model card, leakage audit, evaluation summary, data dictionary, methodology |

The app never trains; it caches saved artifacts, aggregate tables and local TDA samples.

Required runtime artifacts are included in the public deployment repository. If an artifact is unavailable, the affected page displays a setup message instead of predictions.

## 12. Model comparison results (measured)

Out-of-time **test** set: loans issued 2014-07 to 2014-12, n = 123,634, adverse prevalence 19.12%.

Train: 2012-01 to 2013-12 (n = 188,171).

Validation: 2014 Q1 (47,013).

Calibration: 2014 Q2 (52,456).

Selection, calibration and thresholds never used the test set (verified by retraining with scrambled test labels and perturbed test features: bit-identical artifacts).

| Feature set | Model | Test ROC-AUC [95% CI] | Incl. excluded loans (assumed scenario†) | 36-month only | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| - | Dummy (prior) | 0.500 | - | - | 0.191 | 0.1558 |
| applicant | Logistic regression | 0.695 [0.692, 0.699] | 0.670 | 0.647 | 0.348 | 0.1426 |
| **applicant** | **LightGBM - selected** | **0.698 [0.695, 0.702]** | **0.674** | **0.653** | **0.350** | **0.1423** |
| origination | Logistic regression | 0.712 [0.708, 0.715] | 0.692 | 0.675 | 0.354 | 0.1409 |
| origination | LightGBM - selected | 0.715 [0.711, 0.719] | 0.697 | 0.681 | 0.364 | 0.1404 |

† Hypothetical: the outcomes of excluded unresolved loans are unknown and are **assumed** (Current/Grace → repaid, Late → adverse). These are sensitivity values, not observed results.

**How to read this:**

- **The reported test ROC-AUC (0.698) is likely optimistic.** 9,236 test-window loans (2014 60-month survivors) were excluded as unresolved; their true outcomes are unknown. Adding them back under the assumed labels above gives 0.674 for the selected model (assuming all repaid: 0.671; implausible all-adverse bound: 0.732). The only exclusion-free observed result is on fully resolved 36-month loans: 0.653 (`term`, the strongest predictor, is constant there).

- **Calibration.** Calibration on 2014 Q2 reduced the reported test ECE from 0.031 to 0.007, and 36-month loans are well calibrated on observed outcomes (predicted 0.142 vs observed 0.141). The calibration split shares the exclusion bias, however; under the assumed scenario 60-month loans would be over-predicted (scenario rate 0.263 vs predicted 0.299). Consistent with this, the TDA diagnostics on the validation split find regions dominated by long-term loans where the mean prediction exceeds the observed adverse rate.

- Both models clearly beat the dummy. LightGBM's AUC edge over logistic regression is statistically detectable but practically small: +0.0029 (paired bootstrap 95% CI [0.0014, 0.0042]). Bootstrap CIs resample loans i.i.d. within one 6-month window and do not capture period-to-period variation or the exclusion bias, which is larger.

- A stratified random split gives AUC 0.689, between the assumed-scenario (0.674) and reported (0.698) out-of-time values.

- Lending Club's own grade and rate (origination model) add about +0.02 AUC. That information encodes Lending Club's own underwriting and is not available pre-decision.

- At the F1-optimal threshold chosen on calibration data (0.202): precision 0.312, recall 0.590, F1 0.408.

These discrimination levels are typical of application-only Lending Club data without bureau scores (this export has no FICO fields).

## 13. Explainability methodology

| Method | Implementation | Notes |
| --- | --- | --- |
| **SHAP** | TreeSHAP (LightGBM); exact linear SHAP `coef × (x - training mean)` (LogReg, from stored means, no training rows) | Signed direction, binned dependence curves, attribution aggregated back to raw inputs |
| **Permutation importance (PFI)** | Own implementation on raw inputs; ROC-AUC, log loss and Brier on calibrated scores | **Single-input** and **grouped** (related inputs shuffled jointly, e.g. income + loan amount) to handle correlation |
| **LIME** | Tabular LIME (Ribeiro et al., 2016) with quartile discretisation; perturbations sampled from stored training quantiles/frequencies | Reports local fidelity R² and stability across seeds |

On the applicant model, the three methods agree closely for LightGBM (Spearman rank correlations 0.95–0.96) and reasonably for logistic regression (0.75–0.90).

The top raw inputs are `term`, `annual_inc`, `revol_util`, `inq_last_6mths` and `open_acc`.

Grouped PFI ranks `loan_terms`, `income_and_loan_size` and `revolving_credit` highest.

Mean LIME local R² is about 0.45, so LIME's linear story is a rough approximation; treat it as a complement to SHAP.

None of these methods measure causal effects.

## 14. Topological data analysis

- **Point cloud:** A 6,000-loan sample per split, described by the applicant model's engineered numeric features. Values are median-imputed and robustly standardised using **training** statistics.

- **Mapper** (Singh, Mémoli & Carlsson, 2007), implemented in `creditiq.tda.mapper`:
  - Lens (model PD, PCA, k-NN density, or 2-D combinations)
  - Rank-balanced overlapping cover
  - Single-linkage clustering at a k-NN-based scale
  - Nerve graph

- **Diagnostics:** Nodes are coloured by observed rate, predicted PD, calibration gap (with binomial z-scores), risk lift and within-node AUC.

- **Improvement ideas** are generated only from nodes with ≥ 60 loans and |z| ≥ 2.5. They are hypotheses to test on later data, not validated improvements.

- **Default graph** (validation split, PD lens): 65 nodes, with 4 calibration findings, 2 high-risk flares and 3 weak-discrimination regions. Miscalibrated regions appear under every lens and resolution tested (notebook 08).

- **Persistent homology:** H0 is computed from single-linkage merge heights; H1 uses `ripser`. It compares adverse and repaid loans as a descriptive geometric summary.

## 15. Leakage prevention approach

1. **Availability audit** of all 51 source columns (pre-underwriting, at-origination, outcome-derived, sensitive, uncertain, identifier/metadata), each with a written rationale ([`reports/leakage_audit.md`](reports/leakage_audit.md)).

2. **Two feature sets:**
   - `applicant` is primary and uses applicant and bureau fields only.
   - `origination` is clearly labelled and adds sub-grade, rate, instalment, verification and listing status.

   `loan_amnt` is flagged as **uncertain**: Lending Club's dictionary says credit-department reductions are reflected in it.

3. **Hard guards:** Feature sets are validated against the audit, and a hard-coded deny-list rejects outcome fields even if the YAML is mis-edited.

4. **Empirical screen:** Single-column AUC on the training split. For example, `recoveries` alone reaches AUC 0.97.

5. **Train-only preprocessing; schema-enforced inference.** Extra columns, including outcome fields, never reach a model.

6. **Tests** cover all of the above, including bit-identical retraining with scrambled test labels.

## 16. Limitations

- Retrospective lifetime outcome on resolved loans; not a fixed-horizon or regulatory PD.
- Selection bias (approved loans only) and exclusion bias in 2014 60-month vintages. Headline metrics are likely optimistic; assumed-outcome sensitivity bounds are reported.
- Calibration was fitted on a split with the same exclusion bias. 60-month PDs are likely overstated.
- Temporal drift: one 6-month test window; calibration may not transfer.
- No bureau scores, no application timestamps, no loan identifier (duplicates checked by fingerprint; none found).
- `loan_amnt` and `mths_since_last_major_derog` are flagged as uncertain availability.
- Decision economics (LGD 0.85, EAD 100%, margin 12%) are **hypothetical**.
- Segment and TDA analyses are descriptive. No protected attributes exist and none were inferred; similar metrics across groups do not establish fairness.

## 17. Future improvements

- Survival (time-to-default) modelling to treat unresolved loans as censored instead of excluding them.
- Calibrate on fully matured cohorts (or per term) and re-validate on a later period.
- Rolling-origin evaluation and block bootstrap by issue month.
- Test TDA-suggested segment recalibration on later vintages; monitor topological drift.
- Reject inference; monotonic constraints and a scorecard (WoE/IV) baseline.

## 18. Deployment (Streamlit Community Cloud)

CreditIQ has been deployed using **Streamlit Community Cloud**.

**Live application:** https://creditiq-riskanalysis.streamlit.app/

**GitHub repository:** https://github.com/oilegd/CreditIQ

Deployment configuration:

| Setting | Value |
| --- | --- |
| Repository | `oilegd/CreditIQ` |
| Branch | `main` |
| Entry point | `app/streamlit_app.py` |
| Python | 3.12 |
| Dependencies | Root `requirements.txt` |
| Deployment platform | Streamlit Community Cloud |

The application runs with the dependencies in `requirements.txt` (falling back to `src/` when the `creditiq` package is not installed) and reads [`.streamlit/config.toml`](.streamlit/config.toml), which hides error details from visitors.

### Runtime artifacts

The public repository includes selected runtime artifacts required by Streamlit:

- **Trained models:** `models/<feature_set>/*.joblib` and `metadata.json`.
- **Evaluation outputs:** Selected files under `reports/metrics/`.
- **TDA artifacts:** Reduced Parquet samples under `data/processed/tda/`.

These files allow Streamlit to load previously generated results without retraining the machine learning models.

The original Lending Club Excel workbook is not included in the repository.

The published TDA Parquet files contain reduced, row-level derived samples rather than exclusively aggregated diagrams. Joblib artifacts use Python pickle serialization and should only be loaded from trusted sources.

### Runtime characteristics

Measured locally:

- Approximately 340 MB peak memory across dashboard pages.
- Approximately 5 seconds for the initial application render.
- Approximately 1–3 seconds for Mapper reconstruction, with caching enabled.

Actual cloud performance depends on available resources.

### Local execution

```powershell
streamlit run app/streamlit_app.py
```

The app does not retrain models during inference. It reads precomputed model bundles, evaluation outputs and TDA artifacts.

If required runtime artifacts are unavailable, the affected page displays a setup message rather than fabricated predictions.

---

### Reproducibility notes

- **Seeds:** Fixed via `project.random_seed`.
- **Splits:** Date-based.
- **Tie-break:** Grid candidates within 0.0005 validation AUC are broken by log loss.
- **Artifacts:** Record library versions and the full config (`models/<set>/metadata.json`).
- **Platform:** Validated on Windows 11 with Python 3.12.13 and the versions in [`requirements.txt`](requirements.txt).