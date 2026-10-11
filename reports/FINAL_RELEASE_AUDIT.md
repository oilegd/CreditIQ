# CreditIQ v1.0 - Final Security, Debugging & Release Audit

Audit date: 2026-10-12 · Scope: current working tree on branch `main` (local commit
`973e209` plus the uncommitted audit fixes listed in section J) · Auditor: AI assistant
(GitHub Copilot SDK in VS Code).

Status legend: **PASS** = verified pass · **FAIL** = verified failure · **NOT TESTED** ·
**EXTERNAL** = requires action by the repository owner.

---

## A. Executive summary

* **Overall status:** the code, models, dashboard and documentation pass every engineering
  check run in this audit:
  * 128/128 tests pass; Ruff lint and format pass
  * all 8 dashboard pages × 2 feature sets render with 0 exceptions on real artifacts
  * a clean-environment deployment simulation passes
  * pip-audit reports no known vulnerabilities
* **Release readiness: NOT READY - one blocking issue remains (C1, external).** The public
  GitHub repository `oilegd/CreditRiskPredictionBankingI` still serves the original commit
  `8c03b42`. The 130 MB Lending Club workbook is still downloadable from its Git LFS storage
  (checked via the LFS batch API on 2026-10-12). Local history is clean, but rewriting history
  does not purge remote LFS storage. Only the owner can fix this (delete and recreate the
  repository, or ask GitHub Support).
* **Primary accomplishments of this pass:**
  * a deployment blocker fixed (app import path)
  * public error-detail exposure disabled
  * unbounded TDA caches bounded
  * CI hardened
  * hypothetical-scenario results re-worded so they are no longer presented as observed
  * the test-set isolation re-verified on the current models
* **Remaining risks:**
  * C1 (critical, external)
  * runtime artifacts are git-ignored, so a deployed app needs an owner decision on publishing
    them (high, external)
  * headline AUC is likely optimistic (documented)
* **Recommended next action:**
  1. Delete and recreate the GitHub repository; re-run the LFS check in section G.
  2. Review and commit the audit changes.
  3. Decide which runtime artifacts to publish (section K).
  4. Push.
  5. Deploy.

## B. Repository inventory

| Area | Contents |
| --- | --- |
| Package `src/creditiq/` | `config`, `cli`; `data/` (load, validate, clean, target, split, pipeline, synthetic); `features/` (build_features, leakage_checks); `models/` (train, calibrate, evaluate, evaluation_run, predict, artifacts); `explainability/` (shap_analysis, lime_tabular, permutation, suite); `tda/` (mapper, lenses, persistence, point_cloud, insights, build); `risk/` (decision_analysis, segments); `analytics/duckdb_queries`; `reporting/` (figures, interactive, markdown, results) |
| Scripts | `inspect_data.py`, `prepare_data.py`, `train_model.py`, `evaluate_model.py`, `generate_reports.py` |
| SQL (DuckDB) | `sql/01_data_quality.sql`, `02_portfolio_analysis.sql`, `03_feature_queries.sql` |
| Notebooks | 01 data understanding · 02 EDA · 03 feature engineering · 04 baseline · 05 advanced modelling · 06 evaluation · 07 explainability · 08 TDA (all executed; aggregates only) |
| Dashboard | `app/streamlit_app.py` - 8 pages: Overview, Portfolio Analysis, Model Performance, Explainability, Risk Prediction, Decision Simulation, Topology (TDA), Model Documentation |
| Model artifacts (git-ignored) | `models/{applicant,origination}/` - `dummy`, `logistic_regression`, `lightgbm` `.joblib` bundles, `metadata.json`, tuning/validation/calibration CSVs (0.7 MB total) |
| Evaluation outputs (git-ignored) | `reports/metrics/` (0.4 MB, aggregate tables); `data/processed/tda/` (1.5 MB, row-level samples) |
| Tests | 8 modules + `conftest.py` (synthetic fixtures only), 128 tests; CI in `.github/workflows/ci.yml` |
| Reports (tracked) | data dictionary, leakage audit, model card, evaluation summary, this report, 43 figures |

## C. Data and modelling

* **Dataset:** Figshare "12_13_14 Excel.xlsx" (DOI 10.6084/m9.figshare.15074250, stated
  licence CC0; original publisher Lending Club). 423,810 loans × 51 columns. MD5 matches
  Figshare.
* **Target:** `Charged Off` (70,829) or `Default` (1) = 1; `Fully Paid` (340,444) = 0.
  12,536 unresolved loans (`Current`, `Late`, `In Grace Period`) are excluded. 411,274
  modelled loans, prevalence 17.2%. This is a lifetime outcome; no fixed-horizon PD is
  possible because the export has no charge-off date.
* **Cleaning:**
  * 0 duplicates (by fingerprint)
  * 7 `NONE`/`ANY` home ownership values mapped to `OTHER`
  * 2 `revol_util` values outside [0, 2] set to NaN
* **Splits (by issue month):**

  | Split | Period | Loans |
  | --- | --- | --- |
  | Train | 2012-01 to 2013-12 | 188,171 |
  | Validation | 2014 Q1 | 47,013 |
  | Calibration | 2014 Q2 | 52,456 |
  | Test | 2014 H2 | 123,634 |

* **Features:**
  * *applicant* (primary): 18 pre-underwriting fields, plus `loan_amnt` and
    `mths_since_last_major_derog` flagged as uncertain availability, plus `issue_d` as the
    reference date only
  * *origination*: adds sub-grade, rate, instalment, verification status and listing status
  * leakage guards: catalogue categories plus a hard-coded deny-list; a leakage test suite
    passes
* **Models:**
  * Dummy, Logistic Regression and LightGBM, with bounded grids
  * selection by validation ROC-AUC; ties within 0.0005 are broken by validation log loss
  * calibration (none / Platt / isotonic) chosen by 5-fold CV inside the calibration split
  * threshold: F1-optimal on the calibration split
* **Test isolation: PASS (re-verified on the current models in this audit).** Retraining with
  scrambled test labels and perturbed test features reproduced the selected models,
  hyper-parameters, calibration methods, thresholds, input schemas and predictions exactly
  (max |Δp| = 0) for both feature sets.
* **Test results (resolved loans, calibrated, 95% bootstrap CI):**

  | Model | ROC-AUC [95% CI] | PR-AUC | Brier | ECE |
  | --- | --- | --- | --- | --- |
  | Applicant LightGBM (selected) | 0.698 [0.695, 0.702] | 0.350 | 0.1423 | 0.007 |
  | Applicant LogReg | 0.695 [0.692, 0.699] | - | - | - |
  | Origination LightGBM | 0.715 [0.711, 0.719] | - | - | - |

* **Exclusion-bias sensitivity (H1/H2).** The outcomes of excluded loans are unknown; these
  are assumed scenarios, not observed results. For the selected applicant model:

  | Population / scenario | ROC-AUC |
  | --- | --- |
  | Excluded loans assumed: Current/Grace repaid, Late adverse | 0.674 |
  | Excluded loans assumed: all repaid | 0.671 |
  | Implausible bound: all excluded loans adverse | 0.732 |
  | Observed, fully resolved 36-month loans | 0.653 |

  On 36-month loans the mean prediction is 0.142 vs 0.141 observed.
* **Known limitations:**
  * selection bias (funded loans only)
  * exclusion bias (2014 60-month loans)
  * calibrator fitted on a split with the same bias
  * one 6-month test window
  * bootstrap CIs assume i.i.d. loans
  * no bureau scores or application timestamps
  * decision economics are hypothetical

### C1 / H1 / H2 re-check

| Finding | How it was addressed | Current verdict |
| --- | --- | --- |
| C1 dataset exposure | Local single clean commit `973e209`; LFS rule removed; reflog-only dataset blob pruned; `.gitignore` covers data formats | **Local: PASS. Remote: FAIL (EXTERNAL)** - public repo still at `8c03b42`; LFS object still downloadable |
| H1 optimistic AUC | Pipeline scores excluded loans and reports scenario bounds plus the 36-month-only metric in the README, model card, evaluation summary and app | **PASS after this audit's wording fix** - earlier text called a scenario "realistic" and its rates "observed"; now labelled as assumed scenarios |
| H2 calibration claims | Same bounds, plus 60-month calibration under the scenario; TDA cross-check on validation | **PASS after wording fix** - the claim "60-month loans are over-predicted" is now conditional on the assumed scenario |

## D. Explainability

* **SHAP:**
  * TreeSHAP for LightGBM; exact linear SHAP for LogReg, computed as `coef × (x − training
    mean)` from stored aggregate means
  * additivity is tested to 1e-6
  * the linear version equals `shap.LinearExplainer` on the full training set (tested)
  * values are in log-odds, before calibration
* **LIME:**
  * own tabular LIME (Ribeiro et al., 2016), with quartile discretisation and categorical
    equality
  * perturbations come from stored training quantiles and frequencies, not training rows
  * the reference date is held fixed; seeded and reproducible (tested)
  * reports surrogate R² (applicant mean ≈ 0.45) and a stability check across seeds
* **PFI:**
  * own implementation on the test split, with ROC-AUC on scores and log loss / Brier on
    calibrated probabilities
  * single-input and grouped (correlated inputs shuffled jointly)
  * descriptive only; never used for tuning
* **Agreement (applicant):** SHAP/PFI/LIME rank correlations are 0.95-0.96 for LightGBM and
  0.75-0.90 for LogReg. Top inputs: `term`, `annual_inc`, `revol_util`, `inq_last_6mths`,
  `open_acc`.
* **Limitations:** none of the three methods is causal; correlated inputs share attribution;
  LIME fidelity is moderate.

## E. Topological data analysis

* **Methods:**
  * Mapper (Singh, Mémoli & Carlsson 2007), implemented in `creditiq.tda.mapper`
  * H0 persistence from single-linkage merge heights (exact for Vietoris-Rips H0)
  * H1 via `ripser`
  * PCA projection for visual intuition only
* **Pre-processing:** a 6,000-loan sample per split (fixed seed). Features are the applicant
  model's engineered numerics, median-imputed and robustly standardised with training-split
  statistics, then clipped to ±5.
* **Mapper:**
  * lens: PD, PCA 1, k-NN density, or 2-D combinations
  * rank-balanced overlapping cover
  * single-linkage clustering cut at a k-NN distance quantile
  * nerve graph
* **Correctness checks (tests):**
  * a circle produces a cycle
  * two separated blobs produce two components
  * H1 of a circle has one dominant bar
  * edge weights equal the members shared between nodes
* **Supported findings (validation split, PD lens, 65 nodes):** 4 nodes with significant
  calibration gaps (|z| ≥ 2.5, n ≥ 60), mostly long-term-loan regions where predictions exceed
  observed rates. Regions with |z| > 2.5 appear under all 5 lenses × 3 resolutions (notebook 08).
* **Interpretation safeguards:**
  * ideas are labelled as hypotheses to validate on later data
  * the persistence comparison is described as descriptive geometry
  * no causal claims are made
* **Resource control:** sliders are bounded; Mapper and persistence results are cached with
  `max_entries` (added in this audit).

## F. Streamlit application

* **Pages checked:** all 8, for both feature sets, against real artifacts (AppTest, 0
  exceptions). Also checked earlier in a real browser session.
* **Interactions tested:**
  * threshold slider (live confusion metrics)
  * Explainability model switch
  * Risk Prediction form submission with both models (PD, raw score, SHAP, LIME R²)
  * TDA lens changes (PCA 1: 47 nodes; PD × PCA 1: 238; k-NN density: 73)
  * TDA split change to test
  * Decision Simulation LGD slider
* **Missing-artifact behaviour:** every page shows a setup message; no predictions are shown
  (tested).
* **Performance:** startup ≈5 s; TDA page 2.6-3.1 s on first render (cached afterwards); peak
  RSS ≈341 MB across all pages.
* **Outstanding:**
  * Mapper node-click selection was exercised in a real browser earlier but is not covered by
    AppTest (**NOT TESTED** automatically).
  * The LIME stability toggle and the TDA "Highlight on map" buttons were not exercised in this
    audit (**NOT TESTED**).

## G. Security audit

| Check | Result |
| --- | --- |
| Secrets in tracked files | **PASS** - regex scan for keys/tokens/passwords/AWS/GitHub tokens found only a comment in `.env.example` and an ignore rule for `.streamlit/secrets.toml`; notebook hit was a base64 image false positive |
| Raw/derived data in intended history | **PASS (local)** - `973e209` has no data/model files; `.gitignore` covers xlsx/xls/csv/parquet, `models/`, `reports/metrics/`, `data/*` |
| Remote history / LFS | **FAIL (EXTERNAL)** - `git ls-remote` → `8c03b42`; LFS batch API returns a download URL for OID `5f5f78c6…` (130,823,263 bytes); repo is public |
| Local agent refs | `refs/agents/*` (VS Code checkpoints) contain code only; local, never pushed - do not use `git push --mirror` |
| Model deserialisation | **PASS** - joblib loads only from the configured server-side `models/` directory; no upload or user-supplied paths exist |
| User input | **PASS** - bounded number/select/date widgets derived from the training schema; schema coercion drops extra columns |
| HTML rendering | **PASS** - `unsafe_allow_html` used only for static CSS, config-defined bucket labels and computed numbers |
| Error disclosure | **FIXED** - `.streamlit/config.toml` sets `showErrorDetails = "none"` (verified effective) |
| Resource exhaustion | **FIXED** - TDA caches bounded (`max_entries` 32/8); sliders bounded |
| Dependencies | **PASS** - `pip-audit` on the frozen environment: "No known vulnerabilities found" (2026-10-12); `requirements.txt` pins all runtime packages |
| CI | **PASS / hardened** - `permissions: contents: read`, `pull_request` (not `_target`), no secrets, LFS not fetched, synthetic data only; added `persist-credentials: false` and `timeout-minutes: 30`. Actions are pinned to major tags (`@v4`, `@v5`), not SHAs (low residual risk, first-party actions) |
| Personal data | Commit author e-mail will be public on push (owner choice) |

## H. Testing results

| Check | Result |
| --- | --- |
| `pytest` (full suite) | **128 passed, 0 failed, 0 skipped** (51 s) |
| Ruff lint / format check | **PASS** / **PASS** (63 files) |
| Coverage | NOT TESTED (not measured) |
| Real-artifact AppTest (all pages, both feature sets, interactions) | **PASS**, 0 exceptions |
| Clean Python 3.12 env with only `requirements.txt` (package not installed) | **PASS** - 8 pages, 0 exceptions |
| Streamlit server start + `/_stcore/health` | **PASS** (`ok`) |
| Test-set isolation re-check on real data (current models) | **PASS** (bit-identical) |
| CI synthetic end-to-end pipeline (previous pass, unchanged code paths) | **PASS** (226 s) |
| Notebooks | Executed in the previous pass; not re-run (no code they use changed except report wording) |

New regression test: `tests/test_inference.py::test_bias_paragraph_labels_scenarios_as_assumed`.

## I. External reference review

| Resource (actually inspected) | Applied finding |
| --- | --- |
| Streamlit docs - App dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies | Requirements file at repo root or entry-point dir; `requirements.txt` takes priority over `pyproject.toml`; Python version must match → import-path fallback added, Python 3.12 documented |
| Streamlit docs - Manage your app: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app | Resource limits (690 MB-2.7 GB RAM, as of Feb 2024); pin Streamlit → memory measured (≈341 MB); Streamlit already pinned |
| Local `streamlit config show` (v1.65) | `client.showErrorDetails` values → set `"none"` |
| GitHub docs - Removing files from Git LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage | LFS objects persist after history rewrite; deletion requires deleting the repo or GitHub Support → C1 remediation |
| GitHub docs - Removing sensitive data: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository | Clones/forks/cached views may retain data → stated in remaining issues |
| Figshare API https://api.figshare.com/v2/articles/15074250 | Licence CC0, checksum, author (earlier pass) |
| Lending Club data dictionary (via web search, earlier pass) | `loan_amnt` may reflect credit-department reductions → flagged uncertain |
| pip-audit (PyPA tool, via `uvx`) | Vulnerability scan |

Not inspected in this audit:
* OWASP guidance, scikit-learn/LightGBM/SHAP/LIME example repositories, giotto-tda (not used by
  this project) and the GitHub Actions security-hardening guide.
* The CI changes follow commonly documented practice (`persist-credentials: false`, job
  timeouts) without a fresh document fetch.

## J. Changes made in this audit

| # | File | Problem | Severity | Change | Validation |
| --- | --- | --- | --- | --- | --- |
| 1 | `app/streamlit_app.py` | `import creditiq` fails on platforms that install only `requirements.txt` (Community Cloud) | High (deployment blocker) | Fallback adds `<repo>/src` to `sys.path` only if the package is not installed | Clean 3.12 env without the package: 8 pages, 0 exceptions |
| 2 | `.streamlit/config.toml` (new) | Default Streamlit shows full stack traces to visitors | Medium | `showErrorDetails = "none"`, `gatherUsageStats = false` | `streamlit config show` reflects values; server health ok |
| 3 | `app/streamlit_app.py` | `run_mapper` / `run_persistence` caches unbounded (one entry per slider combination) | Medium | `max_entries=32` / `8` | Full suite + AppTest pass |
| 4 | `src/creditiq/reporting/markdown.py`, regenerated `reports/evaluation_summary.md`, `reports/model_card.md`; `README.md`; app Overview caveat | Hypothetical scenario called "realistic", scenario rates called "observed", "is optimistic" stated as fact; app claimed "all 60-month" without data support in generic code | High (scientific wording) | Labelled as assumed scenarios / scenario rates; "likely optimistic" with reasoning; 36-month result marked as the observed exclusion-free metric | Report diffs are wording-only (all numbers unchanged); new regression test passes |
| 5 | `.github/workflows/ci.yml` | Checkout persisted the token; no job timeout | Low | `persist-credentials: false`, `timeout-minutes: 30` | YAML unchanged otherwise |
| 6 | `README.md` | Missing tech stack, repository link, deployment instructions and artifact caveats | Low | Added; no deployment URL claimed | Manual review |
| 7 | `tests/test_inference.py` | No guard against scenario/observed wording regression | - | 1 regression test | Passes |

No models were retrained, no saved evaluation numbers changed, no Git history was modified,
nothing was committed or pushed.

## K. Deployment readiness (Streamlit Community Cloud)

* **Entry point** `app/streamlit_app.py`. **Python 3.12** must be selected (pins require
  ≥ 3.11). Dependencies: root `requirements.txt` (includes dev tools; harmless, adds install
  time - 15 s locally with uv). No secrets needed.
* **Portability:** no Windows-specific paths (root detection via `pyproject.toml`/`config/`);
  verified in a fresh environment.
* **Resources:** ≈341 MB peak RSS, ≈5 s first render - within documented limits.
* **Blocker (EXTERNAL):** runtime artifacts are git-ignored. Without them the deployed app
  shows setup messages. Owner options:
  1. Publish `models/<set>/` (fitted parameters/aggregates, 0.7 MB) and
     `reports/metrics/<set>/` (aggregates, 0.4 MB) via `git add -f` after review. Exclude
     `reports/metrics/data_profile.json` and never publish `data/processed/tda/` (row-level).
     The TDA page then shows its setup message.
  2. Host artifacts as a GitHub Release asset and download them at start-up with a checksum
     (not implemented).
  3. Deploy a synthetic demo.

  No training dataset is required at runtime in any option.
* **Not done:** no deployment was performed; no URL exists.

## L. Portfolio presentation

* **Title:** CreditIQ - Explainable Credit Risk Scoring with Out-of-Time Validation and
  Topological Diagnostics
* **50-word summary:** An educational end-to-end credit-risk project on 411k historical Lending
  Club loans: leakage-audited features, out-of-time validated LightGBM and logistic regression,
  calibrated probabilities, SHAP/LIME/permutation explanations, Mapper-based topological
  diagnostics and an interactive Streamlit dashboard, with transparent reporting of selection
  and outcome-maturity biases.
* **150-word technical description:** CreditIQ builds a reproducible Python pipeline that:
  * validates the Figshare Lending Club 2012-2014 export
  * defines a lifetime adverse-outcome target, excluding unresolved loans
  * audits every column for availability at the decision point
  * trains Dummy, logistic-regression and LightGBM models on 2012-2013, tuning on 2014 Q1,
    calibrating on 2014 Q2 and testing once on 2014 H2

  Test isolation is verified by retraining with scrambled test labels. Evaluation reports
  bootstrap confidence intervals, a 36-month exclusion-free check and assumed-outcome scenarios
  for unresolved loans. Explanations combine exact SHAP, a LIME-style surrogate with fidelity
  and stability checks, and grouped permutation importance. A Mapper graph and persistent
  homology map the applicant space and locate miscalibrated regions. Results, a hypothetical
  decision simulator and documentation are served through an eight-page Streamlit app, backed
  by 128 automated tests and CI.
* **Verified performance (applicant model, out-of-time test):**
  * ROC-AUC 0.698 [0.695, 0.702] on resolved loans (likely optimistic)
  * 0.653 on fully resolved 36-month loans
  * 0.671-0.674 under assumed outcomes for excluded loans
  * dummy baseline 0.500
* **Stack:** Python, pandas, DuckDB, scikit-learn, LightGBM, SHAP, SciPy, NetworkX, ripser,
  Altair, Streamlit, pytest, Ruff, GitHub Actions.
* **Website card:**
  * title, the 50-word summary and three badges (Out-of-time validated · Explainable · TDA)
  * screenshot of the Topology page and the Model Performance page
  * links: GitHub repository, plus the live demo only once it exists
* **Disclaimer to display:** "Educational project on historical public data; not a validated or
  approved credit-decision system. Scores are not regulatory probabilities of default."

## M. Remaining issues

| Severity | Issue | Impact | Evidence | Proposed solution | Owner action? |
| --- | --- | --- | --- | --- | --- |
| **Critical** | Dataset publicly downloadable from GitHub LFS (C1) | Redistribution of the dataset contrary to project policy and possible upstream terms | `git ls-remote` → `8c03b42`; LFS batch API returns a download URL; repo public | Back up, delete and recreate the repo (or contact GitHub Support), then `git push -u origin main`; re-run the LFS batch check. Earlier clones and forks cannot be recalled | **Yes** |
| High | Runtime artifacts not in Git | Deployed app shows only setup messages | `.gitignore` excludes `models/`, `reports/metrics/`, `data/processed/` | Choose an option in section K | **Yes** |
| Medium | Headline AUC likely optimistic; calibrator fitted on biased split | Over-reading 0.698 | Scenario bounds 0.671-0.674; 36-month 0.653 | Already disclosed; future: survival model, calibrate on matured cohorts | No (research work) |
| Medium | Uncommitted audit changes | Fixes lost if not committed | `git status` shows 7 modified/new paths | Review and commit | **Yes** |
| Low | GitHub Actions pinned to tags, not SHAs | Supply-chain risk if a tag is moved | `ci.yml` | Pin to commit SHAs if desired | Optional |
| Low | Bootstrap CIs ignore temporal clustering | CIs too narrow | i.i.d. resampling | Block bootstrap by issue month | No |
| Low | `requirements.txt` includes dev tools | Longer cloud installs | file contents | Optional separate runtime file | Optional |
| Low | Public commit e-mail | Personal data exposure | git config | Use GitHub noreply e-mail before pushing | Optional |
| Low | Mapper node-click, LIME stability toggle, highlight buttons not covered by automated tests | Possible unnoticed UI regression | AppTest cannot drive Vega selections | Manual check after deployment | No |

## N. Final release verdict

**NOT READY - BLOCKING ISSUES REMAIN**

The repository content passes every engineering check that was run:
* tests, lint, real-artifact app checks and a clean-environment deployment simulation
* dependency audit, secrets scan and test-isolation re-verification
* scientific claims are now worded consistently with the evidence

Two items still block a public release:
* the restricted dataset remains publicly downloadable from the existing GitHub repository's
  LFS storage (C1)
* the deployment cannot show predictions until the owner decides how runtime artifacts are
  provided

After the repository is recreated, the LFS check passes, and the artifact decision is made,
the expected verdict is **READY WITH NON-BLOCKING LIMITATIONS**. That depends on owner
actions not yet performed. No deployment was performed.
