"""CreditIQ Streamlit dashboard.

Run from the repository root:
    streamlit run app/streamlit_app.py

Set CREDITIQ_SYNTHETIC=1 to browse the synthetic demo outputs under data/synthetic/.
The app only reads saved artifacts, aggregate tables and local (git-ignored) TDA samples;
it never trains models.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Deployments that install only requirements.txt (e.g. Streamlit Community Cloud) do not
# install the local package; fall back to the in-repo source tree.
try:
    import creditiq  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # server-side rendering only

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from creditiq.cli import SYNTHETIC_PATH_OVERRIDES
from creditiq.config import Config, load_config
from creditiq.explainability.lime_tabular import explain_lime, lime_stability
from creditiq.explainability.shap_analysis import compute_shap, explain_instance
from creditiq.models.artifacts import ModelBundle, read_json
from creditiq.models.predict import ArtifactsNotFoundError, available_models, load_model
from creditiq.reporting import interactive as ix
from creditiq.reporting.results import (
    load_results,
    load_sql_results,
    results_available,
    tda_cloud_path,
)
from creditiq.risk.decision_analysis import DecisionAssumptions, assign_risk_bucket, breakeven_pd
from creditiq.tda.insights import STATIC_IDEAS, generate_ideas
from creditiq.tda.lenses import LENSES, compute_lens
from creditiq.tda.mapper import (
    build_mapper,
    edge_segments,
    knn_eps,
    layout,
    node_feature_profile,
    node_statistics,
)
from creditiq.tda.persistence import (
    diagram_frame,
    persistence_diagrams,
    persistence_summary,
    ripser_available,
)
from creditiq.tda.point_cloud import feature_columns, feature_matrix

DISCLAIMER = (
    "Educational portfolio project. Not validated or approved for real-world lending "
    "decisions. Scores are not regulatory probabilities of default."
)
PAGES = [
    "🏠 Overview",
    "📊 Portfolio Analysis",
    "📈 Model Performance",
    "🔍 Explainability",
    "🎯 Risk Prediction",
    "💼 Decision Simulation",
    "🕸️ Topology (TDA)",
    "📚 Model Documentation",
]
FRIENDLY = {
    "loan_amnt": "Loan amount (USD)",
    "annual_inc": "Annual income (USD, self-reported)",
    "dti": "Debt-to-income ratio (%, excl. this loan)",
    "delinq_2yrs": "30+ day delinquencies, last 2 years",
    "inq_last_6mths": "Credit inquiries, last 6 months",
    "mths_since_last_delinq": "Months since last delinquency",
    "mths_since_last_record": "Months since last public record",
    "mths_since_last_major_derog": "Months since last 90+ day rating",
    "open_acc": "Open credit lines",
    "pub_rec": "Derogatory public records",
    "revol_bal": "Revolving balance (USD)",
    "revol_util": "Revolving utilisation (fraction, 0.5 = 50%)",
    "total_acc": "Total credit lines",
    "collections_12_mths_ex_med": "Collections last 12 months (excl. medical)",
    "int_rate": "Interest rate (fraction)",
    "installment": "Monthly instalment (USD)",
    "term": "Term",
    "emp_length": "Employment length",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "sub_grade": "Lending Club sub-grade",
    "verification_status": "Income verification status",
    "initial_list_status": "Initial listing status",
    "earliest_cr_line": "Earliest credit line (month)",
    "issue_d": "Application / issue month",
}
INPUT_GROUPS = {
    "Loan request": [
        "loan_amnt",
        "term",
        "purpose",
        "int_rate",
        "installment",
        "sub_grade",
        "initial_list_status",
    ],
    "Income & employment": [
        "annual_inc",
        "emp_length",
        "home_ownership",
        "verification_status",
        "dti",
    ],
    "Credit bureau": [
        "revol_bal",
        "revol_util",
        "open_acc",
        "total_acc",
        "inq_last_6mths",
        "delinq_2yrs",
        "pub_rec",
        "collections_12_mths_ex_med",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "earliest_cr_line",
    ],
    "Application date": ["issue_d"],
}
OPTIONAL_HISTORY = {
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
}
BUCKET_COLORS = {"A": "#2ca02c", "B": "#98df8a", "C": "#ffbb78", "D": "#ff7f0e", "E": "#d62728"}

st.set_page_config(page_title="CreditIQ", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {background: rgba(28,131,225,0.06); border-radius: 10px;
        padding: 10px 14px; border: 1px solid rgba(28,131,225,0.15);}
      .ciq-badge {display:inline-block; padding:2px 10px; border-radius:12px; color:white;
        font-weight:600; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- loaders
@st.cache_resource
def get_config() -> Config:
    cfg = load_config()
    if os.environ.get("CREDITIQ_SYNTHETIC") == "1":
        cfg = cfg.with_overrides({"paths": SYNTHETIC_PATH_OVERRIDES})
    return cfg


@st.cache_resource
def get_bundle(models_dir: str, feature_set: str, name: str) -> ModelBundle:
    return load_model(Path(models_dir), feature_set, name)


@st.cache_data
def get_results(models_dir: str, metrics_dir: str, feature_set: str) -> dict:
    return load_results(Path(models_dir), Path(metrics_dir), feature_set)


@st.cache_data
def get_sql(metrics_dir: str) -> dict[str, pd.DataFrame]:
    return load_sql_results(Path(metrics_dir))


@st.cache_data
def get_json(path: str) -> dict | None:
    p = Path(path)
    return read_json(p) if p.is_file() else None


@st.cache_data
def get_cloud(processed_dir: str, feature_set: str, split: str) -> pd.DataFrame | None:
    p = tda_cloud_path(Path(processed_dir), feature_set, split)
    return pd.read_parquet(p) if p.is_file() else None


@st.cache_data(show_spinner="Building Mapper graph ...", max_entries=32)
def run_mapper(
    processed_dir: str,
    feature_set: str,
    split: str,
    lens: str,
    n_intervals: int,
    overlap: float,
    eps_quantile: float,
    min_cluster: int,
    knn_k: int,
    seed: int,
) -> dict:
    cloud = get_cloud(processed_dir, feature_set, split)
    X = feature_matrix(cloud)
    eps = knn_eps(X, k=knn_k, quantile=eps_quantile, seed=seed)
    graph = build_mapper(
        X,
        compute_lens(lens, X, cloud, seed=seed),
        n_intervals=n_intervals,
        overlap=overlap,
        eps=eps,
        min_cluster_size=min_cluster,
        seed=seed,
    )
    nodes = node_statistics(graph, cloud, feature_columns(cloud))
    pos = layout(graph, seed=seed)
    return {
        "graph": graph,
        "nodes": nodes,
        "positions": pos,
        "segments": edge_segments(graph.edges, pos),
        "ideas": generate_ideas(nodes, float(cloud["y"].mean()), graph.n_components),
        "eps": eps,
    }


@st.cache_data(show_spinner="Computing persistent homology ...", max_entries=8)
def run_persistence(
    processed_dir: str, feature_set: str, split: str, n: int, seed: int
) -> pd.DataFrame:
    cloud = get_cloud(processed_dir, feature_set, split)
    X = feature_matrix(cloud)
    rng = np.random.default_rng(seed)
    frames = []
    for label, val in (("adverse", 1), ("repaid", 0)):
        idx = np.flatnonzero(cloud["y"].to_numpy() == val)
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        frames.append(diagram_frame(persistence_diagrams(X[pick]), label))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def run_pca(processed_dir: str, feature_set: str, split: str, seed: int) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    cloud = get_cloud(processed_dir, feature_set, split)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(feature_matrix(cloud))
    out = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "pd": cloud["pd"]})
    out["Outcome"] = np.where(cloud["y"] == 1, "Adverse", "Repaid")
    return out.sample(min(3000, len(out)), random_state=seed)


def read_report(cfg: Config, name: str) -> str | None:
    path = cfg.path("reports_dir") / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def setup_message(what: str = "trained/evaluated model artifacts") -> None:
    st.warning(
        f"No {what} found. Run, from the repository root:\n\n"
        "```\npython scripts/inspect_data.py\npython scripts/prepare_data.py\n"
        "python scripts/train_model.py --feature-set all\n"
        "python scripts/evaluate_model.py --feature-set all\n"
        "python scripts/generate_reports.py\n```\n"
        "Nothing is shown or predicted until real artifacts exist."
    )


def chart(c: alt.TopLevelMixin, **kw):
    kw.setdefault("width", "stretch")
    return st.altair_chart(c, **kw)


def pct(v: float, digits: int = 1) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}%}"


def selected_row(res: dict) -> pd.Series:
    tm = res["test_metrics"]
    sel = res["metadata"]["selected_model"]
    return tm[(tm["model"] == sel) & tm["variant"].str.startswith("calibrated")].iloc[0]


def bounds_row(res: dict, model: str, scenario_prefix: str) -> pd.Series | None:
    b = res.get("unresolved_bounds")
    if b is None or b.empty:
        return None
    m = b[(b["model"] == model) & b["scenario"].str.startswith(scenario_prefix)]
    return m.iloc[0] if len(m) else None


# ----------------------------------------------------------------------------- overview
def page_overview(cfg: Config, res: dict | None) -> None:
    st.title("CreditIQ")
    st.caption("Explainable credit risk scoring & decision intelligence · Lending Club 2012-2014")
    st.info(DISCLAIMER, icon="⚠️")
    if res is None:
        setup_message()
        return
    meta, summ = res["metadata"], res["summary"]
    sel = meta["selected_model"]
    row = selected_row(res)
    dummy = res["test_metrics"].query("model == 'dummy' and variant == 'raw'").iloc[0]
    c = st.columns(5)
    c[0].metric("Selected model", ix.MODEL_LABELS.get(sel, sel))
    c[1].metric(
        "Test ROC-AUC (reported)", f"{row['roc_auc']:.3f}", f"{row['roc_auc'] - 0.5:+.3f} vs dummy"
    )
    c[2].metric(
        "PR-AUC", f"{row['pr_auc']:.3f}", f"{row['pr_auc'] - dummy['pr_auc']:+.3f} vs dummy"
    )
    c[3].metric(
        "Brier",
        f"{row['brier']:.4f}",
        f"{row['brier'] - dummy['brier']:+.4f}",
        delta_color="inverse",
    )
    c[4].metric(
        "Test loans",
        f"{summ['test_n']:,}",
        f"prevalence {summ['test_prevalence']:.1%}",
        delta_color="off",
    )

    realistic = bounds_row(res, sel, "Current/Grace")
    full36 = res["sensitivity_36_month"]
    s36 = full36[(full36["model"] == sel) & full36["population"].str.startswith("36")]
    if realistic is not None:
        st.warning(
            f"**Read the headline AUC with care.** {int(realistic['n_unresolved_added']):,} test-window "
            f"loans were excluded as unresolved (outcome unknown at the snapshot). Under an *assumed* "
            f"labelling (Current/Grace repaid, Late adverse) the test ROC-AUC would be "
            f"**{realistic['roc_auc']:.3f}** instead of {row['roc_auc']:.3f}; this is a scenario, not an "
            f"observed result. On fully resolved 36-month loans it is "
            f"**{s36['roc_auc'].iloc[0]:.3f}**. See *Model Performance → Robustness*.",
            icon="🔎",
        )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What this project does")
        st.markdown(
            """
- **Predicts** the probability of an *adverse outcome* (charged off / default) from information
  available when an application is assessed — leakage-audited, out-of-time validated, calibrated.
- **Explains** predictions with **SHAP**, **LIME** and **permutation importance**, and checks
  whether the three agree.
- **Maps** the applicant space with **topological data analysis** (Mapper graphs, persistent
  homology) to find regions where the model is miscalibrated or weak.
- **Simulates** approval cut-offs under explicit, *hypothetical* economics.
"""
        )
        splits = pd.DataFrame(
            [{"split": k, **v} for k, v in meta["split"].items() if isinstance(v, dict)]
        )
        splits["first_issue"] = pd.to_datetime(splits["first_issue"])
        splits["last_issue"] = pd.to_datetime(splits["last_issue"]) + pd.offsets.MonthEnd(0)
        timeline = (
            alt.Chart(splits)
            .mark_bar(height=22)
            .encode(
                x=alt.X("first_issue:T", title="Issue date"),
                x2="last_issue:T",
                y=alt.Y(
                    "split:N",
                    sort=["train", "validation", "calibration", "test"],
                    title=None,
                    axis=alt.Axis(labelOverlap=False),
                ),
                color=alt.Color("split:N", legend=None),
                tooltip=[
                    "split",
                    alt.Tooltip("n:Q", format=","),
                    alt.Tooltip("adverse_rate:Q", format=".2%"),
                ],
            )
            .properties(height=210, title="Out-of-time design: each split is a later period")
        )
        chart(timeline)
    with right:
        st.subheader("Data funnel")
        prep = get_json(str(cfg.path("metrics_dir") / "data_preparation.json"))
        if prep:
            funnel = pd.DataFrame(
                {
                    "stage": ["Source loans", "After cleaning", "Resolved (modelled)"],
                    "loans": [
                        prep["source_rows"],
                        prep["cleaned_rows_all_statuses"],
                        prep["modelling_rows"],
                    ],
                }
            )
            chart(
                alt.Chart(funnel)
                .mark_bar()
                .encode(
                    x=alt.X("loans:Q", title="Loans"),
                    y=alt.Y("stage:N", sort=None, title=None),
                    tooltip=["stage", alt.Tooltip("loans:Q", format=",")],
                    color=alt.value("#1f77b4"),
                )
                .properties(height=150)
            )
        st.markdown(
            """
**Target:** `Charged Off`/`Default` = 1, `Fully Paid` = 0; `Current`, `Late` and
`In Grace Period` loans are excluded (unresolved), not treated as good.

**Main limitations:** lifetime outcome, not a 12-month PD · only funded loans (selection bias) ·
2014 60-month loans partly unresolved · no bureau scores in this export.
"""
        )


# ----------------------------------------------------------------------------- portfolio
def page_portfolio(cfg: Config) -> None:
    st.header("Portfolio analysis")
    st.caption(
        "Historical, descriptive associations in Lending Club's *approved* portfolio — not causal effects."
    )
    sql = get_sql(str(cfg.path("metrics_dir")))
    if not sql:
        setup_message("SQL analytics outputs")
        return
    q = "02_portfolio_analysis."
    status = sql.get("01_data_quality.status_distribution")
    c1, c2 = st.columns([2, 3])
    with c1:
        if status is not None:
            chart(
                alt.Chart(status)
                .mark_bar()
                .encode(
                    x=alt.X("n_loans:Q", title="Loans", scale=alt.Scale(type="symlog")),
                    y=alt.Y("loan_status:N", sort="-x", title=None),
                    color=alt.Color("loan_status:N", legend=None),
                    tooltip=["loan_status", alt.Tooltip("n_loans:Q", format=","), "pct"],
                )
                .properties(title="Loan status at snapshot (log scale)", height=240)
            )
    with c2:
        tab_a, tab_b = st.tabs(["Adverse rate by vintage", "Unresolved share (excluded)"])
        with tab_a:
            chart(
                ix.vintage_chart(
                    sql[q + "adverse_rate_by_vintage_term"],
                    "adverse_rate",
                    "Adverse rate among resolved loans",
                )
            )
        with tab_b:
            chart(
                ix.vintage_chart(
                    sql[q + "unresolved_share_by_vintage_term"],
                    "unresolved_share",
                    "Share unresolved at snapshot",
                )
            )
            st.caption(
                "60-month 2014 loans are increasingly unresolved — the source of the exclusion bias discussed on the Performance page."
            )

    st.subheader("Segment explorer")
    dims = {
        "Loan purpose": (q + "adverse_rate_by_purpose", "purpose"),
        "Lending Club grade": (q + "adverse_rate_by_grade", "grade"),
        "Income band": (q + "adverse_rate_by_income_band", "income_band"),
        "Home ownership": (q + "adverse_rate_by_home_ownership", "home_ownership"),
        "State (top 15 by volume)": (q + "adverse_rate_by_state_top15", "addr_state"),
    }
    choice = st.selectbox("Segment by", list(dims))
    key, cat = dims[choice]
    df = sql[key]
    sort = "ascending" if cat in ("grade", "income_band") else "-x"
    chart(
        ix.bar_rate_chart(
            df,
            cat,
            "adverse_rate",
            "n_loans",
            f"Adverse rate by {choice.lower()}",
            sort=sort if sort != "ascending" else None,
        )
    )

    st.subheader("Risk gradients of key applicant features (training split)")
    g1, g2, g3 = st.columns(3)
    f = "03_feature_queries."
    for col, (key, x, title) in zip(
        (g1, g2, g3),
        (
            (f + "dti_decile_risk", "dti_decile", "DTI decile"),
            (f + "loan_to_income_quintile_risk", "lti_quintile", "Loan-to-income quintile"),
            (f + "credit_history_band_risk", "credit_history_band", "Credit history band"),
        ),
        strict=True,
    ):
        if key in sql:
            col.altair_chart(
                alt.Chart(sql[key])
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{x}:O", title=title),
                    y=alt.Y("adverse_rate:Q", axis=alt.Axis(format="%"), title="Adverse rate"),
                    tooltip=list(sql[key].columns),
                )
                .properties(height=230)
            )
    with st.expander("Realised loss on charged-off loans (descriptive, from outcome fields)"):
        if q + "realized_loss_charged_off" in sql:
            st.dataframe(sql[q + "realized_loss_charged_off"], hide_index=True)
        st.caption("Outcome fields are used for portfolio reporting only — never as model inputs.")


# ----------------------------------------------------------------------------- performance
def page_performance(res: dict | None) -> None:
    st.header("Model performance — out-of-time test set")
    if res is None:
        setup_message()
        return
    tm = res["test_metrics"]
    cal = tm[tm["variant"].str.startswith("calibrated")].set_index("model")
    tabs = st.tabs(
        ["Summary", "Discrimination", "Calibration", "Threshold explorer", "Robustness", "Segments"]
    )
    with tabs[0]:
        st.dataframe(
            tm[
                [
                    "model",
                    "variant",
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "log_loss",
                    "ece",
                    "prevalence",
                    "mean_predicted",
                ]
            ],
            hide_index=True,
        )
        ci = res["bootstrap_ci"]
        if not ci.empty:
            st.markdown(
                "**Bootstrap 95% CIs (calibrated scores; i.i.d. resampling — understates period-to-period uncertainty)**"
            )
            chart(
                alt.Chart(ix._label_models(ci))
                .mark_errorbar(ticks=True)
                .encode(
                    x=alt.X("lower:Q", title="Value", scale=alt.Scale(zero=False)),
                    x2="upper:Q",
                    y=alt.Y("Model:N", title=None),
                    color="Model:N",
                )
                .properties(height=90)
                .facet(row=alt.Row("metric:N", title=None))
                .resolve_scale(x="independent")
            )
        diff = res["summary"].get("auc_diff_lightgbm_minus_logreg")
        if diff:
            st.write(
                f"Paired bootstrap AUC(LightGBM) − AUC(LogReg) = **{diff['point']:+.4f}** (95% CI [{diff['lower']:+.4f}, {diff['upper']:+.4f}])."
            )
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            chart(ix.roc_chart(res["roc_curves"], cal["roc_auc"].to_dict()))
        with c2:
            chart(
                ix.pr_chart(
                    res["pr_curves"], cal["pr_auc"].to_dict(), res["summary"]["test_prevalence"]
                )
            )
    with tabs[2]:
        chart(ix.reliability_chart(res["calibration_tables"]))
        st.caption(
            "Calibration was fitted on 2014 Q2, which also excludes unresolved 60-month loans; see Robustness."
        )
    with tabs[3]:
        scores = res.get("score_distribution")
        if scores is None or scores.empty:
            st.info("Score distribution not available — re-run evaluate_model.py.")
        else:
            default = float(round(res["summary"]["selected_threshold"] / 0.005) * 0.005)
            t = st.slider(
                "Flag as high risk if calibrated PD ≥", 0.0, 0.8, default, 0.005, format="%.3f"
            )
            m = ix.threshold_metrics_from_bins(scores, t)
            c = st.columns(5)
            c[0].metric("Precision", pct(m["precision"]))
            c[1].metric("Recall", pct(m["recall"]))
            c[2].metric("F1", f"{m['f1']:.3f}")
            c[3].metric("False-positive rate", pct(m["false_positive_rate"]))
            c[4].metric("Flagged share", pct(m["flagged_rate"]))
            left_col, right_col = st.columns([3, 2])
            with left_col:
                chart(ix.score_histogram(scores, t))
            with right_col:
                cm = pd.DataFrame(
                    {
                        "Actual": ["Repaid", "Repaid", "Adverse", "Adverse"],
                        "Predicted": ["Low risk", "High risk", "Low risk", "High risk"],
                        "Loans": [m["tn"], m["fp"], m["fn"], m["tp"]],
                    }
                )
                heat = (
                    alt.Chart(cm)
                    .mark_rect()
                    .encode(
                        x="Predicted:N",
                        y="Actual:N",
                        color=alt.Color("Loans:Q", scale=alt.Scale(scheme="blues"), legend=None),
                    )
                )
                text = (
                    alt.Chart(cm)
                    .mark_text(fontSize=16)
                    .encode(x="Predicted:N", y="Actual:N", text=alt.Text("Loans:Q", format=","))
                )
                chart((heat + text).properties(height=260, title="Confusion matrix"))
            st.caption(
                f"The pipeline's threshold ({res['summary']['selected_threshold']:.3f}) was chosen on the calibration split. Exploring thresholds here on the test set is descriptive only."
            )
    with tabs[4]:
        b = res.get("unresolved_bounds")
        if b is not None and not b.empty:
            st.markdown(
                "**Sensitivity to excluded unresolved loans** — test-window loans excluded as unresolved are scored and added back under explicit labelling scenarios."
            )
            bb = ix._label_models(b)
            chart(
                alt.Chart(bb)
                .mark_bar()
                .encode(
                    x=alt.X("roc_auc:Q", scale=alt.Scale(zero=False), title="Test ROC-AUC"),
                    y=alt.Y("scenario:N", title=None, sort=None),
                    color="Model:N",
                    yOffset="Model:N",
                    tooltip=[
                        "Model",
                        "scenario",
                        alt.Tooltip("roc_auc:Q", format=".4f"),
                        alt.Tooltip("prevalence:Q", format=".2%"),
                        alt.Tooltip("mean_predicted:Q", format=".2%"),
                        "n",
                    ],
                )
                .properties(height=260)
            )
            st.dataframe(
                b[
                    [
                        "model",
                        "scenario",
                        "n",
                        "prevalence",
                        "mean_predicted",
                        "roc_auc",
                        "brier",
                        "prevalence_60m",
                        "mean_predicted_60m",
                    ]
                ],
                hide_index=True,
            )
            st.caption(
                "Excluded loans are survivors (not charged off by the snapshot). The 'all adverse' row is an extreme bound, not a plausible outcome."
            )
        st.markdown("**Fully resolved 36-month loans** (no exclusion bias)")
        st.dataframe(
            res["sensitivity_36_month"][
                [
                    "model",
                    "population",
                    "n",
                    "prevalence",
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "mean_predicted",
                ]
            ],
            hide_index=True,
        )
        q = res["metrics_by_quarter"]
        if not q.empty:
            chart(
                alt.Chart(ix._label_models(q))
                .mark_line(point=True)
                .encode(
                    x="issue_quarter:N",
                    y=alt.Y("roc_auc:Q", scale=alt.Scale(zero=False)),
                    color="Model:N",
                    tooltip=["Model", "issue_quarter", "n", alt.Tooltip("roc_auc:Q", format=".4f")],
                )
                .properties(height=220, title="ROC-AUC by issue quarter (test)")
            )
        if not res["random_split_benchmark"].empty:
            st.markdown("**Random vs out-of-time split (LightGBM)**")
            st.dataframe(res["random_split_benchmark"], hide_index=True)
    with tabs[5]:
        seg = res["segment_performance"]
        name = st.selectbox("Segment", list(dict.fromkeys(seg["segment"])), key="seg_pick")
        chart(ix.segment_chart(seg, name))
        st.dataframe(
            seg[seg["segment"] == name].drop(columns=["segment", "segment_label"]), hide_index=True
        )
        st.caption(
            "Descriptive only; similar metrics across groups do not establish fairness. No protected attributes exist in the data."
        )


# ----------------------------------------------------------------------------- explainability
def page_explainability(res: dict | None) -> None:
    st.header("Explainability — SHAP · LIME · Permutation importance")
    st.caption(
        "All three describe how the **model** uses its inputs on held-out (test) loans. None of them measures causal effects."
    )
    if res is None or "explainer_agreement" not in res or res["explainer_agreement"].empty:
        setup_message("explainability outputs (run scripts/generate_reports.py)")
        return
    models = [m for m in ("lightgbm", "logistic_regression") if f"shap_signed_{m}" in res]
    model = st.radio("Model", models, format_func=lambda m: ix.MODEL_LABELS[m], horizontal=True)
    tabs = st.tabs(
        [
            "SHAP global",
            "SHAP dependence",
            "Permutation importance",
            "LIME",
            "Agreement",
            "How to read",
        ]
    )
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            chart(
                ix.signed_shap_chart(
                    res[f"shap_signed_{model}"], "Model features: mean |SHAP| and direction"
                )
            )
        with c2:
            chart(
                ix.importance_chart(
                    res[f"shap_raw_{model}"],
                    "feature",
                    "mean_abs_shap",
                    "Aggregated to raw inputs",
                    color="#9467bd",
                )
            )
            st.caption(
                "Engineered features built from several inputs (e.g. loan_to_income) are split equally among their sources."
            )
    with tabs[1]:
        dep = res[f"shap_dependence_{model}"]
        feats = list(dict.fromkeys(dep["feature"]))
        feat = st.selectbox("Feature", feats)
        chart(ix.dependence_chart(dep, feat))
        st.caption(
            "Each point is a quantile bin of the feature on test loans; the line is mean SHAP and the band the 10-90% range (interaction/heterogeneity)."
        )
    with tabs[2]:
        metric = st.radio(
            "Metric",
            ["roc_auc", "log_loss", "brier"],
            horizontal=True,
            format_func=lambda m: {
                "roc_auc": "ROC-AUC (ranking)",
                "log_loss": "Log loss (probabilities)",
                "brier": "Brier",
            }[m],
        )
        c1, c2 = st.columns(2)
        with c1:
            chart(
                ix.importance_chart(
                    res[f"pfi_single_{model}"],
                    "feature",
                    f"{metric}_drop_mean",
                    "Single-input PFI",
                    err=f"{metric}_drop_std",
                )
            )
        with c2:
            chart(
                ix.importance_chart(
                    res[f"pfi_grouped_{model}"],
                    "feature",
                    f"{metric}_drop_mean",
                    "Grouped PFI (related inputs shuffled jointly)",
                    err=f"{metric}_drop_std",
                    color="#ff7f0e",
                )
            )
            with st.expander("Group definitions"):
                st.dataframe(res[f"pfi_grouped_{model}"][["feature", "columns"]], hide_index=True)
        st.caption(
            "Single-input PFI understates correlated inputs; grouped PFI is the remedy. Negative values = noise (shuffling did not hurt)."
        )
    with tabs[3]:
        lime = res[f"lime_global_{model}"]
        chart(
            ix.importance_chart(
                lime,
                "feature",
                "mean_abs_lime_weight",
                "Mean |LIME weight| over explained test loans",
                color="#2ca02c",
            )
        )
        r2 = float(lime["mean_local_r2"].iloc[0]) if "mean_local_r2" in lime else float("nan")
        st.metric("Mean local surrogate R² (fidelity)", f"{r2:.2f}")
        st.caption(
            "LIME fits a weighted linear surrogate around each applicant using perturbations drawn from training distributions (aggregates only). Low R² = the local linear story is a rough approximation."
        )
    with tabs[4]:
        agr = res["explainer_agreement"]
        chart(ix.agreement_chart(agr, model))
        g = agr[agr["model"] == model]
        corr = g[["rank_shap", "rank_pfi", "rank_lime"]].corr(method="spearman")
        corr.index = corr.columns = ["SHAP", "PFI", "LIME"]
        st.markdown("**Spearman rank correlation between methods**")
        st.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0, vmax=1))
        st.caption(
            "High agreement suggests robust conclusions; disagreement usually points to correlated inputs or non-linear effects."
        )
    with tabs[5]:
        st.markdown(
            """
| Method | Question it answers | Scale | Caveats |
| --- | --- | --- | --- |
| **SHAP** | How much did each feature push *this* prediction away from the average? | log-odds (before calibration) | Additive and exact for these models, but correlated features can share credit. |
| **Permutation importance** | How much worse does the model get on test data if an input is scrambled? | metric degradation | Model-agnostic and outcome-aware; understates correlated inputs (use grouped). |
| **LIME** | Which simple local rule best mimics the model around this applicant? | change in calibrated probability | Depends on sampling and kernel; check R² and stability across seeds. |
"""
        )


# ----------------------------------------------------------------------------- prediction
def _input_widget(col: str, meta: dict, key: str):
    label = FRIENDLY.get(col, col)
    if meta["kind"] == "categorical":
        options = list(meta["categories"])
        if meta["pct_missing"] > 0:
            options.append("(not provided)")
        idx = options.index(meta["mode"]) if meta["mode"] in options else 0
        value = st.selectbox(label, options, index=idx, key=key)
        return None if value == "(not provided)" else value
    if meta["kind"] == "date":
        lo, hi = date.fromisoformat(meta["min"]), date.fromisoformat(meta["max"])
        default = hi if col == "issue_d" else date.fromisoformat(meta["median"])
        return pd.Timestamp(
            st.date_input(label, value=default, min_value=lo, max_value=hi, key=key)
        )
    if col in OPTIONAL_HISTORY and st.checkbox(
        f"None on file: {label.lower()}", value=True, key=f"{key}_none"
    ):
        return np.nan
    lo, hi = float(meta["min"]), float(meta["max"])
    default = float(min(max(meta["median"], lo), hi))
    if meta["integer_valued"]:
        return float(
            st.number_input(
                label, min_value=int(lo), max_value=int(hi), value=int(default), step=1, key=key
            )
        )
    return float(st.number_input(label, min_value=lo, max_value=hi, value=default, key=key))


def _what_if(bundle: ModelBundle, app_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    meta = bundle.input_schema[feature]
    if meta["kind"] == "categorical":
        values = meta["categories"][:20]
    else:
        q = np.asarray(meta["quantiles"], dtype=float)
        values = np.unique(
            np.linspace(q[1], q[-2], 30).round(0 if meta.get("integer_valued") else 4)
        )
    rows = pd.concat([app_df] * len(values), ignore_index=True)
    rows[feature] = list(values)
    return pd.DataFrame({"value": values, "pd": bundle.predict_proba(rows)})


def page_prediction(cfg: Config, feature_set: str, res: dict | None) -> None:
    st.header("Risk prediction (illustrative)")
    st.info(
        DISCLAIMER
        + " Application dates should fall inside the training period; other periods are extrapolations.",
        icon="⚠️",
    )
    models_dir = cfg.path("models_dir")
    names = [n for n in available_models(models_dir, feature_set) if n != "dummy"]
    if res is None or not names:
        setup_message()
        return
    sel = res["metadata"]["selected_model"]
    name = st.radio(
        "Model",
        names,
        index=names.index(sel) if sel in names else 0,
        format_func=lambda m: ix.MODEL_LABELS.get(m, m),
        horizontal=True,
    )
    bundle = get_bundle(str(models_dir), feature_set, name)
    schema = bundle.input_schema
    grouped = {g: [c for c in cols if c in schema] for g, cols in INPUT_GROUPS.items()}
    other = [c for c in schema if not any(c in cols for cols in grouped.values())]
    if other:
        grouped["Other"] = other
    with st.form("application"):
        values = {}
        for group, cols in grouped.items():
            if not cols:
                continue
            st.markdown(f"**{group}**")
            grid = st.columns(4)
            for i, col in enumerate(cols):
                with grid[i % 4]:
                    values[col] = _input_widget(col, schema[col], key=f"in_{col}")
        submitted = st.form_submit_button("Score application", type="primary")
    if submitted:
        st.session_state["scored_application"] = pd.DataFrame([values])[list(schema)]
        st.session_state["scored_application_model"] = (feature_set, name)
    app_df = st.session_state.get("scored_application")
    if app_df is None or st.session_state.get("scored_application_model") != (feature_set, name):
        st.caption(
            "Fill in the form and press **Score application**. Defaults are training medians/modes."
        )
        return

    p_raw = float(bundle.predict_raw(app_df)[0])
    p_cal = float(bundle.calibrator.transform(np.array([p_raw]))[0])
    bucket = assign_risk_bucket(np.array([p_cal]), cfg["decision"]["risk_buckets"])[0]
    color = BUCKET_COLORS.get(str(bucket)[0], "#7f7f7f")
    c = st.columns([1, 1, 1, 1])
    c[0].metric("Calibrated adverse-outcome probability", f"{p_cal:.1%}")
    c[1].metric("Raw model score", f"{p_raw:.1%}")
    c[2].markdown(
        f"Risk bucket<br><span class='ciq-badge' style='background:{color}'>{bucket}</span>",
        unsafe_allow_html=True,
    )
    c[3].metric(
        "Above classification threshold?",
        "yes" if p_cal >= bundle.threshold else "no",
        f"threshold {bundle.threshold:.3f}",
        delta_color="off",
    )
    gauge = pd.DataFrame({"x": [p_cal]})
    bands = pd.DataFrame(
        [
            {
                "lo": 0 if i == 0 else cfg["decision"]["risk_buckets"][i - 1]["upper"],
                "hi": min(b["upper"], 1.0),
                "bucket": b["name"],
            }
            for i, b in enumerate(cfg["decision"]["risk_buckets"])
        ]
    )
    chart(
        (
            alt.Chart(bands)
            .mark_bar(height=26)
            .encode(
                x=alt.X(
                    "lo:Q",
                    scale=alt.Scale(domain=[0, 0.6]),
                    axis=alt.Axis(format="%"),
                    title="Calibrated PD (scale capped at 60%)",
                ),
                x2="hi:Q",
                color=alt.Color(
                    "bucket:N",
                    scale=alt.Scale(range=list(BUCKET_COLORS.values())),
                    legend=alt.Legend(orient="bottom"),
                ),
            )
            + alt.Chart(gauge).mark_tick(color="black", thickness=4, size=40).encode(x="x:Q")
        ).properties(height=70)
    )

    tabs = st.tabs(
        ["SHAP (exact attribution)", "LIME (local surrogate)", "What-if analysis", "Inputs"]
    )
    with tabs[0]:
        local = explain_instance(compute_shap(bundle, app_df), 0, top_k=12)
        chart(
            ix.local_bar_chart(
                local,
                "feature",
                "shap_log_odds",
                "SHAP contributions (log-odds, before calibration)",
                "SHAP value",
            )
        )
        st.caption(
            "Contributions add up exactly to the model's raw log-odds. They describe the model, not causes; correlated inputs can share credit."
        )
    with tabs[1]:
        n = cfg["explainability"]["lime_samples"]
        lime = explain_lime(bundle, app_df, n_samples=n, seed=cfg.seed)
        lime_col, fid_col = st.columns([3, 1])
        with lime_col:
            chart(
                ix.local_bar_chart(
                    lime.explanation.head(12),
                    "condition",
                    "weight",
                    "LIME weights (change in calibrated probability)",
                    "LIME weight",
                )
            )
        with fid_col:
            st.metric("Local fidelity R²", f"{lime.r2:.2f}")
            st.metric("Model prediction", f"{lime.model_prediction:.1%}")
            st.metric("Surrogate at applicant", f"{lime.surrogate_prediction:.1%}")
        if st.toggle("Check LIME stability across 5 random seeds"):
            stab = lime_stability(
                bundle,
                app_df,
                seeds=tuple(cfg["explainability"]["lime_stability_seeds"]),
                n_samples=n,
            )
            st.dataframe(
                stab.head(12).style.format(
                    {"mean_weight": "{:+.4f}", "std_weight": "{:.4f}", "sign_agreement": "{:.0%}"}
                ),
                hide_index=True,
            )
            st.caption(
                "Low sign agreement or large std relative to the mean = treat that LIME term as unreliable."
            )
    with tabs[2]:
        options = [c for c, m in schema.items() if not m.get("reference_only")]
        feat = st.selectbox(
            "Vary one input (others fixed)", options, format_func=lambda c: FRIENDLY.get(c, c)
        )
        curve = _what_if(bundle, app_df, feat)
        current = app_df[feat].iloc[0]
        chart(
            ix.what_if_chart(
                curve,
                FRIENDLY.get(feat, feat),
                current
                if schema[feat]["kind"] == "categorical"
                else float(current)
                if pd.notna(current)
                else float("nan"),
                schema[feat]["kind"],
            )
        )
        st.caption(
            "Individual conditional expectation (ICE) for this applicant — shows the model's response, not what would causally happen."
        )
    with tabs[3]:
        st.dataframe(app_df.T.rename(columns={0: "value"}).astype(str), width="content")


# ----------------------------------------------------------------------------- decision
def page_decision(cfg: Config, res: dict | None) -> None:
    st.header("Hypothetical decision simulation")
    st.warning(
        "All economics are **user-chosen hypothetical assumptions**, not estimated LGD/EAD or margins. "
        "The label is Lending Club's historical charge-off outcome, not a regulatory default. Not an approval system.",
        icon="⚠️",
    )
    if res is None or res["decision_curve"].empty:
        setup_message()
        return
    d = cfg["decision"]
    c = st.columns(4)
    lgd = c[0].slider("LGD (loss given default)", 0.0, 1.0, float(d["lgd"]), 0.05)
    ead = c[1].slider("EAD (fraction of loan amount)", 0.1, 1.0, float(d["ead_fraction"]), 0.05)
    margin = c[2].slider(
        "Lifetime margin on repaid loan", 0.0, 0.4, float(d["profit_rate_good_loan"]), 0.01
    )
    threshold = c[3].slider("Approve if calibrated PD <", 0.02, 0.98, 0.20, 0.005)
    a = DecisionAssumptions(lgd=lgd, ead_fraction=ead, profit_rate_good_loan=margin)
    curve = res["decision_curve"].copy()
    curve["net_value_musd"] = (
        margin * curve["volume_approved_good"] - lgd * ead * curve["volume_approved_bad"]
    ) / 1e6
    curve["approval_rate"] = curve["n_approved"] / curve["n_total"]
    curve["bad_rate_approved"] = curve["n_approved_bad"] / curve["n_approved"].replace(0, np.nan)
    row = curve.iloc[int(np.argmin(np.abs(curve["threshold"] - threshold)))]
    best = curve.iloc[int(curve["net_value_musd"].idxmax())]
    m = st.columns(5)
    m[0].metric("Approval rate", pct(row["approval_rate"]))
    m[1].metric("Bad rate (approved)", pct(row["bad_rate_approved"]))
    m[2].metric(
        "Expected loss on approved",
        f"${lgd * ead * row['pd_weighted_volume_approved'] / 1e6:,.1f}m",
    )
    m[3].metric(
        "Net value",
        f"${row['net_value_musd']:,.1f}m",
        f"best ${best['net_value_musd']:,.1f}m at {best['threshold']:.3f}",
    )
    m[4].metric("Break-even PD", f"{breakeven_pd(a):.3f}")
    base = alt.Chart(curve).encode(
        x=alt.X("threshold:Q", title="Approve if calibrated PD < threshold")
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(x="t:Q")
    )
    be = (
        alt.Chart(pd.DataFrame({"t": [breakeven_pd(a)]})).mark_rule(color="#2ca02c").encode(x="t:Q")
    )
    top = (
        base.mark_line(color="#2ca02c").encode(
            y=alt.Y("net_value_musd:Q", title="Net value (USD m)"),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("net_value_musd:Q", format=",.1f"),
            ],
        )
        + rule
        + be
    ).properties(height=260)
    bottom = (
        base.mark_area(opacity=0.4).encode(
            y=alt.Y("approval_rate:Q", title="Approval rate", axis=alt.Axis(format="%"))
        )
        + base.mark_line(color="#d62728").encode(
            y=alt.Y("bad_rate_approved:Q", axis=alt.Axis(format="%"))
        )
        + rule
    ).properties(height=180)
    chart(alt.vconcat(top, bottom).resolve_scale(x="shared"))
    st.caption(
        f"Green line = break-even PD for these assumptions; black dashed = chosen cut-off. Out-of-time test portfolio, n = {int(row['n_total']):,}."
    )
    t1, t2 = st.tabs(["Risk buckets", "Sensitivity of best cut-off"])
    with t1:
        rb = res["risk_buckets"]
        chart(
            alt.Chart(
                rb.melt(
                    id_vars=["bucket"], value_vars=["mean_predicted_pd", "observed_adverse_rate"]
                )
            )
            .mark_bar()
            .encode(
                x=alt.X("bucket:N", title=None),
                xOffset="variable:N",
                y=alt.Y("value:Q", axis=alt.Axis(format="%"), title="Rate"),
                color="variable:N",
                tooltip=["bucket", "variable", alt.Tooltip("value:Q", format=".2%")],
            )
            .properties(height=260, title="Predicted vs observed by risk bucket")
        )
        st.dataframe(rb, hide_index=True)
    with t2:
        st.dataframe(res["decision_sensitivity"], hide_index=True)


# ----------------------------------------------------------------------------- topology
def page_topology(cfg: Config, feature_set: str, res: dict | None) -> None:
    st.header("Topology of the applicant space (TDA)")
    st.caption(
        "Topological data analysis summarises the *shape* of high-dimensional data. A **Mapper graph** "
        "compresses thousands of loans into a network of overlapping clusters; **persistent homology** "
        "measures which shapes are robust across scales. Here both are used as a diagnostic map of where "
        "the model is well or poorly calibrated."
    )
    tcfg = cfg["tda"]
    pdir = str(cfg.path("processed_dir"))
    splits = [s for s in tcfg["splits"] if get_cloud(pdir, feature_set, s) is not None]
    if res is None or not splits:
        setup_message("TDA point clouds (run scripts/generate_reports.py)")
        return
    mcfg = tcfg["mapper"]
    defaults = {
        "tda_n_int": int(mcfg["n_intervals"]),
        "tda_overlap": float(mcfg["overlap"]),
        "tda_eps_q": float(mcfg["eps_quantile"]),
        "tda_min_c": int(mcfg["min_cluster_size"]),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    with st.container(border=True):
        c = st.columns([1, 1.4, 1.4])
        split = c[0].selectbox(
            "Split",
            splits,
            index=splits.index(tcfg["default_split"]) if tcfg["default_split"] in splits else 0,
            help="Validation is the default so diagnostics never touch the test set.",
        )
        lens = c[1].selectbox(
            "Lens (filter)", list(LENSES), help="The function used to build the overlapping cover."
        )
        color = c[2].selectbox(
            "Colour nodes by",
            list(ix.TDA_COLOR_FIELDS),
            format_func=lambda k: ix.TDA_COLOR_FIELDS[k][0],
            index=2,
        )
        s = st.columns([1, 1, 1, 1, 0.6])
        n_int = s[0].slider(
            "Intervals",
            4,
            24,
            key="tda_n_int",
            help="Resolution of the cover (per lens dimension).",
        )
        overlap = s[1].slider("Overlap", 0.1, 0.7, step=0.05, key="tda_overlap")
        eps_q = s[2].slider(
            "Cluster scale",
            0.2,
            0.9,
            step=0.05,
            key="tda_eps_q",
            help="Quantile of k-NN distances used as single-linkage cut.",
        )
        min_c = s[3].slider("Min node size", 3, 40, key="tda_min_c")
        if s[4].button("Reset", help="Restore the configured defaults"):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()
        st.caption(LENSES[lens])
    if "PD" in lens and split == "test":
        st.info(
            "Using the test split for diagnostics is descriptive; don't tune models on what you see here.",
            icon="ℹ️",
        )
    out = run_mapper(
        pdir, feature_set, split, lens, n_int, overlap, eps_q, min_c, int(mcfg["knn_k"]), cfg.seed
    )
    graph, nodes = out["graph"], out["nodes"]
    cloud = get_cloud(pdir, feature_set, split)
    k = st.columns(5)
    k[0].metric("Loans in sample", f"{len(cloud):,}")
    k[1].metric("Nodes", f"{len(nodes):,}")
    k[2].metric("Edges", f"{len(graph.edges):,}")
    k[3].metric("Connected components", f"{graph.n_components}")
    k[4].metric(
        "Coverage",
        pct(graph.coverage),
        help="Share of loans in at least one node (small clusters are dropped as noise).",
    )

    tabs = st.tabs(
        [
            "🗺️ Topological map",
            "☁️ Point cloud",
            "🧬 Persistent homology",
            "💡 Improvement ideas",
            "🔁 Stability",
            "📖 How it works",
        ]
    )
    with tabs[0]:
        highlight = st.session_state.get("tda_highlight", [])
        left, right = st.columns([3, 1.3])
        with left:
            event = st.altair_chart(
                ix.mapper_chart(nodes, out["positions"], out["segments"], color, highlight),
                on_select="rerun",
                selection_mode=["node"],
                width="stretch",
                key=f"mapper_{split}_{lens}_{n_int}_{overlap}_{eps_q}_{min_c}",
            )
            st.caption(
                "Scroll to zoom · drag to pan · hover for statistics · click a node to inspect it. Node size = loans; edge width = shared loans."
            )
        picked = None
        try:
            sel = event.selection.get("node", []) if event else []
            picked = int(sel[0]["node_id"]) if sel else None
        except (AttributeError, KeyError, TypeError, IndexError):
            picked = None
        with right:
            options = nodes.sort_values("size", ascending=False)["node_id"].tolist()
            if picked is None:
                picked = st.selectbox(
                    "…or choose a node",
                    options,
                    format_func=lambda n: (
                        f"node {n} ({int(nodes.set_index('node_id').loc[n, 'size'])} loans)"
                    ),
                )
            r = nodes.set_index("node_id").loc[picked]
            st.subheader(f"Node {picked}")
            auc = "-" if pd.isna(r["within_node_auc"]) else f"{r['within_node_auc']:.3f}"
            gap_color = "#d62728" if r["calibration_gap"] > 0 else "#1f77b4"
            st.markdown(
                f"""
| | |
| --- | --- |
| Loans | **{int(r["size"]):,}** |
| Observed adverse rate | **{pct(r["observed_rate"])}** |
| Mean predicted PD | **{pct(r["mean_pd"])}** |
| Calibration gap | <span style="color:{gap_color}"><b>{r["calibration_gap"]:+.1%}</b></span> (z = {r["gap_z"]:+.1f}) |
| Risk lift vs average | **{r["risk_lift"]:.2f}×** |
| Within-node AUC | **{auc}** |
""",
                unsafe_allow_html=True,
            )
            chart(
                ix.profile_chart(
                    node_feature_profile(graph, cloud, feature_columns(cloud), picked), top_n=8
                )
            )
            members = cloud.iloc[graph.members[picked]]
            if "purpose" in members:
                mix = (
                    members["purpose"]
                    .value_counts(normalize=True)
                    .head(5)
                    .rename("share")
                    .reset_index()
                )
                chart(
                    alt.Chart(mix)
                    .mark_bar()
                    .encode(
                        x=alt.X("share:Q", axis=alt.Axis(format="%")),
                        y=alt.Y("purpose:N", sort="-x", title=None),
                    )
                    .properties(height=140, title="Purpose mix")
                )
        with st.expander("Node table"):
            st.dataframe(
                nodes.drop(columns=["cube"]).sort_values("size", ascending=False), hide_index=True
            )
    with tabs[1]:
        colour = st.radio(
            "Colour points by",
            ["pd", "Outcome"],
            horizontal=True,
            format_func=lambda v: "Predicted PD" if v == "pd" else "Observed outcome",
        )
        chart(ix.point_cloud_chart(run_pca(pdir, feature_set, split, cfg.seed), colour))
        st.caption(
            "A linear 2-D projection for intuition; Mapper works in the full standardised feature space and can reveal structure this view hides."
        )
    with tabs[2]:
        n_pts = st.slider("Points per outcome group", 150, 800, int(tcfg["persistence_sample"]), 50)
        dg = run_persistence(pdir, feature_set, split, n_pts, cfg.seed)
        dims = sorted(dg["dimension"].unique())
        cols = st.columns(len(dims))
        for col, d in zip(cols, dims, strict=True):
            with col:
                chart(ix.persistence_chart(dg, d))
                chart(ix.barcode_chart(dg, d))
        st.dataframe(persistence_summary(dg), hide_index=True)
        if not ripser_available():
            st.info(
                "Install the optional `tda` extra (`pip install -e .[tda]`) for H1 (loops) via ripser."
            )
        st.caption(
            "**H0** tracks connected clusters as the distance scale grows (deaths = merge scales); **H1** tracks loops. "
            "Points far from the diagonal are robust structure. Comparing adverse vs repaid loans shows whether the two "
            "outcome groups occupy feature space differently (e.g. more fragmented or more spread out)."
        )
    with tabs[3]:
        st.markdown("#### Evidence-based ideas from the current map")
        ideas = out["ideas"]
        if not ideas:
            st.success(
                "No node passes the evidence thresholds (n ≥ 60 and |z| ≥ 2.5 for calibration; lift ≥ 1.6 for risk). Try another lens or resolution."
            )
        for i, idea in enumerate(ideas):
            with st.container(border=True):
                st.markdown(f"**{idea['kind']} · {idea['title']}**")
                st.markdown(f"*Evidence:* {idea['evidence']}")
                st.markdown(f"*Suggested next step:* {idea['action']}")
                if idea["nodes"] and st.button("Highlight on map", key=f"hl_{i}"):
                    st.session_state["tda_highlight"] = idea["nodes"]
                    st.rerun()
        if st.session_state.get("tda_highlight") and st.button("Clear highlights"):
            st.session_state["tda_highlight"] = []
            st.rerun()
        st.markdown("#### Methodological improvement ideas")
        for title, text in STATIC_IDEAS:
            st.markdown(f"- **{title}.** {text}")
        st.caption(
            "Ideas are hypotheses generated from one sample and one parameter setting. Validate any change on a later out-of-time period before adopting it."
        )
    with tabs[4]:
        st.markdown(
            "Does the picture survive changes of resolution? Regions that persist across settings are more trustworthy."
        )
        rows = []
        for n_i in (max(4, n_int - 4), n_int, min(24, n_int + 4)):
            o = run_mapper(
                pdir,
                feature_set,
                split,
                lens,
                n_i,
                overlap,
                eps_q,
                min_c,
                int(mcfg["knn_k"]),
                cfg.seed,
            )
            kinds = pd.Series([i["kind"] for i in o["ideas"]]).value_counts().to_dict()
            rows.append(
                {
                    "intervals": n_i,
                    "nodes": len(o["nodes"]),
                    "edges": len(o["graph"].edges),
                    "components": o["graph"].n_components,
                    "coverage": o["graph"].coverage,
                    **{f"ideas: {k}": v for k, v in kinds.items()},
                }
            )
        st.dataframe(pd.DataFrame(rows).fillna(0), hide_index=True)
    with tabs[5]:
        st.markdown(
            f"""
1. **Point cloud.** {len(cloud):,} loans from the *{split}* split, each described by the
   engineered numeric features of the applicant model, median-imputed and robustly
   standardised with **training-split** statistics.
2. **Lens.** `{lens}` maps every loan to a number (or a pair).
3. **Cover.** The lens range is split into {n_int} overlapping intervals ({overlap:.0%} overlap),
   using rank-balanced intervals so each holds a similar number of loans.
4. **Clustering.** Inside each interval, loans are clustered by single linkage in the full feature
   space, cut at distance {out["eps"]:.2f} (the {eps_q:.0%} quantile of 15-NN distances).
   Clusters smaller than {min_c} loans are treated as noise.
5. **Nerve.** Each cluster is a node; clusters sharing loans are linked.

*References:* Singh, Mémoli & Carlsson (2007), *Topological methods for the analysis of high
dimensional data sets*; Carlsson (2009), *Topology and data*; Edelsbrunner & Harer (2010),
*Computational Topology*.
"""
        )


# ----------------------------------------------------------------------------- documentation
def page_documentation(cfg: Config) -> None:
    st.header("Model documentation")
    tabs = st.tabs(
        ["Model card", "Leakage audit", "Evaluation summary", "Data dictionary", "Methodology"]
    )
    for tab, name in zip(
        tabs[:4],
        ("model_card.md", "leakage_audit.md", "evaluation_summary.md", "data_dictionary.md"),
        strict=True,
    ):
        with tab:
            text = read_report(cfg, name)
            if text:
                st.markdown(text)
            else:
                st.info(f"{name} not generated yet.")
    with tabs[4]:
        st.markdown(
            """
* **Prediction point:** underwriting of a new application; the primary (`applicant`) model uses
  applicant and bureau inputs only.
* **Splits (by issue month):** train 2012-2013; validation 2014 Q1 (tuning, selection, early
  stopping); calibration 2014 Q2 (calibrator, calibration-method CV, threshold); test 2014 H2.
* **Selection:** validation ROC-AUC, with near-ties (≤ 0.0005) broken by validation log loss.
* **Calibration:** none / Platt / isotonic compared by 5-fold CV inside the calibration split.
* **Explainability:** TreeSHAP / exact linear SHAP, single + grouped permutation importance
  (ROC-AUC, log loss, Brier) and LIME local surrogates with fidelity and stability checks.
* **Topology:** Mapper graphs and persistent homology on a standardised applicant feature space.
* **Leakage controls:** availability audit of every column, config + hard-coded deny-list,
  schema-enforced inference, automated tests (including a scrambled-test-label retraining check).
"""
        )


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg = get_config()
    models_dir, metrics_dir = cfg.path("models_dir"), cfg.path("metrics_dir")
    sets = [fs for fs in cfg["features"]["sets"] if results_available(models_dir, metrics_dir, fs)]
    with st.sidebar:
        st.title("📊 CreditIQ")
        if os.environ.get("CREDITIQ_SYNTHETIC") == "1":
            st.error("SYNTHETIC DEMO DATA — results are meaningless.")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        default = cfg["features"]["default_set"]
        feature_set = (
            st.selectbox("Feature set", sets, index=sets.index(default) if default in sets else 0)
            if sets
            else default
        )
        if feature_set == "origination":
            st.caption(
                "⚠️ Origination model uses Lending Club grade/rate — not a pre-approval model."
            )
        st.caption(DISCLAIMER)
    res = (
        get_results(str(models_dir), str(metrics_dir), feature_set) if feature_set in sets else None
    )
    try:
        if page == PAGES[0]:
            page_overview(cfg, res)
        elif page == PAGES[1]:
            page_portfolio(cfg)
        elif page == PAGES[2]:
            page_performance(res)
        elif page == PAGES[3]:
            page_explainability(res)
        elif page == PAGES[4]:
            page_prediction(cfg, feature_set, res)
        elif page == PAGES[5]:
            page_decision(cfg, res)
        elif page == PAGES[6]:
            page_topology(cfg, feature_set, res)
        else:
            page_documentation(cfg)
    except ArtifactsNotFoundError:
        setup_message()


main()
