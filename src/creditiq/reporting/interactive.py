"""Interactive Altair chart builders for the Streamlit dashboard (aggregate inputs only)."""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "dummy": "Dummy (prior)",
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
}
YAXIS = alt.Axis(labelOverlap=False, labelLimit=220)  # show every category label
MODEL_COLORS = alt.Scale(
    domain=list(MODEL_LABELS.values()), range=["#9e9e9e", "#1f77b4", "#d62728"]
)


def _label_models(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Model=df["model"].map(MODEL_LABELS).fillna(df["model"]))


# ----------------------------------------------------------------------------- performance
def roc_chart(roc: pd.DataFrame, aucs: dict[str, float]) -> alt.Chart:
    data = _label_models(roc)
    data["Model"] = data.apply(
        lambda r: f"{r['Model']} (AUC {aucs.get(r['model'], np.nan):.3f})", axis=1
    )
    sel = alt.selection_point(fields=["Model"], bind="legend")
    lines = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate (recall)"),
            color=alt.Color("Model:N", legend=alt.Legend(orient="bottom-right")),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.15)),
            tooltip=[
                "Model",
                alt.Tooltip("fpr:Q", format=".3f"),
                alt.Tooltip("tpr:Q", format=".3f"),
            ],
        )
        .add_params(sel)
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="black")
        .encode(x="x", y="y")
    )
    return (lines + diag).properties(title="ROC curves (click legend to highlight)", height=380)


def pr_chart(pr: pd.DataFrame, aps: dict[str, float], prevalence: float) -> alt.Chart:
    data = _label_models(pr)
    data["Model"] = data.apply(
        lambda r: f"{r['Model']} (AP {aps.get(r['model'], np.nan):.3f})", axis=1
    )
    sel = alt.selection_point(fields=["Model"], bind="legend")
    lines = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            color=alt.Color("Model:N", legend=alt.Legend(orient="top-right")),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.15)),
            tooltip=[
                "Model",
                alt.Tooltip("recall:Q", format=".3f"),
                alt.Tooltip("precision:Q", format=".3f"),
            ],
        )
        .add_params(sel)
    )
    base = alt.Chart(pd.DataFrame({"y": [prevalence]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    return (lines + base).properties(
        title=f"Precision-recall (dashed = prevalence {prevalence:.1%})", height=380
    )


def reliability_chart(cal: pd.DataFrame) -> alt.HConcatChart:
    """One reliability panel per model (raw vs calibrated deciles, diagonal = perfect)."""
    data = _label_models(cal)
    lim = float(max(data["mean_predicted"].max(), data["observed_rate"].max()) * 1.05)
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]}))
        .mark_line(strokeDash=[4, 4], color="black")
        .encode(x="x", y="y")
    )
    panels = []
    for model in dict.fromkeys(data["Model"]):
        pts = (
            alt.Chart(data[data["Model"] == model])
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "mean_predicted:Q",
                    title="Mean predicted probability (decile)",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, lim]),
                ),
                y=alt.Y(
                    "observed_rate:Q",
                    title="Observed adverse rate",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, lim]),
                ),
                color=alt.Color("variant:N", title="Scores"),
                strokeDash=alt.StrokeDash("variant:N", legend=None),
                tooltip=[
                    "Model",
                    "variant",
                    "n",
                    alt.Tooltip("mean_predicted:Q", format=".3f"),
                    alt.Tooltip("observed_rate:Q", format=".3f"),
                ],
            )
        )
        panels.append((pts + diag).properties(width=330, height=330, title=model))
    return alt.hconcat(*panels)


def threshold_metrics_from_bins(scores: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Confusion counts for 'flag if PD >= threshold' using score-bin aggregates."""
    flag = scores["bin_lower"] >= threshold - 1e-12
    tp = int(scores.loc[flag, "n_adverse"].sum())
    fp = int(scores.loc[flag, "n_repaid"].sum())
    fn = int(scores.loc[~flag, "n_adverse"].sum())
    tn = int(scores.loc[~flag, "n_repaid"].sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "flagged_rate": (tp + fp) / max(1, tp + fp + fn + tn),
    }


def score_histogram(scores: pd.DataFrame, threshold: float) -> alt.Chart:
    data = scores[scores["n"] > 0].melt(
        id_vars=["bin_lower"],
        value_vars=["n_repaid", "n_adverse"],
        var_name="Outcome",
        value_name="Loans",
    )
    data["Outcome"] = data["Outcome"].map({"n_repaid": "Repaid", "n_adverse": "Adverse"})
    bars = (
        alt.Chart(data)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X(
                "bin_lower:Q",
                title="Calibrated PD",
                bin=alt.Bin(step=0.01),
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("sum(Loans):Q", title="Loans", stack=None),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=["Repaid", "Adverse"], range=["#2ca02c", "#d62728"]),
            ),
            tooltip=["Outcome", alt.Tooltip("sum(Loans):Q", title="Loans")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="black", strokeWidth=2)
        .encode(x="t:Q")
    )
    return (bars + rule).properties(
        height=300, title="Score distribution by outcome (black line = threshold)"
    )


def bar_rate_chart(
    df: pd.DataFrame, category: str, rate: str, n: str, title: str, sort: str = "-x"
) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{rate}:Q", title="Observed adverse rate", axis=alt.Axis(format="%")),
            y=alt.Y(f"{category}:N", sort=sort, title=None, axis=YAXIS),
            color=alt.Color(f"{rate}:Q", scale=alt.Scale(scheme="orangered"), legend=None),
            tooltip=[
                category,
                alt.Tooltip(f"{rate}:Q", format=".2%"),
                alt.Tooltip(f"{n}:Q", format=","),
            ],
        )
        .properties(title=title, height=max(160, 24 * len(df)))
    )


def vintage_chart(vintage: pd.DataFrame, value: str, title: str) -> alt.Chart:
    data = vintage.assign(issue_quarter=pd.to_datetime(vintage["issue_quarter"]))
    hover = alt.selection_point(
        fields=["issue_quarter"], nearest=True, on="pointerover", empty=False
    )
    base = alt.Chart(data).encode(
        x=alt.X("issue_quarter:T", title="Issue quarter"),
        y=alt.Y(f"{value}:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("term:N", title="Term"),
    )
    lines = base.mark_line(point=True)
    points = (
        base.mark_point(size=80, filled=True)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("issue_quarter:T", format="%Y-Q%q"),
                "term",
                alt.Tooltip(f"{value}:Q", format=".2%"),
                "n_loans",
            ],
        )
        .add_params(hover)
    )
    return (lines + points).properties(title=title, height=300)


def segment_chart(seg: pd.DataFrame, segment: str) -> alt.Chart:
    g = seg[seg["segment"] == segment].copy()
    g["order"] = range(len(g))
    long = g.melt(
        id_vars=["group", "order", "n"],
        value_vars=["observed_adverse_rate", "mean_predicted"],
        var_name="Measure",
        value_name="Rate",
    )
    long["Measure"] = long["Measure"].map(
        {"observed_adverse_rate": "Observed", "mean_predicted": "Mean predicted"}
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("group:N", sort=alt.SortField("order"), title=None),
            xOffset="Measure:N",
            y=alt.Y("Rate:Q", axis=alt.Axis(format="%")),
            color=alt.Color("Measure:N", scale=alt.Scale(range=["#7f7f7f", "#d62728"])),
            tooltip=[
                "group",
                "Measure",
                alt.Tooltip("Rate:Q", format=".2%"),
                alt.Tooltip("n:Q", format=","),
            ],
        )
        .properties(height=300)
    )


# ----------------------------------------------------------------------------- explainability
def importance_chart(
    df: pd.DataFrame,
    feature: str,
    value: str,
    title: str,
    err: str | None = None,
    top_n: int = 20,
    color: str = "#1f77b4",
) -> alt.Chart:
    data = df.nlargest(top_n, value).copy()
    order = data[feature].tolist()  # explicit order shared by all layers
    enc = {
        "x": alt.X(f"{value}:Q", title=value.replace("_", " ")),
        "y": alt.Y(f"{feature}:N", sort=order, title=None, axis=YAXIS),
        "tooltip": [feature, alt.Tooltip(f"{value}:Q", format=".4f")]
        + ([alt.Tooltip(f"{err}:Q", format=".4f")] if err else []),
    }
    bars = alt.Chart(data).mark_bar(color=color).encode(**enc)
    layers = bars
    if err:
        data["lo"] = data[value] - data[err]
        data["hi"] = data[value] + data[err]
        layers = bars + alt.Chart(data).mark_errorbar().encode(
            x="lo:Q", x2="hi:Q", y=alt.Y(f"{feature}:N", sort=order)
        )
    return layers.properties(title=title, height=max(180, 22 * len(data)))


def signed_shap_chart(df: pd.DataFrame, title: str, top_n: int = 20) -> alt.Chart:
    data = df.nlargest(top_n, "mean_abs_shap").copy()
    data["direction"] = np.where(
        data["value_shap_spearman"] >= 0,
        "higher value -> higher risk",
        "higher value -> lower risk",
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=YAXIS),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(range=["#d62728", "#1f77b4"]),
                title="Typical direction",
                legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=320),
            ),
            tooltip=[
                "feature",
                alt.Tooltip("mean_abs_shap:Q", format=".4f"),
                alt.Tooltip("value_shap_spearman:Q", format=".2f", title="Spearman(value, SHAP)"),
            ],
        )
        .properties(title=title, height=max(200, 22 * len(data)))
    )


def dependence_chart(dep: pd.DataFrame, feature: str) -> alt.Chart:
    g = dep[(dep["feature"] == feature) & dep["feature_value"].notna()]
    base = alt.Chart(g).encode(x=alt.X("feature_value:Q", title=f"{feature} (model space)"))
    band = base.mark_area(opacity=0.25).encode(
        y=alt.Y("p10_shap:Q", title="SHAP (log-odds)"), y2="p90_shap:Q"
    )
    line = base.mark_line(point=True).encode(
        y="mean_shap:Q",
        tooltip=[
            alt.Tooltip("feature_value:Q", format=".3g"),
            alt.Tooltip("mean_shap:Q", format=".3f"),
            alt.Tooltip("p10_shap:Q", format=".3f"),
            alt.Tooltip("p90_shap:Q", format=".3f"),
            "n",
        ],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    return (band + line + zero).properties(
        height=320, title=f"Binned SHAP dependence: {feature} (band = 10-90%)"
    )


def agreement_chart(agreement: pd.DataFrame, model: str, top_n: int = 15) -> alt.Chart:
    g = agreement[agreement["model"] == model].nsmallest(top_n, "mean_rank")
    long = g.melt(
        id_vars=["feature", "mean_rank"],
        value_vars=["rank_shap", "rank_pfi", "rank_lime"],
        var_name="Method",
        value_name="Rank",
    )
    long["Method"] = long["Method"].map(
        {"rank_shap": "SHAP", "rank_pfi": "PFI", "rank_lime": "LIME"}
    )
    order = g.sort_values("mean_rank")["feature"].tolist()
    lines = (
        alt.Chart(long)
        .mark_line(color="#cccccc")
        .encode(
            x=alt.X("Rank:Q", scale=alt.Scale(reverse=True), title="Rank (1 = most important)"),
            y=alt.Y("feature:N", sort=order, title=None),
            detail="feature:N",
        )
    )
    pts = (
        alt.Chart(long)
        .mark_point(filled=True, size=90)
        .encode(
            x="Rank:Q",
            y=alt.Y("feature:N", sort=order),
            color="Method:N",
            shape="Method:N",
            tooltip=["feature", "Method", "Rank"],
        )
    )
    return (lines + pts).properties(
        height=max(220, 24 * len(order)),
        title=f"Importance rank by method - {MODEL_LABELS.get(model, model)}",
    )


def local_bar_chart(
    df: pd.DataFrame, label: str, value: str, title: str, x_title: str
) -> alt.Chart:
    data = df.copy()
    data["effect"] = np.where(data[value] > 0, "raises risk", "lowers risk")
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=x_title),
            y=alt.Y(
                f"{label}:N",
                sort=alt.EncodingSortField(field=value, op="sum", order="descending"),
                title=None,
                axis=YAXIS,
            ),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=["#d62728", "#1f77b4"]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[label, alt.Tooltip(f"{value}:Q", format=".4f")],
        )
        .properties(title=title, height=max(200, 24 * len(data)))
    )


def what_if_chart(curve: pd.DataFrame, feature: str, current: float | str, kind: str) -> alt.Chart:
    if kind == "categorical":
        return (
            alt.Chart(curve)
            .mark_bar()
            .encode(
                x=alt.X("pd:Q", title="Calibrated PD", axis=alt.Axis(format="%")),
                y=alt.Y("value:N", sort="-x", title=feature),
                color=alt.condition(
                    alt.datum.value == str(current), alt.value("#d62728"), alt.value("#1f77b4")
                ),
                tooltip=["value", alt.Tooltip("pd:Q", format=".2%")],
            )
            .properties(
                height=max(180, 22 * len(curve)), title=f"What-if: {feature} (red = current)"
            )
        )
    line = (
        alt.Chart(curve)
        .mark_line(point=True)
        .encode(
            x=alt.X("value:Q", title=feature),
            y=alt.Y("pd:Q", title="Calibrated PD", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("value:Q", format=",.4g"), alt.Tooltip("pd:Q", format=".2%")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"v": [current]}))
        .mark_rule(color="#d62728", strokeDash=[4, 4])
        .encode(x="v:Q")
    )
    return (line + rule).properties(
        height=300, title=f"What-if: vary {feature}, all else fixed (dashed = current)"
    )


# ----------------------------------------------------------------------------- TDA
TDA_COLOR_FIELDS = {
    "observed_rate": ("Observed adverse rate", "orangered", "%"),
    "mean_pd": ("Mean predicted PD", "orangered", "%"),
    "calibration_gap": ("Calibration gap (pred - obs)", "redblue", "+.1%"),
    "gap_z": ("Calibration gap z-score", "redblue", "+.1f"),
    "risk_lift": ("Risk lift vs average", "orangered", ".2f"),
    "brier": ("Brier score", "viridis", ".3f"),
    "share_60m": ("Share 60-month", "purples", "%"),
    "mean_annual_inc": ("Mean annual income", "blues", ",.0f"),
    "mean_dti": ("Mean DTI", "greens", ".1f"),
    "mean_revol_util": ("Mean revolving utilisation", "greens", ".2f"),
    "lens_mean": ("Mean lens value", "viridis", ".3f"),
}


def mapper_chart(
    nodes: pd.DataFrame,
    positions: pd.DataFrame,
    segments: pd.DataFrame,
    color: str,
    highlight: Sequence[int] = (),
) -> alt.Chart:
    """Force-directed Mapper graph: node size = loans, colour = chosen statistic.

    Scroll to zoom, drag to pan, click a node to inspect it.
    """
    title, scheme, fmt = TDA_COLOR_FIELDS.get(color, (color, "viridis", ".3f"))
    data = nodes.merge(positions, on="node_id")
    data["highlight"] = data["node_id"].isin(list(highlight))
    diverging = scheme == "redblue"
    if diverging:
        m = float(np.nanmax(np.abs(data[color]))) or 1.0
        scale = alt.Scale(scheme="redblue", domain=[m, -m], reverse=False)
    else:
        scale = alt.Scale(scheme=scheme)
    pick = alt.selection_point(name="node", fields=["node_id"], on="click", empty=False)
    zoom = alt.selection_interval(name="zoom", bind="scales")
    pad = alt.Scale(zero=False, nice=False, padding=36)  # keep large nodes inside the view
    edges = (
        alt.Chart(segments)
        .mark_rule(color="#9e9e9e", opacity=0.55)
        .encode(
            x=alt.X("x:Q", axis=None, scale=pad),
            y=alt.Y("y:Q", axis=None, scale=pad),
            x2="x2:Q",
            y2="y2:Q",
            strokeWidth=alt.StrokeWidth("shared:Q", scale=alt.Scale(range=[0.4, 4]), legend=None),
        )
    )
    tooltip = [
        alt.Tooltip("node_id:Q", title="Node"),
        alt.Tooltip("size:Q", title="Loans"),
        alt.Tooltip("observed_rate:Q", title="Observed", format=".1%"),
        alt.Tooltip("mean_pd:Q", title="Predicted", format=".1%"),
        alt.Tooltip("calibration_gap:Q", title="Gap", format="+.1%"),
        alt.Tooltip("profile:N", title="Profile"),
    ]
    if "top_purpose" in data.columns:
        tooltip.append(alt.Tooltip("top_purpose:N", title="Top purpose"))
    circles = (
        alt.Chart(data)
        .mark_circle(stroke="black", opacity=0.92)
        .encode(
            x=alt.X("x:Q", axis=None, scale=pad),
            y=alt.Y("y:Q", axis=None, scale=pad),
            size=alt.Size("size:Q", scale=alt.Scale(range=[40, 1100]), legend=None),
            color=alt.Color(
                f"{color}:Q",
                scale=scale,
                title=title,
                legend=alt.Legend(
                    format=fmt, orient="bottom", direction="horizontal", gradientLength=320
                ),
            ),
            strokeWidth=alt.condition(pick, alt.value(3), alt.value(0.4)),
            tooltip=tooltip,
        )
        .add_params(pick)
    )
    layered = edges + circles
    if data["highlight"].any():
        layered += (
            alt.Chart(data[data["highlight"]])
            .mark_circle(fillOpacity=0, stroke="#ff7f0e", strokeWidth=3, size=1600)
            .encode(x="x:Q", y="y:Q")
        )
    return layered.add_params(zoom).properties(height=620)


def point_cloud_chart(cloud2d: pd.DataFrame, color: str) -> alt.Chart:
    enc_color = (
        alt.Color(
            "pd:Q",
            scale=alt.Scale(scheme="orangered"),
            title="Predicted PD",
            legend=alt.Legend(format="%"),
        )
        if color == "pd"
        else alt.Color(
            "Outcome:N", scale=alt.Scale(domain=["Repaid", "Adverse"], range=["#2ca02c", "#d62728"])
        )
    )
    return (
        alt.Chart(cloud2d)
        .mark_circle(size=14, opacity=0.55)
        .encode(
            x=alt.X("PC1:Q"),
            y=alt.Y("PC2:Q"),
            color=enc_color,
            tooltip=["Outcome", alt.Tooltip("pd:Q", format=".1%")],
        )
        .interactive()
        .properties(height=420, title="Loans in the first two principal components")
    )


def persistence_chart(diagrams: pd.DataFrame, dimension: str) -> alt.Chart:
    g = diagrams[diagrams["dimension"] == dimension]
    hi = float(g["death"].max() * 1.05) if len(g) else 1.0
    pts = (
        alt.Chart(g)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X("birth:Q", scale=alt.Scale(domain=[0, hi]), title="Birth (scale)"),
            y=alt.Y("death:Q", scale=alt.Scale(domain=[0, hi]), title="Death (scale)"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=["adverse", "repaid"], range=["#d62728", "#2ca02c"]),
            ),
            tooltip=[
                "group",
                alt.Tooltip("birth:Q", format=".3f"),
                alt.Tooltip("death:Q", format=".3f"),
                alt.Tooltip("persistence:Q", format=".3f"),
            ],
        )
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]}))
        .mark_line(strokeDash=[4, 4], color="black")
        .encode(x="x", y="y")
    )
    return (
        (pts + diag).interactive().properties(height=380, title=f"{dimension} persistence diagram")
    )


def barcode_chart(diagrams: pd.DataFrame, dimension: str, top_n: int = 40) -> alt.Chart:
    g = (
        diagrams[diagrams["dimension"] == dimension]
        .nlargest(top_n, "persistence")
        .reset_index(drop=True)
    )
    g["bar"] = range(len(g))
    return (
        alt.Chart(g)
        .mark_rule(strokeWidth=3)
        .encode(
            x=alt.X("birth:Q", title="Scale"),
            x2="death:Q",
            y=alt.Y("bar:O", axis=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=["adverse", "repaid"], range=["#d62728", "#2ca02c"]),
            ),
            tooltip=["group", alt.Tooltip("persistence:Q", format=".3f")],
        )
        .properties(height=380, title=f"{dimension} barcode (top {top_n} most persistent)")
    )


def profile_chart(profile: pd.DataFrame, top_n: int = 12) -> alt.Chart:
    data = profile.head(top_n)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("std_mean_difference:Q", title="Node mean - overall mean (in SD)"),
            y=alt.Y(
                "feature:N",
                sort=alt.EncodingSortField(
                    field="std_mean_difference", op="sum", order="descending"
                ),
                title=None,
            ),
            color=alt.condition(
                alt.datum.std_mean_difference > 0, alt.value("#d62728"), alt.value("#1f77b4")
            ),
            tooltip=["feature", alt.Tooltip("std_mean_difference:Q", format="+.2f")],
        )
        .properties(height=max(200, 22 * len(data)), title="What distinguishes this node")
    )
