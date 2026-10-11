"""Publication-quality figures. Functions take tidy DataFrames and return a Figure.

The matplotlib backend is chosen by the caller (scripts/app use the non-interactive
"Agg" backend; notebooks keep their inline backend).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PALETTE = {
    "dummy": "#9e9e9e",
    "logistic_regression": "#1f77b4",
    "lightgbm": "#d62728",
}
MODEL_LABELS = {
    "dummy": "Dummy (prior)",
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def _pct(ax: plt.Axes, axis: str = "y") -> None:
    fmt = mtick.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------- EDA
def plot_status_distribution(status_counts: pd.DataFrame) -> plt.Figure:
    """Bar chart of loan_status counts coloured by target class."""
    df = status_counts.reset_index().sort_values("n_loans", ascending=True)
    colors = (
        df["target_class"]
        .map(
            {
                "positive (adverse)": "#d62728",
                "negative": "#2ca02c",
                "excluded (unresolved)": "#9e9e9e",
            }
        )
        .fillna("#000000")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["loan_status"], df["n_loans"], color=colors)
    for y, (n, pct) in enumerate(zip(df["n_loans"], df["share_pct"], strict=True)):
        ax.text(n, y, f"  {n:,} ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Number of loans")
    ax.set_title("Loan status at data snapshot (red = adverse, green = repaid, grey = excluded)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlim(0, df["n_loans"].max() * 1.3)
    return fig


def plot_vintage_rates(vintage: pd.DataFrame, unresolved: pd.DataFrame) -> plt.Figure:
    """Adverse rate (resolved loans) and unresolved share by issue quarter and term."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for term, g in vintage.groupby("term"):
        axes[0].plot(
            pd.to_datetime(g["issue_quarter"]), g["adverse_rate"], marker="o", label=str(term)
        )
    axes[0].set_title("Adverse-outcome rate among resolved loans")
    axes[0].set_ylabel("Adverse rate")
    _pct(axes[0])
    for term, g in unresolved.groupby("term"):
        axes[1].plot(
            pd.to_datetime(g["issue_quarter"]), g["unresolved_share"], marker="o", label=str(term)
        )
    axes[1].set_title("Share unresolved at snapshot (excluded from target)")
    axes[1].set_ylabel("Unresolved share")
    _pct(axes[1])
    for ax in axes:
        ax.set_xlabel("Issue quarter")
        ax.legend(title="Term")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, target: str) -> plt.Figure:
    """Distributions of key applicant numerics split by outcome (density-normalised)."""
    hist_years = (
        pd.to_datetime(df["issue_d"]) - pd.to_datetime(df["earliest_cr_line"])
    ).dt.days / 365.25
    panels = [
        ("Loan amount (USD)", df["loan_amnt"], None),
        ("Annual income (USD, log scale)", df["annual_inc"].clip(lower=1000), "log"),
        ("Debt-to-income ratio (%)", df["dti"], None),
        ("Credit history length (years)", hist_years, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    outcome = df[target].map({0: "Fully paid", 1: "Adverse"})
    for ax, (label, values, scale) in zip(axes.ravel(), panels, strict=True):
        frame = pd.DataFrame({"v": values, "Outcome": outcome}).dropna()
        if scale == "log":
            frame = frame[
                frame["v"].between(frame["v"].quantile(0.001), frame["v"].quantile(0.999))
            ]
        sns.histplot(
            data=frame,
            x="v",
            hue="Outcome",
            stat="density",
            common_norm=False,
            bins=50,
            element="step",
            log_scale=(scale == "log"),
            ax=ax,
            palette={"Fully paid": "#2ca02c", "Adverse": "#d62728"},
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    fig.suptitle("Applicant characteristics by outcome (resolved loans)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rate_by_category(
    df: pd.DataFrame,
    category: str,
    rate: str,
    n: str,
    title: str,
    order: Sequence[str] | None = None,
) -> plt.Figure:
    data = df.copy()
    if order is not None:
        data = data.set_index(category).reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(data))))
    ax.barh(data[category].astype(str), data[rate], color="#1f77b4")
    for y, (r, k) in enumerate(zip(data[rate], data[n], strict=True)):
        ax.text(r, y, f"  {r:.1%}  (n={int(k):,})", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Observed adverse-outcome rate")
    _pct(ax, "x")
    ax.set_xlim(0, data[rate].max() * 1.35)
    ax.set_title(title)
    return fig


def plot_grade_analysis(grade_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        grade_df["grade"],
        grade_df["adverse_rate"],
        color="#d62728",
        alpha=0.8,
        label="Adverse rate",
    )
    ax.set_ylabel("Observed adverse rate")
    _pct(ax)
    ax2 = ax.twinx()
    ax2.plot(
        grade_df["grade"],
        grade_df["mean_int_rate"],
        color="#1f77b4",
        marker="o",
        label="Mean interest rate",
    )
    ax2.set_ylabel("Mean interest rate")
    _pct(ax2)
    ax2.grid(False)
    ax.set_xlabel("Lending Club grade (origination-time, not used in applicant model)")
    ax.set_title("Adverse rate and pricing by Lending Club grade")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    return fig


def plot_missingness(quality: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    data = (
        quality[quality["pct_missing"] > 0].sort_values("pct_missing", ascending=False).head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(data))))
    ax.barh(data.index.astype(str), data["pct_missing"] / 100, color="#ff7f0e")
    ax.invert_yaxis()
    _pct(ax, "x")
    ax.set_xlabel("Share of rows missing")
    ax.set_title(f"Missingness (top {len(data)} columns with any missing values)")
    return fig


def plot_correlation(features: pd.DataFrame) -> plt.Figure:
    corr = features.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman correlation"},
        linewidths=0.3,
    )
    ax.set_title("Spearman correlation of engineered numeric features (training split)")
    return fig


# ------------------------------------------------------------------------ Models
def plot_roc_curves(roc: pd.DataFrame, aucs: dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model, g in roc.groupby("model"):
        ax.plot(
            g["fpr"],
            g["tpr"],
            color=PALETTE.get(model),
            label=f"{MODEL_LABELS.get(model, model)} (AUC {aucs.get(model, np.nan):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - out-of-time test set")
    ax.legend(loc="lower right")
    return fig


def plot_pr_curves(pr: pd.DataFrame, aps: dict[str, float], prevalence: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model, g in pr.groupby("model"):
        ax.plot(
            g["recall"],
            g["precision"],
            color=PALETTE.get(model),
            label=f"{MODEL_LABELS.get(model, model)} (AP {aps.get(model, np.nan):.3f})",
        )
    ax.axhline(prevalence, ls="--", color="black", lw=0.8, label=f"Prevalence {prevalence:.1%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - out-of-time test set")
    ax.legend(loc="upper right")
    return fig


def plot_reliability(cal: pd.DataFrame) -> plt.Figure:
    models = list(dict.fromkeys(cal["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        g = cal[cal["model"] == model]
        for variant, gg in g.groupby("variant"):
            ax.plot(
                gg["mean_predicted"],
                gg["observed_rate"],
                marker="o",
                ls="-" if variant == "calibrated" else ":",
                label=variant,
            )
        lim = max(g["mean_predicted"].max(), g["observed_rate"].max()) * 1.05
        ax.plot([0, lim], [0, lim], ls="--", color="black", lw=0.8, label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability (decile)")
        ax.set_ylabel("Observed adverse rate")
        ax.set_title(f"Reliability - {MODEL_LABELS.get(model, model)}")
        _pct(ax)
        _pct(ax, "x")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(tn: int, fp: int, fn: int, tp: int, threshold: float, model: str) -> plt.Figure:
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        mat,
        annot=np.array([[f"{v:,}\n({v / mat.sum():.1%})" for v in row] for row in mat]),
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted repaid", "Predicted adverse"],
        yticklabels=["Actual repaid", "Actual adverse"],
    )
    ax.set_title(
        f"Confusion matrix - {MODEL_LABELS.get(model, model)}\n"
        f"(threshold {threshold:.3f}, chosen on calibration split)"
    )
    return fig


def plot_importance(
    df: pd.DataFrame, value: str, label: str, title: str, top_n: int = 20, err: str | None = None
) -> plt.Figure:
    data = df.head(top_n).iloc[::-1]
    name_col = data.columns[0]
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.32 * len(data))))
    ax.barh(
        data[name_col].astype(str), data[value], xerr=data[err] if err else None, color="#1f77b4"
    )
    ax.set_xlabel(label)
    ax.set_title(title)
    return fig


def plot_decision_curve(
    curve: pd.DataFrame, lgd: float, ead_fraction: float, margin: float, chosen: float | None = None
) -> plt.Figure:
    net = margin * curve["volume_approved_good"] - lgd * ead_fraction * curve["volume_approved_bad"]
    approval = curve["n_approved"] / curve["n_total"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["threshold"], net / 1e6, color="#2ca02c", label="Net value (USD m)")
    ax.set_xlabel("Approve if calibrated PD < threshold")
    ax.set_ylabel("Net value on test portfolio (USD m)")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], approval, color="#1f77b4", ls="--", label="Approval rate")
    ax2.set_ylabel("Approval rate")
    _pct(ax2)
    ax2.grid(False)
    best = curve["threshold"].iloc[int(np.argmax(net))]
    ax.axvline(best, color="#2ca02c", ls=":", lw=1, label=f"Best cut-off ({best:.3f})")
    if chosen is not None:
        ax.axvline(chosen, color="black", ls=":", lw=1, label=f"Break-even PD ({chosen:.3f})")
    ax.set_title(
        f"HYPOTHETICAL decision simulation (LGD={lgd:.0%}, EAD={ead_fraction:.0%}, "
        f"margin={margin:.0%})"
    )
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.15))
    return fig


def plot_segment_gaps(seg: pd.DataFrame) -> plt.Figure:
    segments = list(dict.fromkeys(seg["segment"]))
    fig, axes = plt.subplots(len(segments), 1, figsize=(9, 2.3 * len(segments)), squeeze=False)
    for ax, s in zip(axes[:, 0], segments, strict=True):
        g = seg[seg["segment"] == s]
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["observed_adverse_rate"], width=0.4, label="Observed", color="#7f7f7f")
        ax.bar(x + 0.2, g["mean_predicted"], width=0.4, label="Mean predicted", color="#d62728")
        ax.set_xticks(
            x, [f"{grp}\n(n={n:,})" for grp, n in zip(g["group"], g["n"], strict=True)], fontsize=7
        )
        ax.set_title(str(g["segment_label"].iloc[0]), fontsize=10)
        _pct(ax)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "Calibration by segment - out-of-time test (descriptive, not a fairness certification)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Explainability
def plot_dependence(dep: pd.DataFrame, features: Sequence[str], title: str) -> plt.Figure:
    """Binned SHAP dependence: mean SHAP (line) and 10-90% band per feature-value bin."""
    feats = [f for f in features if f in set(dep["feature"])][:6]
    ncols = 3
    nrows = max(1, int(np.ceil(len(feats) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), feats, strict=False):
        g = dep[(dep["feature"] == f) & dep["feature_value"].notna()].sort_values("feature_value")
        ax.fill_between(
            g["feature_value"], g["p10_shap"], g["p90_shap"], alpha=0.25, color="#1f77b4"
        )
        ax.plot(g["feature_value"], g["mean_shap"], marker="o", ms=3, color="#1f77b4")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("feature value (model space)")
        ax.set_ylabel("SHAP (log-odds)")
    for ax in axes.ravel()[len(feats) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_grouped_pfi(tables: dict[str, pd.DataFrame], metric: str = "roc_auc") -> plt.Figure:
    """Grouped permutation importance for several models (mean +/- std)."""
    frames = [t.assign(model=m) for m, t in tables.items() if not t.empty]
    data = pd.concat(frames, ignore_index=True)
    order = data.groupby("feature")[f"{metric}_drop_mean"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(order))))
    height = 0.8 / max(1, len(tables))
    for i, (m, g) in enumerate(data.groupby("model")):
        g = g.set_index("feature").reindex(order)
        y = np.arange(len(order)) + i * height
        ax.barh(
            y,
            g[f"{metric}_drop_mean"],
            height=height,
            xerr=g[f"{metric}_drop_std"],
            color=PALETTE.get(m),
            label=MODEL_LABELS.get(m, m),
        )
    ax.set_yticks(np.arange(len(order)) + height * (len(tables) - 1) / 2, order)
    ax.set_xlabel(f"Degradation in test {metric} when the group is shuffled")
    ax.set_title("Grouped permutation importance (related inputs shuffled jointly)")
    ax.legend()
    return fig


def plot_lime(explanation: pd.DataFrame, title: str, top_n: int = 12) -> plt.Figure:
    data = explanation.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * len(data))))
    ax.barh(
        data["condition"], data["weight"], color=np.where(data["weight"] > 0, "#d62728", "#1f77b4")
    )
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("LIME weight (change in calibrated probability)")
    ax.set_title(title)
    return fig


def plot_agreement(agreement: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Rank of each raw input under SHAP, PFI and LIME (lower = more important)."""
    models = list(dict.fromkeys(agreement["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models, strict=True):
        g = agreement[agreement["model"] == m].nsmallest(top_n, "mean_rank").iloc[::-1]
        y = np.arange(len(g))
        for col, mk, lab in (
            ("rank_shap", "o", "SHAP"),
            ("rank_pfi", "s", "PFI"),
            ("rank_lime", "^", "LIME"),
        ):
            ax.scatter(g[col], y, marker=mk, label=lab, s=45)
        ax.set_yticks(y, g["feature"])
        ax.invert_xaxis()
        ax.set_xlabel("Importance rank (1 = most important)")
        ax.set_title(f"Explainer agreement - {MODEL_LABELS.get(m, m)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ TDA
def plot_mapper(
    nodes: pd.DataFrame, positions: pd.DataFrame, segments: pd.DataFrame, color: str, title: str
) -> plt.Figure:
    data = nodes.merge(positions, on="node_id")
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in segments.itertuples(index=False):
        ax.plot([r.x, r.x2], [r.y, r.y2], color="#bbbbbb", lw=0.8, zorder=1)
    vmax = np.nanmax(np.abs(data[color])) if color == "calibration_gap" else None
    sc = ax.scatter(
        data["x"],
        data["y"],
        s=20 + 400 * data["size"] / data["size"].max(),
        c=data[color],
        cmap="RdBu_r" if color == "calibration_gap" else "viridis",
        vmin=-vmax if vmax else None,
        vmax=vmax,
        edgecolor="black",
        lw=0.4,
        zorder=2,
    )
    fig.colorbar(sc, ax=ax, label=color.replace("_", " "))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return fig


def plot_persistence(diagrams: pd.DataFrame, title: str) -> plt.Figure:
    dims = sorted(diagrams["dimension"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(5.5 * len(dims), 5), squeeze=False)
    for ax, d in zip(axes[0], dims, strict=True):
        g = diagrams[diagrams["dimension"] == d]
        for grp, gg in g.groupby("group"):
            ax.scatter(gg["birth"], gg["death"], s=12, alpha=0.6, label=grp)
        hi = g["death"].max() * 1.05 if len(g) else 1
        ax.plot([0, hi], [0, hi], ls="--", color="black", lw=0.8)
        ax.set_xlabel("birth (scale)")
        ax.set_ylabel("death (scale)")
        ax.set_title(f"{d} persistence diagram")
        ax.legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
