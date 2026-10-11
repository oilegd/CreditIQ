"""Generate figures, explanations (SHAP, PFI, LIME), TDA outputs and Markdown reports.

Usage (from the repository root):
    python scripts/generate_reports.py                  # all trained feature sets
    python scripts/generate_reports.py --skip-shap
    python scripts/generate_reports.py --synthetic
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless rendering for saved figures

import numpy as np
import pandas as pd

from creditiq.cli import (
    base_parser,
    clean_all_path,
    config_from_args,
    processed_path,
    require_file,
)
from creditiq.data.validate import data_quality_summary
from creditiq.explainability.lime_tabular import explain_lime
from creditiq.explainability.shap_analysis import compute_shap, explain_instance
from creditiq.explainability.suite import run_explainability
from creditiq.features.build_features import (
    engineer_features,
    engineered_column_groups,
    resolve_feature_spec,
)
from creditiq.models.artifacts import example_application, read_json
from creditiq.models.predict import load_model
from creditiq.reporting import figures as figs
from creditiq.reporting.markdown import write_evaluation_summary, write_model_card
from creditiq.reporting.results import load_results, load_sql_results, results_available
from creditiq.tda.build import build_tda_clouds, default_mapper_summary
from creditiq.tda.mapper import edge_segments
from creditiq.tda.mapper import layout as mapper_layout
from creditiq.tda.persistence import diagram_frame, persistence_diagrams, persistence_summary
from creditiq.tda.point_cloud import feature_matrix

logger = logging.getLogger("generate_reports")


def eda_figures(cfg, processed: pd.DataFrame, clean_all: pd.DataFrame, sql: dict) -> None:
    fig_dir = cfg.path("figures_dir")
    target = cfg["target"]["name"]
    prep = read_json(cfg.path("metrics_dir") / "data_preparation.json")
    status = pd.DataFrame(prep["status_counts"]).set_index("loan_status")
    figs.save(figs.plot_status_distribution(status), fig_dir / "eda_01_status_distribution.png")
    figs.save(
        figs.plot_vintage_rates(
            sql["02_portfolio_analysis.adverse_rate_by_vintage_term"],
            sql["02_portfolio_analysis.unresolved_share_by_vintage_term"],
        ),
        fig_dir / "eda_02_vintage_rates.png",
    )
    figs.save(
        figs.plot_numeric_distributions(processed, target),
        fig_dir / "eda_03_numeric_distributions.png",
    )
    figs.save(
        figs.plot_grade_analysis(sql["02_portfolio_analysis.adverse_rate_by_grade"]),
        fig_dir / "eda_04_grade.png",
    )
    figs.save(
        figs.plot_rate_by_category(
            sql["02_portfolio_analysis.adverse_rate_by_purpose"],
            "purpose",
            "adverse_rate",
            "n_loans",
            "Adverse rate by loan purpose",
        ),
        fig_dir / "eda_05_purpose.png",
    )
    emp = (
        processed.assign(emp_length=processed["emp_length"].astype("string").fillna("Missing"))
        .groupby("emp_length", observed=True)[target]
        .agg(adverse_rate="mean", n_loans="size")
        .reset_index()
    )
    order = [
        "< 1 year",
        *[f"{i} year{'s' if i > 1 else ''}" for i in range(1, 10)],
        "10+ years",
        "Missing",
    ]
    figs.save(
        figs.plot_rate_by_category(
            emp,
            "emp_length",
            "adverse_rate",
            "n_loans",
            "Adverse rate by employment length",
            order=[o for o in order if o in set(emp["emp_length"])],
        ),
        fig_dir / "eda_06_emp_length.png",
    )
    figs.save(
        figs.plot_missingness(data_quality_summary(clean_all)), fig_dir / "eda_07_missingness.png"
    )
    spec = resolve_feature_spec(cfg.raw, cfg["features"]["default_set"])
    train = processed[processed["split"].astype("string") == "train"]
    nums, _ = engineered_column_groups(spec)
    feats = engineer_features(train.sample(min(len(train), 50000), random_state=cfg.seed), spec)[
        nums
    ]
    figs.save(figs.plot_correlation(feats), fig_dir / "eda_08_correlation.png")


def model_figures(cfg, fs: str, res: dict) -> None:
    fig_dir = cfg.path("figures_dir")
    tm = res["test_metrics"]
    cal = tm[tm["variant"].str.startswith("calibrated")].set_index("model")
    figs.save(
        figs.plot_roc_curves(res["roc_curves"], cal["roc_auc"].to_dict()),
        fig_dir / f"model_roc_{fs}.png",
    )
    figs.save(
        figs.plot_pr_curves(
            res["pr_curves"], cal["pr_auc"].to_dict(), res["summary"]["test_prevalence"]
        ),
        fig_dir / f"model_pr_{fs}.png",
    )
    figs.save(
        figs.plot_reliability(res["calibration_tables"]), fig_dir / f"model_calibration_{fs}.png"
    )
    sel = res["metadata"]["selected_model"]
    r = cal.loc[sel]
    figs.save(
        figs.plot_confusion(int(r.tn), int(r.fp), int(r.fn), int(r.tp), float(r.threshold), sel),
        fig_dir / f"model_confusion_{fs}.png",
    )
    a = res["summary"]["decision_assumptions"]
    figs.save(
        figs.plot_decision_curve(
            res["decision_curve"],
            a["lgd"],
            a["ead_fraction"],
            a["profit_rate_good_loan"],
            chosen=res["summary"]["decision_breakeven_pd"],
        ),
        fig_dir / f"decision_curve_{fs}.png",
    )
    figs.save(figs.plot_segment_gaps(res["segment_performance"]), fig_dir / f"segments_{fs}.png")


def explanations(cfg, fs: str, processed: pd.DataFrame) -> None:
    """SHAP + PFI (single and grouped) + LIME for both models, agreement, figures."""
    import shap

    fig_dir = cfg.path("figures_dir")
    out_dir = cfg.path("metrics_dir") / fs
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, objects = run_explainability(cfg, fs, processed)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    bundle = load_model(cfg.path("models_dir"), fs)
    sel = bundle.name
    figs.save(
        figs.plot_importance(
            tables["shap_global_importance"],
            "mean_abs_shap",
            "Mean |SHAP| (log-odds)",
            f"Global SHAP importance - {sel} ({fs})",
        ),
        fig_dir / f"shap_importance_{fs}.png",
    )
    figs.plt.figure()
    shap.plots.beeswarm(objects[f"shap_{sel}"].to_explanation(), max_display=15, show=False)
    fig = figs.plt.gcf()
    fig.suptitle(f"SHAP summary - {sel} ({fs}); model contribution, not causality", fontsize=10)
    figs.save(fig, fig_dir / f"shap_beeswarm_{fs}.png")
    figs.save(
        figs.plot_dependence(
            tables[f"shap_dependence_{sel}"],
            tables[f"shap_signed_{sel}"]["feature"].tolist(),
            f"Binned SHAP dependence - {sel} ({fs}); line = mean, band = 10-90%",
        ),
        fig_dir / f"shap_dependence_{fs}.png",
    )

    example = example_application(bundle.input_schema)
    local = compute_shap(bundle, example)
    explain_instance(local, 0).to_csv(out_dir / "shap_local_median_applicant.csv", index=False)
    figs.plt.figure()
    shap.plots.waterfall(local.to_explanation()[0], max_display=12, show=False)
    fig = figs.plt.gcf()
    fig.suptitle("SHAP - illustrative median applicant (not a real loan)", fontsize=10)
    figs.save(fig, fig_dir / f"shap_local_example_{fs}.png")
    lime_local = explain_lime(
        bundle, example, n_samples=cfg["explainability"]["lime_samples"], seed=cfg.seed
    )
    lime_local.explanation.to_csv(out_dir / "lime_local_median_applicant.csv", index=False)
    figs.save(
        figs.plot_lime(
            lime_local.explanation,
            f"LIME - illustrative median applicant ({sel}, local R2 = {lime_local.r2:.2f})",
        ),
        fig_dir / f"lime_local_example_{fs}.png",
    )

    figs.save(
        figs.plot_importance(
            tables[f"pfi_single_{sel}"],
            "roc_auc_drop_mean",
            "Drop in test ROC-AUC when shuffled",
            f"Permutation importance of raw inputs - {sel} ({fs})",
            err="roc_auc_drop_std",
        ),
        fig_dir / f"permutation_importance_{fs}.png",
    )
    grouped = {
        m: tables[f"pfi_grouped_{m}"]
        for m in ("lightgbm", "logistic_regression")
        if f"pfi_grouped_{m}" in tables
    }
    figs.save(figs.plot_grouped_pfi(grouped), fig_dir / f"permutation_grouped_{fs}.png")
    figs.save(
        figs.plot_agreement(tables["explainer_agreement"]),
        fig_dir / f"explainer_agreement_{fs}.png",
    )


def topology(cfg, fs: str, processed: pd.DataFrame) -> None:
    """Persist TDA point clouds and report figures (default Mapper + persistence)."""
    build_tda_clouds(cfg, fs, processed)
    summary = default_mapper_summary(cfg, fs)
    graph, nodes, cloud = summary["graph"], summary["nodes"], summary["cloud"]
    out_dir = cfg.path("metrics_dir") / fs
    nodes.to_csv(out_dir / "tda_nodes_default.csv", index=False)
    pd.DataFrame(summary["ideas"]).to_csv(out_dir / "tda_ideas_default.csv", index=False)
    pos = mapper_layout(graph, seed=cfg.seed)
    seg = edge_segments(graph.edges, pos)
    split = cfg["tda"]["default_split"]
    for color in ("observed_rate", "calibration_gap"):
        figs.save(
            figs.plot_mapper(
                nodes,
                pos,
                seg,
                color,
                f"Mapper graph ({fs}, {split} split, lens = model PD); colour = {color}",
            ),
            cfg.path("figures_dir") / f"tda_mapper_{color}_{fs}.png",
        )
    X = feature_matrix(cloud)
    rng = np.random.default_rng(cfg.seed)
    n = cfg["tda"]["persistence_sample"]
    frames = []
    for label, val in (("adverse", 1), ("repaid", 0)):
        idx = np.flatnonzero(cloud["y"].to_numpy() == val)
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        frames.append(diagram_frame(persistence_diagrams(X[pick]), label))
    dg = pd.concat(frames, ignore_index=True)
    persistence_summary(dg).to_csv(out_dir / "tda_persistence_summary.csv", index=False)
    figs.save(
        figs.plot_persistence(dg, f"Persistent homology of applicant feature space ({split})"),
        cfg.path("figures_dir") / f"tda_persistence_{fs}.png",
    )


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--feature-set", default="all")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-tda", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)
    figs.set_style()

    processed = pd.read_parquet(
        require_file(processed_path(cfg), "Run: python scripts/prepare_data.py")
    )
    models_dir, metrics_dir = cfg.path("models_dir"), cfg.path("metrics_dir")
    if not args.skip_eda:
        clean_all = pd.read_parquet(
            require_file(clean_all_path(cfg), "Run: python scripts/prepare_data.py")
        )
        eda_figures(cfg, processed, clean_all, load_sql_results(metrics_dir))

    sets = list(cfg["features"]["sets"]) if args.feature_set == "all" else [args.feature_set]
    sets = [fs for fs in sets if results_available(models_dir, metrics_dir, fs)]
    if not sets:
        raise SystemExit(
            "No evaluated models found. Run train_model.py and evaluate_model.py first."
        )
    for fs in sets:
        if not args.skip_shap:
            explanations(cfg, fs, processed)
        if not args.skip_tda:
            topology(cfg, fs, processed)
        model_figures(cfg, fs, load_results(models_dir, metrics_dir, fs))

    results = {fs: load_results(models_dir, metrics_dir, fs) for fs in sets}
    reports_dir = cfg.path("reports_dir")
    write_evaluation_summary(reports_dir / "evaluation_summary.md", results)
    primary = cfg["features"]["default_set"]
    write_model_card(
        reports_dir / "model_card.md", results, primary if primary in results else sets[0]
    )
    logger.info("Reports written to %s", reports_dir)


if __name__ == "__main__":
    main()
