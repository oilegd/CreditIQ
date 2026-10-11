"""Final (test-set) evaluation of trained bundles plus decision and segment analyses.

The test split is used here for the first and only time: models, calibrators and
thresholds were all fixed using train / validation / calibration data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from creditiq.config import Config
from creditiq.data.split import SPLIT_COLUMN, assign_random_split, split_frames
from creditiq.features.build_features import resolve_feature_spec
from creditiq.models.artifacts import write_json
from creditiq.models.calibrate import calibration_table
from creditiq.models.evaluate import (
    bootstrap_ci,
    curve_data,
    evaluate_predictions,
    paired_bootstrap_auc_diff,
    probability_metrics,
)
from creditiq.models.predict import available_models, load_metadata, load_model
from creditiq.models.train import train_lightgbm
from creditiq.risk.decision_analysis import (
    DecisionAssumptions,
    breakeven_pd,
    bucket_summary,
    sensitivity_analysis,
    threshold_scenarios,
)
from creditiq.risk.segments import segment_performance

logger = logging.getLogger(__name__)

DECISION_CURVE_GRID = np.round(np.arange(0.01, 1.0, 0.005), 4)


@dataclass
class EvaluationOutputs:
    feature_set: str
    selected_model: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def decision_curve(y: np.ndarray, p: np.ndarray, loan_amnt: np.ndarray) -> pd.DataFrame:
    """Aggregates per cut-off so any LGD/EAD/margin can be applied later without row data."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(loan_amnt, dtype="float64")
    rows = []
    for t in DECISION_CURVE_GRID:
        a = p < t
        rows.append(
            {
                "threshold": float(t),
                "n_total": int(len(y)),
                "n_approved": int(a.sum()),
                "n_approved_bad": int((a & (y == 1)).sum()),
                "n_rejected_good": int((~a & (y == 0)).sum()),
                "volume_approved_good": float(amt[a & (y == 0)].sum()),
                "volume_approved_bad": float(amt[a & (y == 1)].sum()),
                "volume_rejected_good": float(amt[~a & (y == 0)].sum()),
                "volume_rejected_bad": float(amt[~a & (y == 1)].sum()),
                "pd_weighted_volume_approved": float((p[a] * amt[a]).sum()),
            }
        )
    return pd.DataFrame(rows)


SCORE_BIN_WIDTH = 0.005
LATE_STATUSES = ("Late (16-30 days)", "Late (31-120 days)")


def score_distribution(y: np.ndarray, p: np.ndarray, loan_amnt: np.ndarray) -> pd.DataFrame:
    """Counts of adverse/repaid loans and volume per calibrated-PD bin (aggregate only).

    Enables exact threshold metrics at any cut-off (to bin resolution) without row data.
    """
    edges = np.arange(0, 1 + SCORE_BIN_WIDTH, SCORE_BIN_WIDTH)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "y": np.asarray(y).astype(int), "amt": loan_amnt, "p": p})
    g = df.groupby("bin").agg(
        n_adverse=("y", "sum"), n=("y", "size"), volume=("amt", "sum"), pd_sum=("p", "sum")
    )
    g = g.reindex(range(len(edges) - 1), fill_value=0)
    g["n_repaid"] = g["n"] - g["n_adverse"]
    g["bin_lower"] = edges[:-1]
    g["bin_upper"] = edges[1:]
    return g.reset_index(drop=True)[
        ["bin_lower", "bin_upper", "n", "n_adverse", "n_repaid", "volume", "pd_sum"]
    ]


def unresolved_bounds(
    test: pd.DataFrame, clean_all: pd.DataFrame, bundle, cfg: Config
) -> pd.DataFrame:
    """Test metrics if loans excluded as unresolved were included under explicit scenarios.

    Excluded loans are survivors at the snapshot (not yet charged off), so excluding them
    enriches the remaining - mainly 60-month - loans with charge-offs. Scoring them shows
    how sensitive the reported metrics are to that choice.
    """
    target = cfg["target"]["name"]
    date_col = cfg["split"]["date_column"]
    month = pd.to_datetime(clean_all[date_col]).dt.to_period("M")
    in_window = month > pd.Period(cfg["split"]["calibration_end"], freq="M")
    status = clean_all[cfg["target"]["status_column"]].astype(str)
    unres = clean_all[in_window & status.isin(cfg["target"]["excluded_statuses"])]
    y_t = test[target].to_numpy().astype(int)
    p_t = bundle.predict_proba(test)
    term_t = test["term"].astype(str).to_numpy()
    rows = []
    scenarios: dict[str, np.ndarray | None] = {"reported: unresolved excluded": None}
    if len(unres):
        p_u = bundle.predict_proba(unres)
        st = status.loc[unres.index].to_numpy()
        scenarios |= {
            "all unresolved repaid": np.zeros(len(unres), dtype=int),
            "Current/Grace repaid, Late adverse": np.isin(st, LATE_STATUSES).astype(int),
            "all unresolved adverse (extreme bound)": np.ones(len(unres), dtype=int),
        }
        term_u = unres["term"].astype(str).to_numpy()
    for name, y_u in scenarios.items():
        if y_u is None:
            yy, pp, tt = y_t, p_t, term_t
        else:
            yy, pp, tt = np.r_[y_t, y_u], np.r_[p_t, p_u], np.r_[term_t, term_u]
        m60 = np.char.startswith(tt.astype(str), "60")
        rows.append(
            {
                "scenario": name,
                "n": int(len(yy)),
                "n_unresolved_added": 0 if y_u is None else int(len(y_u)),
                **probability_metrics(yy, pp),
                "prevalence_60m": float(yy[m60].mean()) if m60.any() else np.nan,
                "mean_predicted_60m": float(pp[m60].mean()) if m60.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _metrics_by_period(
    df: pd.DataFrame, y: np.ndarray, p: np.ndarray, date_col: str
) -> pd.DataFrame:
    quarter = pd.to_datetime(df[date_col]).dt.to_period("Q").astype(str).to_numpy()
    rows = []
    for q in sorted(np.unique(quarter)):
        m = quarter == q
        if len(np.unique(y[m])) < 2:
            continue
        rows.append({"issue_quarter": q, **probability_metrics(y[m], p[m])})
    return pd.DataFrame(rows)


def run_evaluation(
    processed: pd.DataFrame,
    cfg: Config,
    feature_set: str,
    clean_all: pd.DataFrame | None = None,
) -> EvaluationOutputs:
    """Evaluate saved models on the test split.

    ``clean_all`` (all cleaned loans, every status) enables the unresolved-loan bounds:
    loans issued in the test window but excluded as unresolved are scored and the test
    metrics recomputed under explicit labelling scenarios.
    """
    models_dir = cfg.path("models_dir")
    meta = load_metadata(models_dir, feature_set)
    selected = meta["selected_model"]
    target = cfg["target"]["name"]
    ecfg = cfg["evaluation"]
    n_bins = ecfg["calibration_bins"]
    frames = split_frames(processed)
    test = frames["test"]
    y = test[target].to_numpy().astype(int)
    out = EvaluationOutputs(feature_set=feature_set, selected_model=selected)

    metric_rows, ci_rows, cal_rows, roc_rows, pr_rows, sens_rows, period_rows = (
        [] for _ in range(7)
    )
    preds: dict[str, np.ndarray] = {}
    for name in available_models(models_dir, feature_set):
        bundle = load_model(models_dir, feature_set, name)
        p_raw = bundle.predict_raw(test)
        p_cal = bundle.calibrator.transform(p_raw)
        preds[name] = p_cal
        raw_m = probability_metrics(y, p_raw, n_bins)
        cal_m = evaluate_predictions(y, p_cal, bundle.threshold, n_bins)
        metric_rows.append({"model": name, "variant": "raw", **raw_m})
        metric_rows.append(
            {"model": name, "variant": f"calibrated ({bundle.calibrator.method})", **cal_m}
        )
        if name != "dummy":
            cis = bootstrap_ci(
                y, p_cal, reps=ecfg["bootstrap_reps"], ci=ecfg["bootstrap_ci"], seed=cfg.seed
            )
            for metric, vals in cis.items():
                ci_rows.append({"model": name, "metric": metric, **vals})
            for variant, p in (("raw", p_raw), ("calibrated", p_cal)):
                cal_rows.append(calibration_table(y, p, n_bins).assign(model=name, variant=variant))
            short = test["term"].astype("string").str.startswith("36").to_numpy()
            sens_rows.append(
                {
                    "model": name,
                    "population": "36-month loans only (fully resolved)",
                    **probability_metrics(y[short], p_cal[short], n_bins),
                }
            )
            sens_rows.append(
                {
                    "model": name,
                    "population": "all resolved test loans",
                    **probability_metrics(y, p_cal, n_bins),
                }
            )
            period_rows.append(
                _metrics_by_period(test, y, p_cal, cfg["split"]["date_column"]).assign(model=name)
            )
        curves = curve_data(y, p_cal)
        roc_rows.append(curves["roc"].assign(model=name))
        pr_rows.append(curves["pr"].assign(model=name))

    out.tables["test_metrics"] = pd.DataFrame(metric_rows)
    out.tables["bootstrap_ci"] = pd.DataFrame(ci_rows)
    out.tables["calibration_tables"] = pd.concat(cal_rows, ignore_index=True)
    out.tables["roc_curves"] = pd.concat(roc_rows, ignore_index=True)
    out.tables["pr_curves"] = pd.concat(pr_rows, ignore_index=True)
    out.tables["sensitivity_36_month"] = pd.DataFrame(sens_rows)
    out.tables["metrics_by_quarter"] = pd.concat(period_rows, ignore_index=True)

    if {"lightgbm", "logistic_regression"} <= preds.keys():
        diff = paired_bootstrap_auc_diff(
            y,
            preds["lightgbm"],
            preds["logistic_regression"],
            reps=ecfg["bootstrap_reps"],
            ci=ecfg["bootstrap_ci"],
            seed=cfg.seed,
        )
        out.summary["auc_diff_lightgbm_minus_logreg"] = diff

    sel = load_model(models_dir, feature_set, selected)
    p_sel = preds[selected]
    out.tables["segment_performance"] = segment_performance(
        test, y, p_sel, sel.threshold, ecfg["segments"]
    )

    dcfg = cfg["decision"]
    assumptions = DecisionAssumptions.from_config(dcfg)
    amount = test["loan_amnt"].to_numpy(dtype="float64")
    out.tables["decision_scenarios"] = threshold_scenarios(
        y, p_sel, amount, dcfg["threshold_grid"], assumptions
    )
    out.tables["decision_sensitivity"] = sensitivity_analysis(
        y,
        p_sel,
        amount,
        dcfg["threshold_grid"],
        dcfg["sensitivity"]["lgd"],
        dcfg["sensitivity"]["profit_rate_good_loan"],
        assumptions.ead_fraction,
    )
    out.tables["risk_buckets"] = bucket_summary(y, p_sel, amount, dcfg["risk_buckets"], assumptions)
    out.tables["decision_curve"] = decision_curve(y, p_sel, amount)
    out.tables["score_distribution"] = score_distribution(y, p_sel, amount)

    if clean_all is not None:
        bounds = []
        for name in (n for n in preds if n != "dummy"):
            bundle = load_model(models_dir, feature_set, name)
            bounds.append(unresolved_bounds(test, clean_all, bundle, cfg).assign(model=name))
        if bounds:
            out.tables["unresolved_bounds"] = pd.concat(bounds, ignore_index=True)

    if ecfg.get("random_split_benchmark") and "lightgbm" in meta["models"]:
        out.tables["random_split_benchmark"] = random_split_benchmark(
            processed,
            cfg,
            feature_set,
            meta["models"]["lightgbm"]["best_params"],
            oot_auc=float(probability_metrics(y, preds["lightgbm"])["roc_auc"]),
            oot_n=int(len(y)),
        )

    out.summary.update(
        {
            "feature_set": feature_set,
            "selected_model": selected,
            "selected_threshold": sel.threshold,
            "selected_calibration": sel.calibrator.method,
            "test_n": int(len(y)),
            "test_prevalence": float(y.mean()),
            "test_period": [
                str(pd.to_datetime(test[cfg["split"]["date_column"]]).min().date()),
                str(pd.to_datetime(test[cfg["split"]["date_column"]]).max().date()),
            ],
            "decision_assumptions": assumptions.to_dict(),
            "decision_breakeven_pd": breakeven_pd(assumptions),
            "assumptions_are_hypothetical": True,
        }
    )
    return out


def random_split_benchmark(
    processed: pd.DataFrame,
    cfg: Config,
    feature_set: str,
    lgb_params: dict[str, Any],
    oot_auc: float,
    oot_n: int,
) -> pd.DataFrame:
    """Retrain LightGBM (same hyper-parameters) on a stratified random split for comparison.

    Random splits mix vintages, so they usually look optimistic relative to the
    out-of-time test. Numbers are not directly comparable (different test loans).
    """
    target = cfg["target"]["name"]
    df = processed.copy()
    df[SPLIT_COLUMN] = assign_random_split(df, target, cfg["split"], cfg.seed)
    frames = split_frames(df)
    spec = resolve_feature_spec(cfg.raw, feature_set)
    params = {k: v for k, v in lgb_params.items() if k != "best_iteration"}
    fm = train_lightgbm(frames["train"], frames["validation"], target, spec, cfg, params)
    y_rt = frames["test"][target].to_numpy()
    p_rt = fm.pipeline.predict_proba(frames["test"])[:, 1]
    auc_random = float(probability_metrics(y_rt, p_rt)["roc_auc"])
    return pd.DataFrame(
        [
            {
                "design": "out-of-time split (configured issue-date boundaries)",
                "test_n": oot_n,
                "roc_auc": oot_auc,
            },
            {
                "design": "stratified random split (all vintages)",
                "test_n": int(len(y_rt)),
                "roc_auc": auc_random,
            },
        ]
    )


def save_evaluation(outputs: EvaluationOutputs, metrics_dir: Path) -> Path:
    out_dir = metrics_dir / outputs.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    write_json(outputs.summary, out_dir / "evaluation_summary.json")
    logger.info("Evaluation outputs written to %s", out_dir)
    return out_dir
