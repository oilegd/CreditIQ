"""Model training: Dummy baseline, Logistic Regression and LightGBM with bounded tuning."""

from __future__ import annotations

import inspect
import itertools
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from creditiq.config import Config
from creditiq.data.split import split_frames
from creditiq.features.build_features import (
    FeatureSpec,
    engineered_column_groups,
    make_preprocessor,
    resolve_feature_spec,
)
from creditiq.models.artifacts import (
    METADATA_FILE,
    ModelBundle,
    capture_input_schema,
    environment_info,
    save_bundle,
    write_json,
)
from creditiq.models.calibrate import Calibrator, select_calibration
from creditiq.models.evaluate import choose_threshold, probability_metrics

logger = logging.getLogger(__name__)

MODEL_NAMES = ("dummy", "logistic_regression", "lightgbm")


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline
    tuning: pd.DataFrame
    best_params: dict[str, Any]
    fit_seconds: float


@dataclass
class TrainingResult:
    feature_set: str
    models: dict[str, FittedModel]
    bundles: dict[str, ModelBundle]
    validation_metrics: pd.DataFrame
    calibration_selection: dict[str, pd.DataFrame]
    selected_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _score(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    if metric == "pr_auc":
        return float(average_precision_score(y, p))
    raise ValueError(f"Unsupported primary metric '{metric}'")


def select_candidate(rows: list[dict[str, Any]], metric: str, tolerance: float) -> int:
    """Index of the chosen grid candidate.

    Candidates whose validation ``metric`` is within ``tolerance`` of the best are treated
    as ties (differences that small are noise); among them the lowest validation log loss
    wins, so a calibration-distorting setting cannot be chosen on a negligible AUC edge.
    """
    key = f"val_{metric}"
    best = max(r[key] for r in rows)
    eligible = [i for i, r in enumerate(rows) if r[key] >= best - tolerance]
    return min(eligible, key=lambda i: rows[i]["val_log_loss"])


def train_dummy(train: pd.DataFrame, target: str, spec: FeatureSpec, cfg: Config) -> FittedModel:
    t0 = time.perf_counter()
    pre = make_preprocessor("tree", spec, {})
    pipe = Pipeline(
        [("prep", pre), ("model", DummyClassifier(strategy=cfg["model"]["dummy"]["strategy"]))]
    )
    pipe.fit(train, train[target].to_numpy())
    return FittedModel(
        "dummy", pipe, pd.DataFrame(), {"strategy": "prior"}, time.perf_counter() - t0
    )


def train_logistic_regression(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
    cfg: Config,
) -> FittedModel:
    """Grid over C x class_weight; each candidate fitted on train, scored on validation."""
    mcfg = cfg["model"]["logistic_regression"]
    metric = cfg["model"]["primary_metric"]
    y_tr, y_va = train[target].to_numpy(), valid[target].to_numpy()
    t0 = time.perf_counter()
    pre = make_preprocessor("linear", spec, mcfg)
    X_tr = pre.fit_transform(train)
    X_va = pre.transform(valid)
    rows, fitted = [], []
    for C, cw in itertools.product(mcfg["C_grid"], mcfg["class_weight_grid"]):
        clf = LogisticRegression(
            C=C, class_weight=cw, max_iter=mcfg["max_iter"], random_state=cfg.seed
        )
        clf.fit(X_tr, y_tr)
        p_va = clf.predict_proba(X_va)[:, 1]
        score = _score(metric, y_va, p_va)
        rows.append(
            {
                "C": C,
                "class_weight": str(cw),
                f"val_{metric}": score,
                "val_log_loss": float(log_loss(y_va, p_va, labels=[0, 1])),
            }
        )
        fitted.append((clf, {"C": C, "class_weight": cw}))
        logger.info("LogReg C=%s class_weight=%s -> val %s=%.4f", C, cw, metric, score)
    idx = select_candidate(rows, metric, cfg["model"].get("selection_tolerance", 0.0))
    for i, r in enumerate(rows):
        r["selected"] = i == idx
    pipe = Pipeline([("prep", pre), ("model", fitted[idx][0])])
    return FittedModel(
        "logistic_regression", pipe, pd.DataFrame(rows), fitted[idx][1], time.perf_counter() - t0
    )


def train_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
    cfg: Config,
    params_override: Mapping[str, Any] | None = None,
) -> FittedModel:
    """Small parameter grid with early stopping on the validation split."""
    mcfg = cfg["model"]["lightgbm"]
    metric = cfg["model"]["primary_metric"]
    y_tr, y_va = train[target].to_numpy(), valid[target].to_numpy()
    t0 = time.perf_counter()
    pre = make_preprocessor("tree", spec, {})
    X_tr = pre.fit_transform(train)
    X_va = pre.transform(valid)
    _, cats = engineered_column_groups(spec)
    grid = [dict(params_override)] if params_override else mcfg["param_grid"]
    rows, fitted = [], []
    for params in grid:
        clf = lgb.LGBMClassifier(
            n_estimators=mcfg["n_estimators"],
            learning_rate=mcfg["learning_rate"],
            random_state=cfg.seed,
            **mcfg["fixed_params"],
            **{k: v for k, v in params.items() if k != "best_iteration"},
        )
        fit_kwargs: dict[str, Any] = {
            "eval_metric": "auc",
            "categorical_feature": cats,
            "callbacks": [lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False)],
        }
        # LightGBM >= 4.6 renamed eval_set -> eval_X/eval_y.
        if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
            fit_kwargs |= {"eval_X": (X_va,), "eval_y": (y_va,)}
        else:
            fit_kwargs["eval_set"] = [(X_va, y_va)]
        clf.fit(X_tr, y_tr, **fit_kwargs)
        p_va = clf.predict_proba(X_va)[:, 1]
        score = _score(metric, y_va, p_va)
        rows.append(
            {
                **params,
                "best_iteration": clf.best_iteration_,
                f"val_{metric}": score,
                "val_log_loss": float(log_loss(y_va, p_va, labels=[0, 1])),
            }
        )
        fitted.append((clf, {**params, "best_iteration": int(clf.best_iteration_)}))
        logger.info(
            "LightGBM %s -> best_iter=%s val %s=%.4f", params, clf.best_iteration_, metric, score
        )
    idx = select_candidate(rows, metric, cfg["model"].get("selection_tolerance", 0.0))
    for i, r in enumerate(rows):
        r["selected"] = i == idx
    pipe = Pipeline([("prep", pre), ("model", fitted[idx][0])])
    return FittedModel(
        "lightgbm", pipe, pd.DataFrame(rows), fitted[idx][1], time.perf_counter() - t0
    )


def train_all(
    processed: pd.DataFrame, cfg: Config, feature_set: str | None = None
) -> TrainingResult:
    """Train all candidates; select on validation; calibrate and set threshold on calibration."""
    spec = resolve_feature_spec(cfg.raw, feature_set)
    target = cfg["target"]["name"]
    frames = split_frames(processed)
    train, valid, calib = frames["train"], frames["validation"], frames["calibration"]
    metric = cfg["model"]["primary_metric"]
    logger.info(
        "Feature set '%s': train=%s valid=%s calib=%s (test held out)",
        spec.name,
        f"{len(train):,}",
        f"{len(valid):,}",
        f"{len(calib):,}",
    )

    models = {
        "dummy": train_dummy(train, target, spec, cfg),
        "logistic_regression": train_logistic_regression(train, valid, target, spec, cfg),
        "lightgbm": train_lightgbm(train, valid, target, spec, cfg),
    }

    ccfg = cfg["model"]["calibration"]
    n_bins = cfg["evaluation"]["calibration_bins"]
    schema = capture_input_schema(train, spec)
    y_va, y_ca = valid[target].to_numpy(), calib[target].to_numpy()
    val_rows, cal_tables, bundles = [], {}, {}
    for name, fm in models.items():
        p_va = fm.pipeline.predict_proba(valid)[:, 1]
        p_ca_raw = fm.pipeline.predict_proba(calib)[:, 1]
        if name == "dummy":
            calibrator = Calibrator("none").fit(p_ca_raw, y_ca)
        else:
            calibrator, table = select_calibration(
                p_ca_raw,
                y_ca,
                methods=ccfg["methods"],
                cv_folds=ccfg["cv_folds"],
                metric=ccfg["selection_metric"],
                seed=cfg.seed,
            )
            cal_tables[name] = table
        p_ca = calibrator.transform(p_ca_raw)
        threshold = (
            choose_threshold(y_ca, p_ca, cfg["model"]["threshold"]["objective"])
            if name != "dummy"
            else 0.5
        )
        # Validation scores are raw (selection only); threshold metrics belong to calibrated
        # scores and are reported on the test set.
        val_metrics = probability_metrics(y_va, p_va, n_bins=n_bins)
        val_rows.append(
            {"model": name, "scores": "raw", **val_metrics, "fit_seconds": fm.fit_seconds}
        )
        bundle_meta: dict[str, Any] = {"best_params": fm.best_params}
        if name == "logistic_regression":
            # Aggregate training means in model space: lets linear SHAP run without training rows.
            means = fm.pipeline.named_steps["prep"].transform(train).mean()
            bundle_meta["transformed_feature_means"] = {k: float(v) for k, v in means.items()}
        bundles[name] = ModelBundle(
            name=name,
            feature_set=spec.name,
            pipeline=fm.pipeline,
            calibrator=calibrator,
            threshold=threshold,
            input_schema=schema,
            metadata=bundle_meta,
        )
    val_df = pd.DataFrame(val_rows).set_index("model")
    candidates = val_df.drop(index="dummy")
    selected = str(candidates[metric].idxmax())
    logger.info("Validation %s by model:\n%s", metric, val_df[[metric, "pr_auc", "brier"]])
    logger.info("Selected model: %s", selected)

    nums, cats = engineered_column_groups(spec)
    metadata = {
        "project_version": cfg["project"]["model_version"],
        "feature_set": spec.name,
        "feature_set_description": spec.description,
        "selected_model": selected,
        "selection_metric": metric,
        "target": {
            k: cfg["target"][k]
            for k in ("name", "positive_statuses", "negative_statuses", "excluded_statuses")
        },
        "split": {
            "strategy": cfg["split"]["strategy"],
            **{
                name: {
                    "n": int(len(f)),
                    "adverse_rate": float(f[target].mean()),
                    "first_issue": str(pd.to_datetime(f[cfg["split"]["date_column"]]).min().date()),
                    "last_issue": str(pd.to_datetime(f[cfg["split"]["date_column"]]).max().date()),
                }
                for name, f in frames.items()
            },
        },
        "input_columns": spec.input_columns,
        "engineered_numeric_features": nums,
        "engineered_categorical_features": cats,
        "models": {
            name: {
                "best_params": models[name].best_params,
                "calibration_method": bundles[name].calibrator.method,
                "threshold": bundles[name].threshold,
                "fit_seconds": models[name].fit_seconds,
            }
            for name in models
        },
        "random_seed": cfg.seed,
        "environment": environment_info(),
    }
    for b in bundles.values():
        b.metadata.update(
            {
                k: metadata[k]
                for k in ("project_version", "feature_set", "target", "split", "environment")
            }
        )
    return TrainingResult(
        feature_set=spec.name,
        models=models,
        bundles=bundles,
        validation_metrics=val_df,
        calibration_selection=cal_tables,
        selected_model=selected,
        metadata=metadata,
    )


def save_training_result(result: TrainingResult, models_dir: Path, cfg: Config) -> Path:
    """Persist bundles, tuning tables and metadata under ``models/<feature_set>/``."""
    out_dir = models_dir / result.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)
    for bundle in result.bundles.values():
        save_bundle(bundle, out_dir)
    for name, fm in result.models.items():
        if not fm.tuning.empty:
            fm.tuning.to_csv(out_dir / f"tuning_{name}.csv", index=False)
    for name, table in result.calibration_selection.items():
        table.to_csv(out_dir / f"calibration_selection_{name}.csv", index=False)
    result.validation_metrics.to_csv(out_dir / "validation_metrics.csv")
    write_json({**result.metadata, "config": cfg.raw}, out_dir / METADATA_FILE)
    logger.info("Artifacts written to %s", out_dir)
    return out_dir
