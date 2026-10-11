"""Probability calibration (none / Platt-sigmoid / isotonic) with leakage-free method selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

EPS = 1e-6
METHODS = ("none", "sigmoid", "isotonic")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Monotone map from raw model scores to calibrated probabilities.

    ``sigmoid`` fits Platt scaling on the raw log-odds; ``isotonic`` fits a
    non-decreasing step function; ``none`` is the identity (clipped to (0, 1)).
    """

    def __init__(self, method: str = "none"):
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method '{method}'. Choose from {METHODS}.")
        self.method = method
        self.model_: LogisticRegression | IsotonicRegression | None = None
        self.n_fit_: int = 0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Calibrator:
        p_raw = np.asarray(p_raw, dtype="float64")
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 and self.method != "none":
            raise ValueError("Calibration data must contain both classes.")
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000)
            self.model_.fit(_logit(p_raw).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
            self.model_.fit(p_raw, y)
        self.n_fit_ = len(y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype="float64")
        if self.method == "none":
            out = p_raw
        elif self.model_ is None:
            raise RuntimeError("Calibrator is not fitted.")
        elif self.method == "sigmoid":
            out = self.model_.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
        else:
            out = self.model_.predict(p_raw)
        return np.clip(out, EPS, 1 - EPS)

    def __repr__(self) -> str:
        return f"Calibrator(method={self.method!r}, n_fit={self.n_fit_})"


def _score(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    if metric == "brier":
        return float(brier_score_loss(y, p))
    if metric == "log_loss":
        return float(log_loss(y, p, labels=[0, 1]))
    raise ValueError(f"Unknown calibration selection metric '{metric}'")


def select_calibration(
    p_raw: np.ndarray,
    y: np.ndarray,
    methods: Sequence[str] = METHODS,
    cv_folds: int = 5,
    metric: str = "brier",
    seed: int = 42,
) -> tuple[Calibrator, pd.DataFrame]:
    """Choose a calibration method by K-fold CV *within* the calibration split.

    Each method is fitted on K-1 folds and scored on the held-out fold, so the
    comparison never scores a calibrator on data it was fitted on. The winner is
    then refitted on the full calibration split. The test split is never touched.
    """
    p_raw = np.asarray(p_raw, dtype="float64")
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows = []
    for method in methods:
        briers, losses = [], []
        for fit_idx, eval_idx in skf.split(p_raw, y):
            cal = Calibrator(method).fit(p_raw[fit_idx], y[fit_idx])
            p_eval = cal.transform(p_raw[eval_idx])
            briers.append(_score(y[eval_idx], p_eval, "brier"))
            losses.append(_score(y[eval_idx], p_eval, "log_loss"))
        rows.append(
            {
                "method": method,
                "cv_brier": float(np.mean(briers)),
                "cv_brier_std": float(np.std(briers)),
                "cv_log_loss": float(np.mean(losses)),
                "cv_log_loss_std": float(np.std(losses)),
            }
        )
    table = pd.DataFrame(rows)
    sort_col = "cv_brier" if metric == "brier" else "cv_log_loss"
    best = str(table.sort_values(sort_col).iloc[0]["method"])
    table["selected"] = table["method"] == best
    logger.info("Calibration CV results:\n%s", table.to_string(index=False))
    return Calibrator(best).fit(p_raw, y), table


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table: mean predicted vs observed adverse rate."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    out = df.groupby("bin").agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        min_predicted=("p", "min"),
        max_predicted=("p", "max"),
    )
    out["abs_gap"] = (out["mean_predicted"] - out["observed_rate"]).abs()
    return out.reset_index()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Sample-weighted mean |predicted - observed| over quantile bins."""
    table = calibration_table(y, p, n_bins)
    return float(np.average(table["abs_gap"], weights=table["n"]))
