"""Discrimination, calibration and threshold metrics with bootstrap confidence intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from creditiq.models.calibrate import EPS, calibration_table, expected_calibration_error

logger = logging.getLogger(__name__)


def _validate(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    if y.shape != p.shape:
        raise ValueError(f"y and p shapes differ: {y.shape} vs {p.shape}")
    if np.any(~np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("Probabilities must be finite and within [0, 1].")
    return y, p


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision/recall/F1/FPR and confusion counts for ``p >= threshold``."""
    y, p = _validate(y, p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "flagged_rate": float(pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def probability_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Threshold-free metrics: ROC-AUC, PR-AUC, Brier, log loss, ECE, prevalence."""
    y, p = _validate(y, p)
    pc = np.clip(p, EPS, 1 - EPS)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, pc)),
        "log_loss": float(log_loss(y, pc, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins)
        if len(np.unique(p)) > 1
        else float(abs(p.mean() - y.mean())),
    }


def evaluate_predictions(
    y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int = 10
) -> dict[str, float]:
    return probability_metrics(y, p, n_bins) | threshold_metrics(y, p, threshold)


def choose_threshold(y: np.ndarray, p: np.ndarray, objective: str = "f1") -> float:
    """Pick the probability threshold maximising ``objective`` on the given (non-test) data."""
    y, p = _validate(y, p)
    if objective != "f1":
        raise ValueError(f"Unsupported threshold objective '{objective}'")
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.where(
        (precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, EPS), 0
    )
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": lambda y, p: brier_score_loss(y, np.clip(p, EPS, 1 - EPS)),
}


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    reps: int = 200,
    ci: float = 0.95,
    seed: int = 42,
    metrics: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals (resampling loans with replacement)."""
    y, p = _validate(y, p)
    metrics = metrics or METRIC_FUNCS
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        for name, fn in metrics.items():
            samples[name].append(float(fn(yb, pb)))
    alpha = (1 - ci) / 2
    out = {}
    for name, vals in samples.items():
        arr = np.asarray(vals)
        out[name] = {
            "point": float(metrics[name](y, p)),
            "lower": float(np.quantile(arr, alpha)) if len(arr) else float("nan"),
            "upper": float(np.quantile(arr, 1 - alpha)) if len(arr) else float("nan"),
            "reps": len(arr),
        }
    return out


def paired_bootstrap_auc_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    reps: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """CI for AUC(a) - AUC(b) using the same bootstrap resamples for both models."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    alpha = (1 - ci) / 2
    return {
        "point": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
        "lower": float(np.quantile(diffs, alpha)),
        "upper": float(np.quantile(diffs, 1 - alpha)),
    }


def curve_data(y: np.ndarray, p: np.ndarray, max_points: int = 400) -> dict[str, pd.DataFrame]:
    """ROC and PR curve points (down-sampled for plotting/storage)."""
    y, p = _validate(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)

    def _thin(frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= max_points:
            return frame
        idx = np.unique(np.linspace(0, len(frame) - 1, max_points).astype(int))
        return frame.iloc[idx].reset_index(drop=True)

    return {
        "roc": _thin(pd.DataFrame({"fpr": fpr, "tpr": tpr})),
        "pr": _thin(pd.DataFrame({"recall": recall, "precision": precision})),
    }


__all__ = [
    "bootstrap_ci",
    "calibration_table",
    "choose_threshold",
    "curve_data",
    "evaluate_predictions",
    "paired_bootstrap_auc_diff",
    "probability_metrics",
    "threshold_metrics",
]
