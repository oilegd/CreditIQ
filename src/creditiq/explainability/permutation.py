"""Permutation feature importance (PFI): multi-metric, single-feature and grouped.

Importance = degradation of a test metric when an input (or a group of inputs) is shuffled,
breaking its relationship with the outcome while keeping its marginal distribution.

* ROC-AUC measures ranking; log loss and Brier use calibrated probabilities, so they also
  capture a feature's contribution to probability accuracy.
* Single-feature PFI under-states correlated inputs (the model can partly recover the
  information from a correlated partner). Grouped PFI shuffles related inputs *jointly*
  with one shared permutation, which is the standard remedy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from creditiq.models.artifacts import ModelBundle, coerce_to_schema

EPS = 1e-6
# name -> (score function, higher_is_better)
METRICS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
    "roc_auc": (roc_auc_score, True),
    "log_loss": (lambda y, p: log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]), False),
    "brier": (brier_score_loss, False),
}


def _degradation(base: float, permuted: float, higher_is_better: bool) -> float:
    return base - permuted if higher_is_better else permuted - base


def permutation_importance(
    bundle: ModelBundle,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Mapping[str, Sequence[str]] | None = None,
    metrics: Sequence[str] = ("roc_auc", "log_loss", "brier"),
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per feature/group with mean and std degradation for each metric.

    ``groups`` maps a group name to raw input columns that are permuted together; when
    omitted, every perturbable raw input is its own group. Reference-only inputs (the
    issue date used for credit-history length) are never permuted.
    """
    Xr = coerce_to_schema(X, bundle.input_schema).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    if groups is None:
        groups = {c: [c] for c, m in bundle.input_schema.items() if not m.get("reference_only")}
    groups = {g: [c for c in cols if c in Xr.columns] for g, cols in groups.items()}
    groups = {g: cols for g, cols in groups.items() if cols}

    p_base = bundle.predict_proba(Xr)
    base = {m: METRICS[m][0](y, p_base) for m in metrics}
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        drops: dict[str, list[float]] = {m: [] for m in metrics}
        for _ in range(n_repeats):
            perm = rng.permutation(len(Xr))
            Xp = Xr.copy()
            for c in cols:
                Xp[c] = Xr[c].to_numpy()[perm]
            p = bundle.predict_proba(Xp)
            for m in metrics:
                fn, hib = METRICS[m]
                drops[m].append(_degradation(base[m], fn(y, p), hib))
        row: dict[str, object] = {
            "feature": name,
            "columns": ", ".join(cols),
            "n_columns": len(cols),
        }
        for m in metrics:
            row[f"{m}_drop_mean"] = float(np.mean(drops[m]))
            row[f"{m}_drop_std"] = float(np.std(drops[m]))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(f"{metrics[0]}_drop_mean", ascending=False)
    out.attrs["baseline"] = base
    return out.reset_index(drop=True)
