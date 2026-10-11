"""Persistent homology summaries of the loan point cloud.

* H0 (connected components) is computed exactly from single-linkage merge heights,
  which coincide with Vietoris-Rips H0 death times (no extra dependency).
* H1 (loops) uses ``ripser`` when installed (optional ``tda`` extra).

Points far from the diagonal are persistent (robust) topological features; points near
it are noise. Diagrams describe the geometry of the applicant feature space; they are a
descriptive diagnostic, not a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


def ripser_available() -> bool:
    try:
        import ripser  # noqa: F401
    except ImportError:
        return False
    return True


def h0_diagram(X: np.ndarray) -> np.ndarray:
    """(birth, death) pairs for H0; one infinite class omitted."""
    X = np.asarray(X, dtype="float64")
    if len(X) < 2:
        return np.empty((0, 2))
    deaths = np.sort(linkage(X, method="single")[:, 2])
    return np.column_stack([np.zeros_like(deaths), deaths])


def persistence_diagrams(X: np.ndarray, maxdim: int = 1) -> dict[int, np.ndarray]:
    """{dimension: (k, 2) array of (birth, death)}; H1 only if ripser is available."""
    out = {0: h0_diagram(X)}
    if maxdim >= 1 and ripser_available():
        from ripser import ripser

        dgms = ripser(np.asarray(X, dtype="float64"), maxdim=1)["dgms"]
        out[1] = dgms[1]
    return out


def diagram_frame(diagrams: dict[int, np.ndarray], label: str) -> pd.DataFrame:
    rows = []
    for dim, d in diagrams.items():
        finite = d[np.isfinite(d[:, 1])] if len(d) else d
        for b, e in finite:
            rows.append(
                {
                    "group": label,
                    "dimension": f"H{dim}",
                    "birth": b,
                    "death": e,
                    "persistence": e - b,
                }
            )
    return pd.DataFrame(rows, columns=["group", "dimension", "birth", "death", "persistence"])


def persistence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count, total and top persistence per group and homology dimension."""
    if frame.empty:
        return frame
    return (
        frame.groupby(["group", "dimension"])["persistence"]
        .agg(
            n_features="size",
            total_persistence="sum",
            max_persistence="max",
            median_persistence="median",
        )
        .reset_index()
    )
