"""Lens (filter) functions for Mapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

LENSES = {
    "Model PD (calibrated)": "Predicted adverse-outcome probability - maps the risk landscape.",
    "PCA 1": "First principal component of the standardised features (main axis of variation).",
    "k-NN density": (
        "Mean distance to the 15 nearest neighbours - separates dense cores from outliers."
    ),
    "PD x PCA 1 (2-D)": "Risk and main feature axis jointly (2-D cover).",
    "PCA 1 x PCA 2 (2-D)": "Two main axes of variation (2-D cover); risk shown only by colour.",
}


def compute_lens(name: str, X: np.ndarray, frame: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Return an (n, 1) or (n, 2) lens array for the named lens."""
    X = np.asarray(X, dtype="float64")
    if name == "Model PD (calibrated)":
        return frame["pd"].to_numpy(dtype="float64").reshape(-1, 1)
    if name in ("PCA 1", "PD x PCA 1 (2-D)", "PCA 1 x PCA 2 (2-D)"):
        pcs = PCA(n_components=2, random_state=seed).fit_transform(X)
        if name == "PCA 1":
            return pcs[:, :1]
        if name == "PCA 1 x PCA 2 (2-D)":
            return pcs
        return np.column_stack([frame["pd"].to_numpy(dtype="float64"), pcs[:, 0]])
    if name == "k-NN density":
        nn = NearestNeighbors(n_neighbors=16).fit(X)
        dist, _ = nn.kneighbors(X)
        return dist[:, 1:].mean(axis=1).reshape(-1, 1)
    raise ValueError(f"Unknown lens '{name}'. Options: {list(LENSES)}")
