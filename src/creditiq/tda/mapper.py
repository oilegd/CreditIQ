"""Mapper: a topological summary (graph) of a point cloud (Singh, Memoli & Carlsson, 2007).

Algorithm
1. A *lens* f: X -> R^k (k = 1 or 2) projects each loan to a low-dimensional value
   (e.g. model PD, first principal component, local density).
2. The lens range is covered by overlapping intervals (or rectangles for k = 2).
3. Inside each cover element, points are clustered in the *full* feature space
   (single linkage, cut at a global distance ``eps``).
4. Each cluster becomes a node; nodes sharing loans are joined by an edge (the nerve).

Nodes therefore group loans that are similar in feature space *and* in lens value;
branches ("flares") and separate components reveal sub-populations a single global
model may treat poorly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class MapperGraph:
    nodes: pd.DataFrame  # node_id, size, cube, lens_mean
    edges: pd.DataFrame  # source, target, shared
    members: dict[int, np.ndarray] = field(repr=False)
    n_points: int = 0
    params: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Share of points that belong to at least one node."""
        if not self.members:
            return 0.0
        covered = np.unique(np.concatenate(list(self.members.values())))
        return len(covered) / self.n_points if self.n_points else 0.0

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(self.nodes["node_id"].tolist())
        g.add_weighted_edges_from(
            self.edges[["source", "target", "shared"]].itertuples(index=False, name=None)
        )
        return g

    @property
    def n_components(self) -> int:
        return nx.number_connected_components(self.to_networkx()) if len(self.nodes) else 0


def knn_eps(X: np.ndarray, k: int = 15, quantile: float = 0.5, seed: int = 0) -> float:
    """Global clustering scale: a quantile of the distance to the k-th nearest neighbour."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, 3000), replace=False)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    dist, _ = nn.kneighbors(X[idx])
    return float(np.quantile(dist[:, -1], quantile))


def _cover_1d(values: np.ndarray, n_intervals: int, overlap: float) -> list[tuple[float, float]]:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return [(lo, hi)]
    width = (hi - lo) / (n_intervals - (n_intervals - 1) * overlap)
    step = width * (1 - overlap)
    return [(lo + i * step, lo + i * step + width) for i in range(n_intervals)]


def build_mapper(
    X: np.ndarray,
    lens: np.ndarray,
    n_intervals: int = 12,
    overlap: float = 0.45,
    eps: float | None = None,
    min_cluster_size: int = 8,
    balanced: bool = True,
    seed: int = 0,
) -> MapperGraph:
    """Build a Mapper graph.

    ``balanced=True`` rank-transforms each lens dimension first so every interval holds a
    similar number of loans (robust to skewed lenses such as PD).
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    X = np.asarray(X, dtype="float64")
    lens = np.asarray(lens, dtype="float64").reshape(len(X), -1)
    if lens.shape[1] > 2:
        raise ValueError("lens must be 1- or 2-dimensional")
    work = np.column_stack(
        [rankdata(lens[:, d]) / len(lens) if balanced else lens[:, d] for d in range(lens.shape[1])]
    )
    eps = eps if eps is not None else knn_eps(X, seed=seed)
    covers = [_cover_1d(work[:, d], n_intervals, overlap) for d in range(work.shape[1])]

    nodes, members = [], {}
    for cube in itertools.product(*[range(len(c)) for c in covers]):
        mask = np.ones(len(X), dtype=bool)
        for d, i in enumerate(cube):
            lo, hi = covers[d][i]
            mask &= (work[:, d] >= lo) & (work[:, d] <= hi)
        idx = np.flatnonzero(mask)
        if len(idx) < min_cluster_size:
            continue
        labels = fcluster(linkage(X[idx], method="single"), t=eps, criterion="distance")
        for lab in np.unique(labels):
            pts = idx[labels == lab]
            if len(pts) < min_cluster_size:
                continue
            node_id = len(nodes)
            members[node_id] = pts
            nodes.append(
                {
                    "node_id": node_id,
                    "size": len(pts),
                    "cube": "-".join(map(str, cube)),
                    "lens_mean": float(lens[pts, 0].mean()),
                }
            )

    point_nodes: dict[int, list[int]] = {}
    for nid, pts in members.items():
        for p in pts:
            point_nodes.setdefault(int(p), []).append(nid)
    shared: dict[tuple[int, int], int] = {}
    for nids in point_nodes.values():
        for a, b in itertools.combinations(sorted(nids), 2):
            shared[(a, b)] = shared.get((a, b), 0) + 1
    edges = pd.DataFrame(
        [{"source": a, "target": b, "shared": s} for (a, b), s in shared.items()],
        columns=["source", "target", "shared"],
    )
    return MapperGraph(
        nodes=pd.DataFrame(nodes, columns=["node_id", "size", "cube", "lens_mean"]),
        edges=edges,
        members=members,
        n_points=len(X),
        params={
            "n_intervals": n_intervals,
            "overlap": overlap,
            "eps": eps,
            "min_cluster_size": min_cluster_size,
            "balanced": balanced,
            "lens_dims": lens.shape[1],
        },
    )


def node_statistics(graph: MapperGraph, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Aggregate statistics per node: outcomes, predictions, calibration and profile.

    ``frame`` must contain ``y`` (observed outcome), ``pd`` (calibrated prediction) and the
    standardised ``features`` used to build the point cloud (same row order as ``X``).
    """
    y = frame["y"].to_numpy()
    p = frame["pd"].to_numpy()
    overall_rate = float(y.mean())
    F = frame[features].to_numpy(dtype="float64")
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    rows = []
    for nid, pts in graph.members.items():
        yn, pn = y[pts], p[pts]
        n = len(pts)
        z = (F[pts].mean(axis=0) - mu) / sd
        order = np.argsort(-np.abs(z))[:3]
        se = np.sqrt(max(overall_rate * (1 - overall_rate), 1e-9) / n)
        row = {
            "node_id": nid,
            "observed_rate": float(yn.mean()),
            "mean_pd": float(pn.mean()),
            "calibration_gap": float(pn.mean() - yn.mean()),
            "gap_z": float((pn.mean() - yn.mean()) / se),
            "brier": float(np.mean((pn - yn) ** 2)),
            "risk_lift": float(yn.mean() / overall_rate) if overall_rate > 0 else np.nan,
            "within_node_auc": float(roc_auc_score(yn, pn))
            if n >= 50 and 0 < yn.sum() < n
            else np.nan,
            "profile": ", ".join(
                f"{'high' if z[j] > 0 else 'low'} {features[j].removeprefix('f__')} "
                f"({z[j]:+.1f} sd)"
                for j in order
            ),
        }
        for col in ("share_60m", "annual_inc", "dti", "revol_util", "loan_amnt"):
            if col in frame.columns:
                key = col if col == "share_60m" else f"mean_{col}"
                row[key] = float(frame[col].to_numpy(dtype="float64")[pts].mean())
        for col in ("purpose", "home_ownership"):
            if col in frame.columns:
                row[f"top_{col}"] = str(frame[col].iloc[pts].astype(str).mode().iloc[0])
        rows.append(row)
    stats = pd.DataFrame(rows)
    return graph.nodes.merge(stats, on="node_id", how="left")


def node_feature_profile(
    graph: MapperGraph, frame: pd.DataFrame, features: list[str], node_id: int
) -> pd.DataFrame:
    """Standardised mean difference of every feature between one node and all loans."""
    F = frame[features].to_numpy(dtype="float64")
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    z = (F[graph.members[node_id]].mean(axis=0) - mu) / sd
    return (
        pd.DataFrame(
            {"feature": [f.removeprefix("f__") for f in features], "std_mean_difference": z}
        )
        .reindex(np.argsort(-np.abs(z)))
        .reset_index(drop=True)
    )


def layout(graph: MapperGraph, seed: int = 0, max_kamada_kawai: int = 400) -> pd.DataFrame:
    """2-D node positions.

    Kamada-Kawai (graph-distance preserving, good separation) for up to
    ``max_kamada_kawai`` nodes; an unweighted force-directed layout for larger graphs.
    """
    g = graph.to_networkx()
    n = g.number_of_nodes()
    if n == 0:
        return pd.DataFrame(columns=["node_id", "x", "y"])
    if n <= max_kamada_kawai:
        pos = nx.kamada_kawai_layout(g, weight=None)
    else:
        pos = nx.spring_layout(g, seed=seed, weight=None, k=2.0 / np.sqrt(n), iterations=300)
    return pd.DataFrame(
        [{"node_id": k, "x": float(v[0]), "y": float(v[1])} for k, v in pos.items()]
    )


def edge_segments(edges: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Edge list with endpoint coordinates for plotting."""
    pos = positions.set_index("node_id")
    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "shared", "x", "y", "x2", "y2"])
    out = edges.copy()
    out["x"] = pos.loc[out["source"], "x"].to_numpy()
    out["y"] = pos.loc[out["source"], "y"].to_numpy()
    out["x2"] = pos.loc[out["target"], "x"].to_numpy()
    out["y2"] = pos.loc[out["target"], "y"].to_numpy()
    return out
