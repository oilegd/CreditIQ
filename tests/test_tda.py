"""Topological data analysis: Mapper, lenses, persistence, point clouds and insights."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from creditiq.data.split import split_frames
from creditiq.features.build_features import resolve_feature_spec
from creditiq.tda.insights import generate_ideas
from creditiq.tda.lenses import LENSES, compute_lens
from creditiq.tda.mapper import (
    _cover_1d,
    build_mapper,
    edge_segments,
    knn_eps,
    layout,
    node_feature_profile,
    node_statistics,
)
from creditiq.tda.persistence import (
    diagram_frame,
    h0_diagram,
    persistence_diagrams,
    persistence_summary,
    ripser_available,
)
from creditiq.tda.point_cloud import build_point_cloud, feature_matrix, robust_scaler


def _circle(n=400, r=1.0, noise=0.02, seed=0, center=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, noise, (n, 2))
    return pts + np.asarray(center)


def test_cover_spans_range_with_overlap():
    cover = _cover_1d(np.linspace(0, 1, 100), 5, 0.3)
    assert len(cover) == 5
    assert cover[0][0] == pytest.approx(0) and cover[-1][1] == pytest.approx(1)
    for (a0, a1), (b0, _) in zip(cover, cover[1:], strict=False):
        assert b0 < a1  # consecutive intervals overlap
        assert (a1 - b0) / (a1 - a0) == pytest.approx(0.3)


def test_mapper_recovers_a_loop():
    X = _circle()
    g = build_mapper(
        X, X[:, :1], n_intervals=10, overlap=0.3, eps=0.25, min_cluster_size=3, balanced=False
    )
    G = g.to_networkx()
    assert nx.number_connected_components(G) == 1
    assert len(nx.cycle_basis(G)) >= 1  # the circle's hole appears as a cycle


def test_mapper_separates_components():
    X = np.vstack([_circle(center=(0, 0)), _circle(center=(10, 0), seed=1)])
    g = build_mapper(
        X, X[:, 1:2], n_intervals=6, overlap=0.3, eps=0.3, min_cluster_size=3, balanced=False
    )
    assert g.n_components == 2


def test_mapper_graph_invariants():
    X = _circle(300)
    g = build_mapper(X, X[:, :1], n_intervals=8, overlap=0.4, eps=0.3, min_cluster_size=5)
    assert (g.nodes["size"] >= 5).all()
    assert 0 < g.coverage <= 1
    for e in g.edges.itertuples():
        shared = np.intersect1d(g.members[e.source], g.members[e.target])
        assert len(shared) == e.shared > 0
    pos = layout(g)
    assert set(pos["node_id"]) == set(g.nodes["node_id"])
    seg = edge_segments(g.edges, pos)
    assert len(seg) == len(g.edges) and seg[["x", "y", "x2", "y2"]].notna().all().all()


def test_mapper_validates_parameters():
    X = _circle(50)
    with pytest.raises(ValueError):
        build_mapper(X, X[:, :1], overlap=1.0)
    with pytest.raises(ValueError):
        build_mapper(X, np.column_stack([X, X[:, :1]]))


def test_h0_diagram_and_persistence():
    X = np.vstack([_circle(60), _circle(60, center=(8, 0), seed=3)])
    d0 = h0_diagram(X)
    assert len(d0) == len(X) - 1 and (d0[:, 0] == 0).all()
    assert d0[:, 1].max() > 5  # the two separated groups merge late
    dg = diagram_frame(persistence_diagrams(X[:80], maxdim=0), "all")
    summ = persistence_summary(dg)
    assert set(summ["dimension"]) == {"H0"}


@pytest.mark.skipif(not ripser_available(), reason="ripser not installed")
def test_h1_detects_the_loop():
    dg = diagram_frame(persistence_diagrams(_circle(150, noise=0.03), maxdim=1), "circle")
    h1 = dg[dg["dimension"] == "H1"].sort_values("persistence", ascending=False)
    assert h1["persistence"].iloc[0] > 3 * h1["persistence"].iloc[1:].max()


def test_knn_eps_positive():
    assert knn_eps(_circle(200), k=5) > 0


@pytest.fixture(scope="module")
def cloud(trained, prepared, cfg):
    result, _ = trained
    spec = resolve_feature_spec(cfg.raw, "applicant")
    frames = split_frames(prepared.data)
    scaler = robust_scaler(frames["train"], spec)
    return build_point_cloud(
        frames["validation"], spec, scaler, result.bundles["lightgbm"], "is_adverse", 500
    )


def test_point_cloud_is_clean_and_train_scaled(cloud, prepared, cfg):
    X = feature_matrix(cloud)
    assert np.isfinite(X).all() and np.abs(X).max() <= 5
    assert cloud["pd"].between(0, 1).all() and set(cloud["y"].unique()) <= {0, 1}
    spec = resolve_feature_spec(cfg.raw, "applicant")
    s1 = robust_scaler(split_frames(prepared.data)["train"], spec)
    tampered = prepared.data.copy()
    tampered.loc[tampered["split"].astype(str) != "train", "dti"] = 999.0
    s2 = robust_scaler(split_frames(tampered)["train"], spec)
    pd.testing.assert_frame_equal(s1, s2)  # statistics come from train only


@pytest.mark.parametrize("lens", list(LENSES))
def test_lenses_and_node_statistics(cloud, lens):
    X = feature_matrix(cloud)
    lv = compute_lens(lens, X, cloud)
    assert lv.shape[0] == len(cloud) and lv.shape[1] in (1, 2)
    g = build_mapper(X, lv, n_intervals=5, overlap=0.4, min_cluster_size=5)
    feats = [c for c in cloud.columns if c.startswith("f__")]
    stats = node_statistics(g, cloud, feats)
    if len(stats):
        nid = int(stats["node_id"].iloc[0])
        members = g.members[nid]
        row = stats.set_index("node_id").loc[nid]
        assert row["observed_rate"] == pytest.approx(cloud["y"].to_numpy()[members].mean())
        assert row["calibration_gap"] == pytest.approx(
            cloud["pd"].to_numpy()[members].mean() - row["observed_rate"]
        )
        prof = node_feature_profile(g, cloud, feats, nid)
        assert len(prof) == len(feats) and not prof["feature"].str.startswith("f__").any()


def test_unknown_lens_raises(cloud):
    with pytest.raises(ValueError):
        compute_lens("nope", feature_matrix(cloud), cloud)


def test_ideas_require_evidence():
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "size": [400, 20, 300],
            "observed_rate": [0.10, 0.60, 0.45],
            "mean_pd": [0.20, 0.20, 0.30],
            "calibration_gap": [0.10, -0.40, -0.15],
            "gap_z": [5.3, -4.6, -1.0],
            "risk_lift": [0.55, 3.3, 2.4],
            "within_node_auc": [0.70, np.nan, 0.52],
            "profile": ["a", "b", "c"],
        }
    )
    ideas = generate_ideas(nodes, overall_rate=0.18, n_components=1)
    kinds = {(i["kind"], tuple(i["nodes"])) for i in ideas}
    assert ("Calibration", (0,)) in kinds  # large, significant gap
    assert all(1 not in i["nodes"] for i in ideas)  # too small to be evidence
    assert ("High-risk region", (2,)) in kinds
    assert ("Weak discrimination", (2,)) in kinds
    assert not any(i["kind"] == "Structure" for i in ideas)
