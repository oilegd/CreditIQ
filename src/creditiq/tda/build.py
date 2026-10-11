"""Build and persist TDA point clouds (local, git-ignored) and a default Mapper summary."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from creditiq.config import Config
from creditiq.data.split import split_frames
from creditiq.features.build_features import resolve_feature_spec
from creditiq.models.predict import load_model
from creditiq.reporting.results import tda_cloud_path
from creditiq.tda.insights import generate_ideas
from creditiq.tda.lenses import compute_lens
from creditiq.tda.mapper import build_mapper, knn_eps, node_statistics
from creditiq.tda.point_cloud import (
    build_point_cloud,
    feature_columns,
    feature_matrix,
    robust_scaler,
)

logger = logging.getLogger(__name__)
DEFAULT_LENS = "Model PD (calibrated)"


def build_tda_clouds(cfg: Config, feature_set: str, processed: pd.DataFrame) -> list[Path]:
    """Sample each configured split, standardise with TRAIN statistics, attach predictions."""
    tcfg = cfg["tda"]
    spec = resolve_feature_spec(cfg.raw, "applicant")  # geometry: applicant-time features
    frames = split_frames(processed)
    scaler = robust_scaler(frames["train"], spec)
    bundle = load_model(cfg.path("models_dir"), feature_set)
    paths = []
    for split in tcfg["splits"]:
        cloud = build_point_cloud(
            frames[split],
            spec,
            scaler,
            bundle,
            cfg["target"]["name"],
            tcfg["sample_size"],
            cfg.seed,
        )
        path = tda_cloud_path(cfg.path("processed_dir"), feature_set, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        cloud.to_parquet(path, index=False)
        logger.info("TDA point cloud %s (%s x %s)", path, *cloud.shape)
        paths.append(path)
    return paths


def default_mapper_summary(cfg: Config, feature_set: str) -> dict[str, pd.DataFrame | list]:
    """Mapper graph with config defaults on the default split (for reports)."""
    tcfg = cfg["tda"]
    m = tcfg["mapper"]
    cloud = pd.read_parquet(
        tda_cloud_path(cfg.path("processed_dir"), feature_set, tcfg["default_split"])
    )
    X = feature_matrix(cloud)
    eps = knn_eps(X, k=m["knn_k"], quantile=m["eps_quantile"], seed=cfg.seed)
    graph = build_mapper(
        X,
        compute_lens(DEFAULT_LENS, X, cloud, seed=cfg.seed),
        n_intervals=m["n_intervals"],
        overlap=m["overlap"],
        eps=eps,
        min_cluster_size=m["min_cluster_size"],
    )
    nodes = node_statistics(graph, cloud, feature_columns(cloud))
    ideas = generate_ideas(nodes, float(cloud["y"].mean()), graph.n_components)
    return {"graph": graph, "nodes": nodes, "ideas": ideas, "cloud": cloud}
