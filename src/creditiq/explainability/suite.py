"""Run the full explainability suite for a feature set: SHAP, PFI and LIME (+ agreement).

All outputs are aggregate tables (no individual loan rows), computed on a random sample
of the out-of-time test split, for every non-dummy model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from creditiq.config import Config
from creditiq.data.split import split_frames
from creditiq.explainability.lime_tabular import global_lime_importance
from creditiq.explainability.permutation import permutation_importance
from creditiq.explainability.shap_analysis import (
    compute_shap,
    dependence_table,
    global_importance,
    shap_by_raw_input,
    signed_importance,
)
from creditiq.features.build_features import resolve_feature_spec
from creditiq.models.predict import available_models, load_metadata, load_model

logger = logging.getLogger(__name__)
EXPLAINED_MODELS = ("lightgbm", "logistic_regression")


def explainer_agreement(
    shap_raw: pd.DataFrame, pfi: pd.DataFrame, lime: pd.DataFrame
) -> pd.DataFrame:
    """Raw-input importance and rank under SHAP, PFI (ROC-AUC drop) and LIME."""
    df = (
        shap_raw.set_index("feature")[["mean_abs_shap"]]
        .join(pfi.set_index("feature")[["roc_auc_drop_mean"]], how="outer")
        .join(lime.set_index("feature")[["mean_abs_lime_weight"]], how="outer")
        .fillna(0.0)
    )
    for col, rank in (
        ("mean_abs_shap", "rank_shap"),
        ("roc_auc_drop_mean", "rank_pfi"),
        ("mean_abs_lime_weight", "rank_lime"),
    ):
        df[rank] = df[col].rank(ascending=False, method="min").astype(int)
    df["mean_rank"] = df[["rank_shap", "rank_pfi", "rank_lime"]].mean(axis=1)
    return df.sort_values("mean_rank").reset_index().rename(columns={"index": "feature"})


def run_explainability(
    cfg: Config, feature_set: str, processed: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return ``(tables, objects)``; tables are written to reports/metrics/<feature_set>/."""
    ecfg, xcfg = cfg["evaluation"], cfg["explainability"]
    target = cfg["target"]["name"]
    models_dir = cfg.path("models_dir")
    spec = resolve_feature_spec(cfg.raw, feature_set)
    selected = load_metadata(models_dir, feature_set)["selected_model"]
    test = split_frames(processed)["test"]
    rng = np.random.default_rng(cfg.seed)

    def _sample(n: int) -> pd.DataFrame:
        return test.iloc[rng.choice(len(test), size=min(n, len(test)), replace=False)]

    shap_rows = _sample(ecfg["shap_sample_size"])
    pfi_rows = _sample(ecfg["permutation_sample_size"])
    lime_rows = _sample(xcfg["lime_global_rows"])
    tables: dict[str, pd.DataFrame] = {}
    objects: dict[str, Any] = {}
    agreement = []
    for name in (m for m in EXPLAINED_MODELS if m in available_models(models_dir, feature_set)):
        bundle = load_model(models_dir, feature_set, name)
        logger.info("[%s] explaining %s", feature_set, name)
        res = compute_shap(bundle, shap_rows)
        objects[f"shap_{name}"] = res
        signed = signed_importance(res)
        tables[f"shap_signed_{name}"] = signed
        top = signed["feature"].head(10).tolist()
        tables[f"shap_dependence_{name}"] = dependence_table(res, top, xcfg["dependence_bins"])
        shap_raw = shap_by_raw_input(res, spec)
        tables[f"shap_raw_{name}"] = shap_raw
        if name == selected:
            tables["shap_global_importance"] = global_importance(res)

        y = pfi_rows[target].to_numpy()
        single = permutation_importance(
            bundle, pfi_rows, y, n_repeats=ecfg["permutation_repeats"], seed=cfg.seed
        )
        grouped = permutation_importance(
            bundle,
            pfi_rows,
            y,
            groups=xcfg["pfi_groups"],
            n_repeats=ecfg["permutation_repeats"],
            seed=cfg.seed,
        )
        tables[f"pfi_single_{name}"] = single
        tables[f"pfi_grouped_{name}"] = grouped

        lime = global_lime_importance(
            bundle, lime_rows, n_samples=max(500, xcfg["lime_samples"] // 2), seed=cfg.seed
        )
        tables[f"lime_global_{name}"] = lime.assign(mean_local_r2=lime.attrs["mean_r2"])
        agreement.append(explainer_agreement(shap_raw, single, lime).assign(model=name))
    if agreement:
        tables["explainer_agreement"] = pd.concat(agreement, ignore_index=True)
    return tables, objects
