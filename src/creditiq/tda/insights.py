"""Evidence-based improvement ideas derived from a Mapper graph.

Every generated idea cites the measured node statistics that triggered it. Rules are
deliberately simple and conservative (minimum node size and a binomial z-threshold) so
that sampling noise is not presented as a finding. Ideas are hypotheses to test on
fresh data, not validated improvements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NODE = 60
Z_THRESHOLD = 2.5


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def generate_ideas(nodes: pd.DataFrame, overall_rate: float, n_components: int) -> list[dict]:
    """Return a list of ideas: {kind, title, evidence, action, nodes}."""
    ideas: list[dict] = []
    big = nodes[nodes["size"] >= MIN_NODE]

    mis = big[big["gap_z"].abs() >= Z_THRESHOLD].sort_values("gap_z", key=np.abs, ascending=False)
    for _, r in mis.head(4).iterrows():
        direction = "over-predicts" if r["calibration_gap"] > 0 else "under-predicts"
        ideas.append(
            {
                "kind": "Calibration",
                "title": f"Model {direction} risk in node {int(r['node_id'])}",
                "evidence": (
                    f"n={int(r['size'])}, observed {_fmt_pct(r['observed_rate'])} vs predicted "
                    f"{_fmt_pct(r['mean_pd'])} (z={r['gap_z']:+.1f}); profile: {r['profile']}."
                ),
                "action": (
                    "Test a segment-aware recalibration or an interaction feature for this "
                    "profile; first confirm the gap on another period (it may reflect the "
                    "exclusion of unresolved 60-month loans if share_60m is high)."
                ),
                "nodes": [int(r["node_id"])],
            }
        )

    flares = big[big["risk_lift"] >= 1.6].sort_values("risk_lift", ascending=False)
    for _, r in flares.head(3).iterrows():
        ideas.append(
            {
                "kind": "High-risk region",
                "title": f"High-risk flare at node {int(r['node_id'])}",
                "evidence": (
                    f"observed adverse rate {_fmt_pct(r['observed_rate'])} = "
                    f"{r['risk_lift']:.1f}x the sample average ({_fmt_pct(overall_rate)}); "
                    f"n={int(r['size'])}; profile: {r['profile']}."
                ),
                "action": (
                    "Candidate early-warning / manual-review segment; check that the model's "
                    "ranking inside the region is adequate (within-node AUC) before using it "
                    "in a policy rule."
                ),
                "nodes": [int(r["node_id"])],
            }
        )

    weak = big[(big["size"] >= 150) & (big["within_node_auc"] < 0.56)].sort_values(
        "within_node_auc"
    )
    for _, r in weak.head(3).iterrows():
        ideas.append(
            {
                "kind": "Weak discrimination",
                "title": f"Model barely ranks loans inside node {int(r['node_id'])}",
                "evidence": (
                    f"within-node AUC {r['within_node_auc']:.3f} (n={int(r['size'])}, adverse "
                    f"{_fmt_pct(r['observed_rate'])}); profile: {r['profile']}."
                ),
                "action": (
                    "The available application fields carry little extra signal here; "
                    "consider additional data (e.g. bureau scores, payment history, income "
                    "verification detail) rather than more model complexity."
                ),
                "nodes": [int(r["node_id"])],
            }
        )

    if n_components > 1:
        ideas.append(
            {
                "kind": "Structure",
                "title": f"The loan space splits into {n_components} disconnected components",
                "evidence": (
                    "At the chosen scale, groups of loans share no members across components."
                ),
                "action": (
                    "Inspect whether components align with product features (e.g. term, purpose); "
                    "if their risk drivers differ, compare a segmented model against the "
                    "global one on validation data."
                ),
                "nodes": [],
            }
        )
    return ideas


STATIC_IDEAS = [
    (
        "Topological drift monitoring",
        "Build the Mapper graph per vintage and compare node composition and calibration gaps "
        "over time; new or vanishing branches signal population shift earlier than global PSI.",
    ),
    (
        "Mapper-guided segmentation",
        "Use stable Mapper regions as candidate segments; accept a segmented model only if it "
        "beats the global model on a later out-of-time period.",
    ),
    (
        "Topological features",
        "Add node-membership or distance-to-high-risk-region features learned on training data "
        "only, then re-validate; guard against leakage by fitting the cover on train.",
    ),
    (
        "Robustness of conclusions",
        "Treat a region as real only if it persists across lens choices and resolutions "
        "(n_intervals, overlap) - the persistence idea applied to Mapper itself.",
    ),
    (
        "Censoring-aware outcome",
        "Regions dominated by 60-month loans are most affected by excluding unresolved loans; "
        "a survival (time-to-default) target would remove this artefact.",
    ),
]
