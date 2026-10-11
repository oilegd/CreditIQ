"""Hypothetical risk-based decision simulation.

IMPORTANT: LGD, EAD and profit parameters are user-supplied *hypothetical*
assumptions, not estimates. The observed label (charged off over the loan's life)
is not a validated regulatory default definition. This is an educational
simulation, not an automated credit-approval system.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionAssumptions:
    lgd: float = 0.85
    ead_fraction: float = 1.0
    profit_rate_good_loan: float = 0.12

    def __post_init__(self) -> None:
        if not 0 <= self.lgd <= 1:
            raise ValueError("lgd must be within [0, 1]")
        if not 0 < self.ead_fraction <= 1.5:
            raise ValueError("ead_fraction must be within (0, 1.5]")
        if self.profit_rate_good_loan < 0:
            raise ValueError("profit_rate_good_loan must be non-negative")

    @classmethod
    def from_config(cls, decision_cfg: Mapping[str, Any]) -> DecisionAssumptions:
        return cls(
            lgd=float(decision_cfg["lgd"]),
            ead_fraction=float(decision_cfg["ead_fraction"]),
            profit_rate_good_loan=float(decision_cfg["profit_rate_good_loan"]),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def assign_risk_bucket(p: np.ndarray, buckets: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Map probabilities to the first bucket whose exclusive ``upper`` bound exceeds them."""
    p = np.asarray(p, dtype="float64")
    uppers = np.array([float(b["upper"]) for b in buckets])
    if np.any(np.diff(uppers) <= 0):
        raise ValueError("Risk bucket upper bounds must be strictly increasing")
    names = np.array([str(b["name"]) for b in buckets], dtype=object)
    idx = np.searchsorted(uppers, p, side="right")
    if np.any(idx >= len(buckets)):
        raise ValueError("Probability exceeds the highest risk-bucket bound")
    return names[idx]


def expected_loss(pd_: np.ndarray, exposure: np.ndarray, lgd: float) -> np.ndarray:
    """Expected Loss = PD x LGD x EAD (element-wise)."""
    return np.asarray(pd_, dtype="float64") * float(lgd) * np.asarray(exposure, dtype="float64")


def breakeven_pd(assumptions: DecisionAssumptions) -> float:
    """PD at which approving has zero expected value: (1-p)*margin = p*LGD (per unit EAD)."""
    margin = assumptions.profit_rate_good_loan
    loss = assumptions.lgd * assumptions.ead_fraction
    return margin / (margin + loss) if (margin + loss) > 0 else 0.0


def simulate_threshold(
    y: np.ndarray,
    p: np.ndarray,
    loan_amnt: np.ndarray,
    threshold: float,
    assumptions: DecisionAssumptions,
) -> dict[str, float]:
    """Approve when ``p < threshold``; compute portfolio outcomes on historical labels."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    amount = np.asarray(loan_amnt, dtype="float64")
    ead = amount * assumptions.ead_fraction
    approve = p < threshold
    bad, good = y == 1, y == 0
    loss_if_bad = assumptions.lgd * ead
    profit_if_good = assumptions.profit_rate_good_loan * amount

    false_approval_cost = float(loss_if_bad[approve & bad].sum())
    false_rejection_cost = float(profit_if_good[~approve & good].sum())
    profit_from_goods = float(profit_if_good[approve & good].sum())
    n_approved = int(approve.sum())
    return {
        "threshold": float(threshold),
        "approval_rate": float(approve.mean()),
        "n_approved": n_approved,
        "approved_volume": float(amount[approve].sum()),
        "observed_bad_rate_approved": float(y[approve].mean()) if n_approved else float("nan"),
        "observed_bad_rate_rejected": float(y[~approve].mean())
        if (~approve).any()
        else float("nan"),
        "expected_loss_approved": float(
            expected_loss(p[approve], ead[approve], assumptions.lgd).sum()
        ),
        "false_approval_cost": false_approval_cost,
        "false_rejection_opportunity_cost": false_rejection_cost,
        "profit_from_good_approvals": profit_from_goods,
        "net_value": profit_from_goods - false_approval_cost,
    }


def threshold_scenarios(
    y: np.ndarray,
    p: np.ndarray,
    loan_amnt: np.ndarray,
    thresholds: Sequence[float],
    assumptions: DecisionAssumptions,
) -> pd.DataFrame:
    """Compare cut-offs, including an 'approve everyone' reference row."""
    rows = [
        simulate_threshold(y, p, loan_amnt, 1.0 + 1e-9, assumptions) | {"scenario": "approve all"}
    ]
    for t in thresholds:
        rows.append(
            simulate_threshold(y, p, loan_amnt, t, assumptions) | {"scenario": f"PD < {t:.3f}"}
        )
    df = pd.DataFrame(rows)
    return df[["scenario", *[c for c in df.columns if c != "scenario"]]]


def sensitivity_analysis(
    y: np.ndarray,
    p: np.ndarray,
    loan_amnt: np.ndarray,
    thresholds: Sequence[float],
    lgd_values: Sequence[float],
    profit_values: Sequence[float],
    ead_fraction: float = 1.0,
) -> pd.DataFrame:
    """For each (LGD, margin) pair: the value-maximising cut-off among ``thresholds``."""
    rows = []
    for lgd in lgd_values:
        for margin in profit_values:
            a = DecisionAssumptions(
                lgd=lgd, ead_fraction=ead_fraction, profit_rate_good_loan=margin
            )
            scen = threshold_scenarios(y, p, loan_amnt, thresholds, a)
            best = scen.loc[scen["net_value"].idxmax()]
            rows.append(
                {
                    "lgd": lgd,
                    "profit_rate_good_loan": margin,
                    "breakeven_pd": breakeven_pd(a),
                    "best_scenario": best["scenario"],
                    "best_threshold": best["threshold"],
                    "best_approval_rate": best["approval_rate"],
                    "best_net_value": best["net_value"],
                    "approve_all_net_value": float(scen.iloc[0]["net_value"]),
                }
            )
    return pd.DataFrame(rows)


def bucket_summary(
    y: np.ndarray,
    p: np.ndarray,
    loan_amnt: np.ndarray,
    buckets: Sequence[Mapping[str, Any]],
    assumptions: DecisionAssumptions,
) -> pd.DataFrame:
    """Per risk bucket: volume, mean predicted PD, observed adverse rate and expected loss."""
    amount = np.asarray(loan_amnt, dtype="float64")
    df = pd.DataFrame(
        {
            "bucket": assign_risk_bucket(p, buckets),
            "y": np.asarray(y).astype(int),
            "p": np.asarray(p, dtype="float64"),
            "amount": amount,
            "el": expected_loss(p, amount * assumptions.ead_fraction, assumptions.lgd),
        }
    )
    order = [str(b["name"]) for b in buckets]
    out = (
        df.groupby("bucket")
        .agg(
            n=("y", "size"),
            mean_predicted_pd=("p", "mean"),
            observed_adverse_rate=("y", "mean"),
            volume=("amount", "sum"),
            expected_loss=("el", "sum"),
        )
        .reindex(order)
        .dropna(subset=["n"])
    )
    out["share"] = out["n"] / out["n"].sum()
    out["expected_loss_rate"] = out["expected_loss"] / out["volume"]
    return out.reset_index()
