"""Capacity-constrained operating mode (Tier 2.5).

A threshold policy answers "should I act on this order?" but a warehouse team answers a
different question: "I can hand-check 200 parcels a shift -- which 200?" That is a
top-k problem, and the right metric is precision@k, not precision at t*.

The two modes disagree in a way worth stating: t* is what you use when capacity is
elastic and cost is the binding constraint (withholding a discount costs nothing to
scale). Top-k is what you use when a human has to touch the parcel. `recommend_mode`
makes that call explicitly rather than leaving it implied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .money import CostModel


def precision_at_k(
    features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    cfg: Config,
    action: str | None = None,
) -> pd.DataFrame:
    """Rank orders by score and report what the riskiest k% actually buys."""
    action = action or "manual_pack_check"  # the capacity-limited action by nature
    y = np.asarray(features["returned"]).astype(int)
    p = np.asarray(proba, dtype="float64")
    value_inr = cost.to_inr(features["order_value_gbp"])
    n = len(y)
    base = float(y.mean()) if n else 0.0

    order = np.argsort(-p, kind="stable")
    rows = []
    for k in [float(x) for x in cfg["capacity"]["k_percents"]]:
        take = max(int(round(k * n)), 1)
        idx = order[:take]
        flags = np.zeros(n, dtype=bool)
        flags[idx] = True

        caught = int(y[idx].sum())
        bd = cost.evaluate_policy(y, flags, value_inr, action, threshold=float(p[idx].min()))
        rows.append(
            {
                "k_pct": k,
                "orders_reviewed": take,
                "score_cutoff": float(p[idx].min()),
                "returns_caught": caught,
                "precision_at_k": caught / take,
                "recall_at_k": caught / max(int(y.sum()), 1),
                "lift_vs_base": (caught / take) / base if base else np.nan,
                "value_at_risk_inr": float(value_inr[idx][y[idx] == 1].sum()),
                "savings_inr": bd.savings,
                "savings_per_review_inr": bd.savings / take,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["action"] = action
    out.attrs["base_rate"] = base
    return out


def recommend_mode(topk: pd.DataFrame, t_star_flag_rate: float) -> str:
    """State which operating mode fits a capacity-limited merchant, and why."""
    if topk.empty:
        return "No capacity table available."
    best = topk.loc[topk["savings_per_review_inr"].idxmax()]
    return (
        f"At t* the policy flags {t_star_flag_rate:.1%} of orders, which is the right mode when "
        f"the action is cheap to scale (withholding a promo costs no labour). A merchant with a "
        f"fixed review desk should instead run top-k: reviewing the riskiest "
        f"{best['k_pct']:.0%} of orders yields precision@k {best['precision_at_k']:.1%} "
        f"({best['lift_vs_base']:.2f}x base rate) and the best return per unit of human effort "
        f"in this table."
    )


__all__ = ["precision_at_k", "recommend_mode"]
