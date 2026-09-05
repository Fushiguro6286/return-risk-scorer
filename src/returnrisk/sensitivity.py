"""Cost-sensitivity sweep (Tier 1.4).

Every rupee figure in this repo rests on two guesses: what a missed return really costs,
and what annoying a good customer really costs. Neither is knowable to better than a
factor of two. So instead of defending one number, we sweep both over a grid of
multipliers and report how t* and rupees-saved move.

What we are looking for is not stability of t* -- t* *should* move, that is the whole
point of a cost-driven threshold. What must hold is that **the model still saves money
across the whole grid**. If there is a corner of plausible costs where the recommended
policy loses money, that is a finding, and it belongs in the README rather than the bin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .money import CostModel


def cost_sensitivity(
    features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    cfg: Config,
    action: str | None = None,
) -> pd.DataFrame:
    """Grid over FN and FP cost multipliers; re-solve t* in each cell.

    The multipliers scale the *whole* cost family (logistics + restocking + lost margin
    for FN; ops + conversion loss for FP), which is the honest way to express "we might
    be wrong about this by 2x" without pretending we know which component is wrong.
    """
    action = action or cost.default_action
    y = features["returned"].to_numpy()
    value_inr = cost.to_inr(features["order_value_gbp"])

    fn_mults = [float(x) for x in cfg["sensitivity"]["fn_multipliers"]]
    fp_mults = [float(x) for x in cfg["sensitivity"]["fp_multipliers"]]

    original = (cost.fn_multiplier, cost.fp_multiplier)
    rows = []
    try:
        for fn_m in fn_mults:
            for fp_m in fp_mults:
                cost.fn_multiplier, cost.fp_multiplier = fn_m, fp_m
                t_star, bd, _ = cost.optimal_threshold(y, proba, value_inr, action)
                rows.append(
                    {
                        "action": action,
                        "fn_multiplier": fn_m,
                        "fp_multiplier": fp_m,
                        "fn_over_fp": fn_m / fp_m,
                        "t_star": t_star,
                        "orders_flagged": bd.n_flagged,
                        "flag_rate": bd.flag_rate,
                        "precision": bd.precision,
                        "recall": bd.recall,
                        "do_nothing_cost_inr": bd.do_nothing_cost,
                        "total_cost_inr": bd.total_cost,
                        "savings_inr": bd.savings,
                        "savings_pct": bd.savings / bd.do_nothing_cost if bd.do_nothing_cost else 0.0,
                        "model_profitable": bool(bd.savings > 0),
                    }
                )
    finally:
        cost.fn_multiplier, cost.fp_multiplier = original

    return pd.DataFrame(rows)


def sensitivity_summary(grid: pd.DataFrame) -> dict[str, float | int | bool]:
    """Headline read of the grid, for the README and summary.json."""
    return {
        "n_cells": int(len(grid)),
        "n_cells_profitable": int(grid["model_profitable"].sum()),
        "pct_cells_profitable": float(grid["model_profitable"].mean()),
        "t_star_min": float(grid["t_star"].min()),
        "t_star_max": float(grid["t_star"].max()),
        "t_star_median": float(grid["t_star"].median()),
        "savings_pct_min": float(grid["savings_pct"].min()),
        "savings_pct_max": float(grid["savings_pct"].max()),
        "worst_cell": grid.loc[grid["savings_pct"].idxmin(), ["fn_multiplier", "fp_multiplier"]]
        .to_dict(),
    }


def heatmap_pivot(grid: pd.DataFrame, value: str = "t_star") -> pd.DataFrame:
    """FN x FP matrix of one quantity, ready for imshow."""
    return grid.pivot(index="fn_multiplier", columns="fp_multiplier", values=value)


__all__ = ["cost_sensitivity", "heatmap_pivot", "sensitivity_summary"]
