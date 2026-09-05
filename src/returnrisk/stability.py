"""Temporal backtest (Tier 2.1).

A single test-set AUC-PR is one sample from a distribution the merchant will keep
drawing from every month. What matters operationally is whether the number holds up, or
quietly rots as assortment, promotions and customer mix drift.

So we re-score consecutive monthly slices of the *same held-out test period* -- no
refitting, no peeking -- and report AUC-PR, precision, recall and rupees saved per
month, plus a simple linear trend. A negative slope is not automatically failure; it is
information about how often the thing needs retraining, which is what an on-call
engineer actually wants to know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .metrics import evaluate
from .money import CostModel


def monthly_backtest(
    features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    cfg: Config,
    t_star: float,
    action: str | None = None,
) -> pd.DataFrame:
    """Per-period metrics across the test window, holding the model and t* fixed."""
    action = action or cost.default_action
    freq = str(cfg["stability"]["freq"])

    df = features.copy()
    df["_p"] = proba
    df["_period"] = df["order_date"].dt.to_period(freq)
    value_all = cost.to_inr(df["order_value_gbp"])
    df["_value_inr"] = value_all

    rows = []
    for period, part in df.groupby("_period", sort=True):
        y = part["returned"].to_numpy()
        p = part["_p"].to_numpy()
        v = part["_value_inr"].to_numpy()
        # A slice with no positives (or all positives) has no defined AUC-PR; we keep
        # the row so the gap is visible rather than silently dropped.
        panel = evaluate(y, p, t_star)
        bd = cost.evaluate_policy(y, p >= t_star, v, action, threshold=t_star)
        rows.append(
            {
                "period": str(period),
                "n_orders": int(len(part)),
                "n_returns": int(y.sum()),
                "base_rate": float(y.mean()) if len(y) else np.nan,
                "auc_pr": panel.auc_pr,
                "roc_auc": panel.roc_auc,
                "precision": panel.precision,
                "recall": panel.recall,
                "brier": panel.brier,
                "orders_flagged": bd.n_flagged,
                "flag_rate": bd.flag_rate,
                "savings_inr": bd.savings,
                "savings_pct": bd.savings / bd.do_nothing_cost if bd.do_nothing_cost else np.nan,
            }
        )
    return pd.DataFrame(rows)


def trend(backtest: pd.DataFrame, column: str = "auc_pr") -> dict[str, float]:
    """Least-squares slope of a metric over the test months, plus first/last values."""
    sub = backtest.dropna(subset=[column])
    if len(sub) < 2:
        return {"slope_per_month": float("nan"), "first": float("nan"), "last": float("nan")}
    x = np.arange(len(sub), dtype="float64")
    y = sub[column].to_numpy(dtype="float64")
    slope, _ = np.polyfit(x, y, 1)
    return {
        "slope_per_month": float(slope),
        "first": float(y[0]),
        "last": float(y[-1]),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
        "relative_change": float((y[-1] - y[0]) / y[0]) if y[0] else float("nan"),
    }


def stability_verdict(backtest: pd.DataFrame) -> str:
    """One sentence a human can paste into a README, stating decay or the absence of it."""
    t = trend(backtest, "auc_pr")
    if np.isnan(t["slope_per_month"]):
        return "Too few test months to assess stability."
    direction = "declines" if t["slope_per_month"] < 0 else "improves"
    return (
        f"Across {len(backtest)} monthly test slices AUC-PR {direction} by "
        f"{abs(t['slope_per_month']):.4f}/month "
        f"(first {t['first']:.3f}, last {t['last']:.3f}, "
        f"range {t['min']:.3f}-{t['max']:.3f}, sd {t['std']:.3f})."
    )


__all__ = ["monthly_backtest", "stability_verdict", "trend"]
