"""Segment breakdown (Tier 2.3) and the fairness / friction check (Tier 2.4).

Two different questions that use the same machinery:

*Where is the model weak?* An average AUC-PR hides the segment where the ranking has
collapsed. `segment_metrics` splits the test set by customer tenure, region, order-value
band and product family, and `weakest_segment` names the worst one out loud.

*Where is the model unfair?* A scorer that adds checkout friction to one region far
above its actual return rate is imposing a cost on innocent customers. `fairness_table`
reports flag rate and precision per segment alongside the segment's own base rate, and
`disparity_report` measures how far each flag rate departs from what the observed return
rate would justify. Precision parity is the right target here, not flag-rate parity:
a segment that genuinely returns more *should* be flagged more, but it should not be
flagged with systematically worse precision -- that is friction paid for nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import evaluate
from .money import CostModel

MIN_SEGMENT_N = 200  # below this, per-segment metrics are noise, and we say so


def add_segment_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Derive the reporting segments. All are checkout-time attributes."""
    df = features.copy()
    df["seg_customer_type"] = np.where(df["is_new_customer"] == 1, "new", "returning")
    df["seg_region"] = df["region"].astype(str)
    df["seg_country"] = df["country"].astype(str)
    df["seg_category"] = df["top_category"].astype(str)
    df["seg_order_value"] = pd.cut(
        df["order_value_gbp"],
        bins=[0, 150, 300, 500, 1000, np.inf],
        labels=["<150", "150-300", "300-500", "500-1k", "1k+"],
    ).astype(str)
    df["seg_basket_size"] = pd.cut(
        df["n_lines"], bins=[0, 5, 15, 30, np.inf], labels=["1-5", "6-15", "16-30", "30+"]
    ).astype(str)
    return df


SEGMENT_COLUMNS = [
    "seg_customer_type",
    "seg_region",
    "seg_order_value",
    "seg_basket_size",
    "seg_category",
]


def segment_metrics(
    features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    t_star: float,
    action: str | None = None,
    columns: list[str] | None = None,
    top_k_levels: int = 6,
) -> pd.DataFrame:
    """Per-segment metric panel, with small segments retained but marked unreliable."""
    df = add_segment_columns(features)
    df["_p"] = proba
    value_inr = cost.to_inr(df["order_value_gbp"])
    df["_v"] = value_inr

    rows = []
    for col in columns or SEGMENT_COLUMNS:
        levels = df[col].value_counts()
        if col == "seg_category":  # dozens of families; report only the biggest
            levels = levels.head(top_k_levels)
        for level in levels.index:
            part = df[df[col] == level]
            y = part["returned"].to_numpy()
            p = part["_p"].to_numpy()
            if len(part) == 0:
                continue
            panel = evaluate(y, p, t_star)
            bd = cost.evaluate_policy(y, p >= t_star, part["_v"].to_numpy(), action, threshold=t_star)
            rows.append(
                {
                    "dimension": col.replace("seg_", ""),
                    "segment": str(level),
                    "n_orders": int(len(part)),
                    "n_returns": int(y.sum()),
                    "base_rate": panel.base_rate,
                    "auc_pr": panel.auc_pr,
                    "auc_pr_lift": panel.auc_pr_lift_vs_base,
                    "roc_auc": panel.roc_auc,
                    "precision": panel.precision,
                    "recall": panel.recall,
                    "brier": panel.brier,
                    "flag_rate": panel.flag_rate,
                    "savings_inr": bd.savings,
                    "reliable": bool(len(part) >= MIN_SEGMENT_N and 0 < y.sum() < len(y)),
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "n_orders"], ascending=[True, False], ignore_index=True)


def weakest_segment(table: pd.DataFrame, metric: str = "auc_pr_lift") -> pd.Series:
    """The worst *reliable* segment on lift over its own base rate.

    Lift, not raw AUC-PR: a segment with a 30% base rate will post a higher AUC-PR than
    one with 8% while actually being ranked worse. Lift over base rate is the comparison
    that survives differing prevalences.
    """
    reliable = table[table["reliable"]]
    if reliable.empty:
        return table.iloc[0] if len(table) else pd.Series(dtype="object")
    return reliable.loc[reliable[metric].idxmin()]


def weakness_sentence(table: pd.DataFrame) -> str:
    """One honest sentence naming where the model is worst."""
    w = weakest_segment(table)
    if w.empty:
        return "No segment had enough orders to assess reliably."
    return (
        f"Weakest reliable segment: {w['dimension']}={w['segment']} "
        f"(n={int(w['n_orders']):,}, base rate {w['base_rate']:.2%}, "
        f"AUC-PR {w['auc_pr']:.3f} = {w['auc_pr_lift']:.2f}x base, "
        f"precision {w['precision']:.3f} vs portfolio-wide behaviour)."
    )


def fairness_table(
    features: pd.DataFrame,
    proba: np.ndarray,
    t_star: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Flag rate and precision by segment, against the segment's own return rate."""
    df = add_segment_columns(features)
    df["_flag"] = np.asarray(proba) >= t_star

    overall_flag = float(df["_flag"].mean())
    rows = []
    for col in columns or ["seg_region", "seg_customer_type", "seg_order_value"]:
        for level, part in df.groupby(col, sort=False):
            y = part["returned"].to_numpy().astype(bool)
            f = part["_flag"].to_numpy()
            n_flag = int(f.sum())
            base = float(y.mean()) if len(y) else np.nan
            flag_rate = float(f.mean()) if len(f) else np.nan
            rows.append(
                {
                    "dimension": col.replace("seg_", ""),
                    "segment": str(level),
                    "n_orders": int(len(part)),
                    "base_rate": base,
                    "flag_rate": flag_rate,
                    "flag_rate_vs_overall": flag_rate / overall_flag if overall_flag else np.nan,
                    # >1 means we flag this segment harder than its return rate warrants.
                    "over_flagging_ratio": (flag_rate / base) if base else np.nan,
                    "precision": float((f & y).sum() / n_flag) if n_flag else np.nan,
                    "recall": float((f & y).sum() / max(int(y.sum()), 1)),
                    "false_positive_rate": float((f & ~y).sum() / max(int((~y).sum()), 1)),
                    "reliable": bool(len(part) >= MIN_SEGMENT_N),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["overall_flag_rate"] = overall_flag
    return out


def disparity_report(fair: pd.DataFrame) -> dict[str, object]:
    """Largest precision gap between reliable segments within each dimension."""
    findings: dict[str, object] = {}
    rel = fair[fair["reliable"] & fair["precision"].notna()]
    for dim, part in rel.groupby("dimension"):
        if len(part) < 2:
            continue
        best = part.loc[part["precision"].idxmax()]
        worst = part.loc[part["precision"].idxmin()]
        findings[str(dim)] = {
            "best_segment": str(best["segment"]),
            "best_precision": float(best["precision"]),
            "worst_segment": str(worst["segment"]),
            "worst_precision": float(worst["precision"]),
            "precision_gap": float(best["precision"] - worst["precision"]),
            "worst_flag_rate": float(worst["flag_rate"]),
            "worst_base_rate": float(worst["base_rate"]),
        }
    return findings


def disparity_sentence(fair: pd.DataFrame) -> str:
    """One sentence on the biggest friction disparity and what we would do about it."""
    rep = disparity_report(fair)
    if not rep:
        return "Not enough reliable segments to assess disparity."
    dim, worst = max(rep.items(), key=lambda kv: kv[1]["precision_gap"])
    w = worst
    return (
        f"Largest precision disparity is on {dim}: "
        f"{w['worst_segment']} is flagged at {w['worst_flag_rate']:.1%} but converts to only "
        f"{w['worst_precision']:.1%} precision, versus {w['best_precision']:.1%} for "
        f"{w['best_segment']} -- a {w['precision_gap']:.1%} gap, meaning "
        f"{w['worst_segment']} customers absorb more friction per prevented return."
    )


__all__ = [
    "SEGMENT_COLUMNS",
    "add_segment_columns",
    "disparity_report",
    "disparity_sentence",
    "fairness_table",
    "segment_metrics",
    "weakest_segment",
    "weakness_sentence",
]
