"""Build the order-level table and attach the return label.

THE LABEL, IN ONE SENTENCE
--------------------------
`returned = 1` if at least one line of the order is later reversed by a cancellation
invoice from the same customer for the same stock code, within `label.lookahead_days`
(90) of the order.

Why this heuristic
------------------
Online Retail II has no returns column. What it has are *cancellation invoices*:
rows whose invoice number starts with "C" and whose quantity is negative. These are
the merchant's own record of goods coming back (or an order being reversed). Linking
one to its originating purchase requires a heuristic because the file carries no
foreign key, so we match on (customer, stock code) and take the most recent prior
purchase that still has unreturned quantity left. Quantity is consumed greedily so a
single cancellation cannot mark two different orders as returned.

Two consequences we report rather than hide:
  * Cancellations that match nothing (wrong customer, product never bought by them,
    outside the window) are dropped. The match rate is printed and recorded.
  * Orders in the last `lookahead_days` of the file have an unobservable outcome --
    their 90-day window runs past the end of the data. They are dropped globally,
    because keeping them would silently label them 0 and deflate the base rate.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class LabelReport:
    """Diagnostics for the linking heuristic, surfaced in the README and model card."""

    n_cancellation_lines: int = 0
    n_matched_lines: int = 0
    n_orders_before_censor: int = 0
    n_orders_after_censor: int = 0
    n_orders_returned: int = 0
    lookahead_days: int = 0
    censor_cutoff: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        if self.n_cancellation_lines == 0:
            return 0.0
        return self.n_matched_lines / self.n_cancellation_lines

    @property
    def base_rate(self) -> float:
        if self.n_orders_after_censor == 0:
            return 0.0
        return self.n_orders_returned / self.n_orders_after_censor

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_definition": (
                "returned=1 if a later cancellation invoice from the same customer reverses "
                "at least one stock code of the order within "
                f"{self.lookahead_days} days"
            ),
            "n_cancellation_lines": self.n_cancellation_lines,
            "n_cancellation_lines_matched": self.n_matched_lines,
            "cancellation_match_rate": round(self.match_rate, 4),
            "n_orders_before_right_censor_drop": self.n_orders_before_censor,
            "n_orders_analysed": self.n_orders_after_censor,
            "n_orders_returned": self.n_orders_returned,
            "base_rate": round(self.base_rate, 6),
            "right_censor_cutoff": str(self.censor_cutoff),
            **self.extra,
        }


def derive_category(stock_code: pd.Series) -> pd.Series:
    """Coarse product family from the stock-code prefix.

    Online Retail II ships no category column. Its stock codes are structured: the
    leading digits group genuinely related giftware (e.g. 200xx are largely bags,
    228xx largely tins). Taking the first three characters gives ~80 usable families
    -- coarse, but honest, and it is a checkout-time attribute.
    """
    return stock_code.astype("string").str.replace(r"[^0-9A-Z]", "", regex=True).str[:3]


def build_orders(txn: pd.DataFrame) -> pd.DataFrame:
    """Collapse purchase lines to one row per invoice (the unit of prediction)."""
    buys = txn[~txn["is_cancellation"]].copy()
    # A real category column beats the stock-code prefix heuristic whenever the source
    # has one. The heuristic is tuned to Online Retail II's numeric codes; on an export
    # whose SKUs share a literal prefix ("SKU-1234") it collapses every product into a
    # single family and silently deletes all product-level signal.
    if "category" in buys.columns and buys["category"].notna().any():
        buys["category"] = buys["category"].astype("string").fillna("OTHER")
    else:
        buys["category"] = derive_category(buys["stock_code"])

    grp = buys.groupby("invoice", sort=False)
    orders = grp.agg(
        customer_id=("customer_id", "first"),
        order_date=("invoice_date", "min"),
        country=("country", "first"),
        n_lines=("stock_code", "nunique"),
        total_quantity=("quantity", "sum"),
        order_value_gbp=("line_value", "sum"),
        max_line_value_gbp=("line_value", "max"),
        avg_unit_price=("unit_price", "mean"),
        max_unit_price=("unit_price", "max"),
        min_unit_price=("unit_price", "min"),
    ).reset_index(names="order_id")

    # The dominant line defines the order's product identity for category features.
    top = buys.loc[buys.groupby("invoice", sort=False)["line_value"].idxmax()]
    top = top[["invoice", "stock_code", "category", "description"]].rename(
        columns={
            "invoice": "order_id",
            "stock_code": "top_stock_code",
            "category": "top_category",
            "description": "top_description",
        }
    )
    orders = orders.merge(top, on="order_id", how="left")
    return orders.sort_values(["order_date", "order_id"], ignore_index=True)


def link_cancellations(
    txn: pd.DataFrame, orders: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, LabelReport]:
    """Attach `returned` / `return_event_date` by matching cancellations to purchases.

    Greedy most-recent-first matching with quantity consumption, so one returned unit
    is attributed to exactly one order.
    """
    lookahead = pd.Timedelta(days=int(cfg["label"]["lookahead_days"]))
    report = LabelReport(lookahead_days=int(cfg["label"]["lookahead_days"]))

    buys = txn[~txn["is_cancellation"]]
    cancels = txn[txn["is_cancellation"]].sort_values("invoice_date")
    report.n_cancellation_lines = int(len(cancels))

    # Index purchase lines by (customer, stock code), date-ascending, for bisect lookup.
    index: dict[tuple[str, str], dict[str, list[Any]]] = {}
    buys_sorted = buys.sort_values("invoice_date")
    for cust, code, date, qty, inv in zip(
        buys_sorted["customer_id"].to_numpy(),
        buys_sorted["stock_code"].to_numpy(),
        buys_sorted["invoice_date"].to_numpy(),
        buys_sorted["quantity"].to_numpy(),
        buys_sorted["invoice"].to_numpy(),
        strict=True,
    ):
        slot = index.setdefault((cust, code), {"dates": [], "qty": [], "inv": []})
        slot["dates"].append(date)
        slot["qty"].append(float(qty))
        slot["inv"].append(inv)

    returned_at: dict[str, Any] = {}
    returned_units: dict[str, float] = {}
    matched_lines = 0

    for cust, code, cdate, cqty in zip(
        cancels["customer_id"].to_numpy(),
        cancels["stock_code"].to_numpy(),
        cancels["invoice_date"].to_numpy(),
        cancels["quantity"].to_numpy(),
        strict=True,
    ):
        slot = index.get((cust, code))
        if slot is None:
            continue
        remaining = abs(float(cqty))
        # Candidate purchases are those at or before the cancellation date.
        hi = bisect.bisect_right(slot["dates"], cdate)
        earliest = cdate - lookahead.to_timedelta64()
        any_match = False
        for j in range(hi - 1, -1, -1):
            if slot["dates"][j] < earliest:
                break
            avail = slot["qty"][j]
            if avail <= 0:
                continue
            take = min(avail, remaining)
            slot["qty"][j] = avail - take
            remaining -= take
            inv = slot["inv"][j]
            returned_units[inv] = returned_units.get(inv, 0.0) + take
            prev = returned_at.get(inv)
            if prev is None or cdate < prev:
                returned_at[inv] = cdate
            any_match = True
            if remaining <= 1e-9:
                break
        if any_match:
            matched_lines += 1

    report.n_matched_lines = matched_lines

    orders = orders.copy()
    orders["returned"] = orders["order_id"].map(lambda i: 1 if i in returned_at else 0).astype("int8")
    orders["return_event_date"] = pd.to_datetime(orders["order_id"].map(returned_at))
    orders["returned_units"] = orders["order_id"].map(returned_units).fillna(0.0)

    # ---- Right-censoring: drop orders whose outcome window runs past the data end.
    report.n_orders_before_censor = int(len(orders))
    data_end = pd.to_datetime(txn["invoice_date"]).max()
    cutoff = data_end - lookahead
    report.censor_cutoff = cutoff
    orders = orders[orders["order_date"] <= cutoff].reset_index(drop=True)
    report.n_orders_after_censor = int(len(orders))
    report.n_orders_returned = int(orders["returned"].sum())

    return orders, report


#: Columns an ingested file carries when it already knows which orders were returned.
FLAG_COLUMNS: tuple[str, str] = ("_returned_flag", "_return_event_date")


def label_from_flags(
    txn: pd.DataFrame, orders: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, LabelReport]:
    """Attach the label directly, for files that record returns explicitly.

    `link_cancellations` exists because Online Retail II has no returns column and no
    foreign key -- it has to *guess* which purchase a cancellation reverses, and it does
    so by taking the most recent prior purchase of the same (customer, stock code).

    When a merchant's own export says "order 700123 was returned", that guess is not
    just unnecessary, it is actively wrong: a customer who buys the same SKU twice would
    have the return attributed to whichever order came last, scrambling the labels of
    both. On a file with repeat purchasing that is enough to destroy the signal entirely.

    So when the ingest layer has ground truth, we use it. Everything else -- the
    right-censoring rule, the return-event date that governs leakage-safe history -- is
    identical to the cancellation path, because those parts are about *time*, not about
    how the return was discovered.
    """
    lookahead = pd.Timedelta(days=int(cfg["label"]["lookahead_days"]))
    report = LabelReport(lookahead_days=int(cfg["label"]["lookahead_days"]))

    buys = txn[~txn["is_cancellation"]]
    flags = (
        buys.groupby("invoice", sort=False)
        .agg(
            _flag=("_returned_flag", "max"),
            _event=("_return_event_date", "min"),
            _units=("quantity", "sum"),
        )
        .reset_index(names="order_id")
    )

    orders = orders.copy()
    merged = orders[["order_id"]].merge(flags, on="order_id", how="left")
    returned = merged["_flag"].fillna(0).astype(bool)
    event = pd.to_datetime(merged["_event"])

    # A return recorded outside the lookahead window is not observable at the horizon
    # the rest of the pipeline is built around, so it is not counted -- the same rule
    # the cancellation path applies via its `earliest` bound.
    order_date = pd.to_datetime(orders["order_date"]).reset_index(drop=True)
    within = event.notna() & (event <= order_date + lookahead) & (event >= order_date)
    dropped_outside = int((returned & ~within).sum())
    returned = returned & within

    orders["returned"] = returned.to_numpy().astype("int8")
    orders["return_event_date"] = event.where(returned)
    orders["returned_units"] = np.where(returned, merged["_units"].fillna(0.0), 0.0)

    report.n_cancellation_lines = int(len(txn[txn["is_cancellation"]]))
    report.n_matched_lines = report.n_cancellation_lines  # ground truth: nothing to match
    report.extra["labelled_from"] = "explicit return flag in the source file"
    report.extra["n_returns_outside_lookahead"] = dropped_outside

    report.n_orders_before_censor = int(len(orders))
    data_end = pd.to_datetime(txn["invoice_date"]).max()
    cutoff = data_end - lookahead
    report.censor_cutoff = cutoff
    orders = orders[orders["order_date"] <= cutoff].reset_index(drop=True)
    report.n_orders_after_censor = int(len(orders))
    report.n_orders_returned = int(orders["returned"].sum())
    return orders, report


def build_labelled_orders(txn: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, LabelReport]:
    """Order table + label, with the size filters from config applied."""
    orders = build_orders(txn)
    lo = float(cfg["data"]["min_order_value_gbp"])
    hi = float(cfg["data"]["max_order_value_gbp"])
    before = len(orders)
    orders = orders[(orders["order_value_gbp"] >= lo) & (orders["order_value_gbp"] <= hi)]
    orders = orders.reset_index(drop=True)

    # Ground truth if the source carried it; the linking heuristic only when it did not.
    if all(c in txn.columns for c in FLAG_COLUMNS):
        orders, report = label_from_flags(txn, orders, cfg)
    else:
        orders, report = link_cancellations(txn, orders, cfg)
    report.extra["n_orders_dropped_by_value_filter"] = int(before - len(orders) if before else 0)
    report.extra["order_value_filter_gbp"] = [lo, hi]
    return orders, report


def summarise(report: LabelReport) -> str:
    """Human-readable one-paragraph summary for logs and the README."""
    return (
        f"Label: returned=1 if a cancellation invoice from the same customer reverses a stock "
        f"code of the order within {report.lookahead_days} days. "
        f"{report.n_matched_lines:,}/{report.n_cancellation_lines:,} cancellation lines "
        f"({report.match_rate:.1%}) linked to an originating order. "
        f"{report.n_orders_after_censor:,} orders analysed after dropping "
        f"{report.n_orders_before_censor - report.n_orders_after_censor:,} right-censored orders "
        f"(placed after {pd.Timestamp(report.censor_cutoff).date()}). "
        f"Base rate = {report.base_rate:.2%}."
    )


__all__ = [
    "FLAG_COLUMNS",
    "LabelReport",
    "build_labelled_orders",
    "build_orders",
    "derive_category",
    "label_from_flags",
    "link_cancellations",
    "summarise",
]
