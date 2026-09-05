"""Temporal split with a label-observability embargo.

A random split would be indefensible here: customer history features and product return
rates both leak backwards through time, and the merchant's real question is always
"how will this do on *next* month's orders".

The embargo is the part people skip. Fix the deployment moment at T, the first day of
the test window. At that instant the merchant can only have labels for orders placed
before ``T - lookahead_days``: a 90-day return window has not closed for anything newer.
Training right up to T quietly teaches the model on labels that would not exist yet, and
inflates test performance.

So the *fitting window* -- train and calibration together, since both are fitted at the
same moment T -- ends at ``T - embargo_days``, and the orders in that 90-day gap are
dropped. There is deliberately no gap between train and calibration: they are fitted
simultaneously, so nothing is hidden from one that is available to the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import Config


@dataclass
class SplitResult:
    """The three slices plus enough metadata to print and audit the split."""

    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    meta: dict[str, Any]

    def describe(self) -> str:
        lines = ["Temporal split (no shuffling, embargoed):"]
        for name in ("train", "calib", "test"):
            df = getattr(self, name)
            lines.append(
                f"  {name:<6} n={len(df):>7,}  "
                f"{df['order_date'].min():%Y-%m-%d} -> {df['order_date'].max():%Y-%m-%d}  "
                f"base rate={df['returned'].mean():.2%}"
            )
        lines.append(
            f"  embargo={self.meta['embargo_days']}d before the deployment moment "
            f"{self.meta['deployment_moment'][:10]} "
            f"(dropped {self.meta['n_embargoed']:,} orders whose return window was still open)"
        )
        return "\n".join(lines)


def temporal_split(orders: pd.DataFrame, cfg: Config) -> SplitResult:
    """Chronological split with one embargo gap between the fitting window and test."""
    s = cfg["split"]
    train_frac = float(s["train_frac"])
    calib_frac = float(s["calib_frac"])
    embargo = pd.Timedelta(days=int(s["embargo_days"]))

    df = orders.sort_values(["order_date", "order_id"], kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError(f"need at least 100 orders to split, got {n}")

    # T = the deployment moment: everything after it is the held-out future.
    t_deploy = df["order_date"].iloc[int(n * (train_frac + calib_frac))]
    fit_end = t_deploy - embargo

    fit = df[df["order_date"] <= fit_end]
    if len(fit) < 50:
        raise ValueError(
            f"embargo of {embargo.days}d leaves only {len(fit)} fitting orders; "
            "shorten split.embargo_days or widen the train/calib fractions"
        )

    # Split the fitting window itself into train then calibration, chronologically.
    share = train_frac / (train_frac + calib_frac)
    t_calib = fit["order_date"].iloc[int(len(fit) * share)]
    train = fit[fit["order_date"] <= t_calib]
    calib = fit[fit["order_date"] > t_calib]
    test = df[df["order_date"] > t_deploy]

    kept = len(train) + len(calib) + len(test)
    meta = {
        "n_total": n,
        "boundary_train_calib": str(t_calib),
        "deployment_moment": str(t_deploy),
        "fitting_window_end": str(fit_end),
        "embargo_days": int(s["embargo_days"]),
        "n_embargoed": int(n - kept),
        "train": _range(train),
        "calib": _range(calib),
        "test": _range(test),
    }
    for name, part in (("train", train), ("calib", calib), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} slice is empty; loosen split fractions or embargo_days")
    return SplitResult(
        train.reset_index(drop=True),
        calib.reset_index(drop=True),
        test.reset_index(drop=True),
        meta,
    )


def _range(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": int(len(df)),
        "start": str(df["order_date"].min()) if len(df) else None,
        "end": str(df["order_date"].max()) if len(df) else None,
        "base_rate": float(df["returned"].mean()) if len(df) else None,
    }


__all__ = ["SplitResult", "temporal_split"]
