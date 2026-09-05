"""Leakage-safe, checkout-time feature construction.

Three families of feature, each with a different leakage hazard and a different guard:

1. **Order-intrinsic** (basket size, value, hour of day, country). Known at checkout by
   construction. No guard needed beyond the allow-list.

2. **Customer history** (prior order count, prior return rate). Hazard: using the
   customer's *future*, or using a prior order's label before that label existed.
   Guard: strictly-prior expanding windows, and -- the subtle part -- a prior return
   only counts once its *cancellation date* has passed. At checkout on 5 March you do
   not yet know that January's order will come back on 20 March. Most implementations
   get this wrong; `tests/test_leakage.py` asserts it row by row.

3. **Product risk** (category return rate). Hazard: a label-derived aggregate computed
   over data the model is scored on. Guard: fitted on the TRAIN slice only, and
   out-of-fold within train so a training row never sees its own label in its encoding.

Anything knowable only after checkout -- delivery time, refund date, review, the
cancellation itself -- is absent by construction and blocked by `POST_CHECKOUT_TERMS`.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import Config

# --- Region grouping. Coarse but stable, and used for the fairness breakdown. -------
_EU: Final[frozenset[str]] = frozenset(
    {
        "France", "Germany", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland",
        "Portugal", "Australia", "Italy", "Finland", "Norway", "Austria", "Denmark",
        "Sweden", "Poland", "Greece", "Cyprus", "Czech Republic", "Lithuania", "Malta",
        "Iceland", "Ireland", "Luxembourg", "Slovenia",
    }
)

#: Substrings that may never appear in a feature name. Anything matching is information
#: the merchant only learns AFTER the order is placed.
POST_CHECKOUT_TERMS: Final[tuple[str, ...]] = (
    "return_event",
    "returned",
    "cancel",
    "refund",
    "delivery",
    "delivered",
    "shipped",
    "review",
    "rating",
    "chargeback",
    "outcome",
    "label",
    "target",
    "future",
)

#: The exact model input space. Anything not listed here never reaches the model.
FEATURE_COLUMNS: Final[list[str]] = [
    # order-intrinsic
    "order_value_gbp",
    "log_order_value",
    "n_lines",
    "total_quantity",
    "avg_qty_per_line",
    "avg_unit_price",
    "max_unit_price",
    "min_unit_price",
    "price_spread",
    "basket_concentration",
    "discount_pct",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "days_to_christmas",
    # customer history (strictly prior)
    "prior_orders",
    "is_new_customer",
    "days_since_last_order",
    "customer_tenure_days",
    "prior_avg_order_value",
    "prior_return_rate",
    "prior_returns_observed",
    # product risk (train-only, out-of-fold)
    "category_return_rate",
    "category_order_count",
    # categoricals
    "country",
    "region",
    "top_category",
]

CATEGORICAL_COLUMNS: Final[list[str]] = ["country", "region", "top_category"]

#: Columns carried alongside the features for reporting/segmentation but NOT modelled.
PASSTHROUGH_COLUMNS: Final[list[str]] = [
    "order_id",
    "customer_id",
    "order_date",
    "returned",
    "return_event_date",
    "top_stock_code",
    "top_description",
]


def assert_no_post_checkout_features(columns: list[str]) -> None:
    """Raise if any modelled column name looks like post-checkout information."""
    offenders = [
        c for c in columns
        if any(term in c.lower() for term in POST_CHECKOUT_TERMS)
        # `prior_return_rate` / `prior_returns_observed` are the sanctioned exceptions:
        # they are label-derived but strictly-prior AND observation-gated.
        and not c.startswith("prior_")
        and c != "category_return_rate"
    ]
    if offenders:
        raise AssertionError(f"post-checkout feature(s) reached the model: {offenders}")


def _region(country: pd.Series) -> pd.Series:
    out = np.where(country == "United Kingdom", "UK", np.where(country.isin(_EU), "EU", "ROW"))
    return pd.Series(out, index=country.index, dtype="object")


def add_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Order-intrinsic features. Everything here is visible on the checkout screen."""
    df = orders.copy()
    ts = pd.to_datetime(df["order_date"])
    df["log_order_value"] = np.log1p(df["order_value_gbp"].clip(lower=0))
    df["avg_qty_per_line"] = df["total_quantity"] / df["n_lines"].clip(lower=1)
    df["price_spread"] = df["max_unit_price"] / df["min_unit_price"].clip(lower=0.01)
    df["basket_concentration"] = df["max_line_value_gbp"] / df["order_value_gbp"].clip(lower=0.01)
    df["hour"] = ts.dt.hour.astype("int16")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    # Gifting seasonality matters a lot for a giftware merchant's return rate.
    doy = ts.dt.dayofyear.astype("int32")
    df["days_to_christmas"] = np.minimum((359 - doy) % 365, (doy + 6) % 365).astype("int16")
    df["region"] = _region(df["country"].astype("object"))
    return df


def add_customer_history(orders: pd.DataFrame) -> pd.DataFrame:
    """Strictly-prior customer aggregates, with return counts gated on observation date.

    Computed over the full chronological order table (not per split): at test time a
    merchant genuinely does know the customer's earlier orders, including ones that
    fell in the training window. What it must NOT know is any order at or after the
    current timestamp, or any return that has not happened yet.
    """
    df = orders.sort_values(["order_date", "order_id"], kind="stable").reset_index(drop=True)

    n = len(df)
    prior_orders = np.zeros(n, dtype="int32")
    prior_returns = np.zeros(n, dtype="int32")
    days_since_last = np.full(n, np.nan)
    tenure = np.zeros(n, dtype="float64")
    prior_avg_value = np.full(n, np.nan)

    order_date = df["order_date"].to_numpy()
    ret_date = df["return_event_date"].to_numpy()
    value = df["order_value_gbp"].to_numpy(dtype="float64")

    # Per-customer running state.
    seen_returns: dict[str, list] = {}
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    last_date: dict[str, object] = {}
    first_date: dict[str, object] = {}

    for i, cust in enumerate(df["customer_id"].to_numpy()):
        t = order_date[i]
        k = counts.get(cust, 0)
        prior_orders[i] = k
        if k:
            prior_avg_value[i] = sums[cust] / k
            days_since_last[i] = (t - last_date[cust]) / np.timedelta64(1, "D")
            tenure[i] = (t - first_date[cust]) / np.timedelta64(1, "D")
            # Returns the merchant has ACTUALLY SEEN by time t.
            rl = seen_returns.get(cust)
            prior_returns[i] = bisect.bisect_left(rl, t) if rl else 0
        else:
            first_date[cust] = t

        counts[cust] = k + 1
        sums[cust] = sums.get(cust, 0.0) + float(value[i])
        last_date[cust] = t
        if not pd.isna(ret_date[i]):
            bisect.insort(seen_returns.setdefault(cust, []), ret_date[i])

    df["prior_orders"] = prior_orders
    df["prior_returns_observed"] = prior_returns
    df["is_new_customer"] = (df["prior_orders"] == 0).astype("int8")
    df["days_since_last_order"] = days_since_last
    df["customer_tenure_days"] = tenure
    df["prior_avg_order_value"] = prior_avg_value
    with np.errstate(invalid="ignore", divide="ignore"):
        df["prior_return_rate"] = np.where(
            prior_orders > 0, prior_returns / np.maximum(prior_orders, 1), np.nan
        )
    return df


@dataclass
class FeatureBuilder:
    """Fits the two train-only artefacts: reference prices and category return rates."""

    cfg: Config
    ref_price_: pd.Series | None = None
    category_rate_: pd.Series | None = None
    category_count_: pd.Series | None = None
    global_rate_: float = 0.0
    countries_: list[str] = field(default_factory=list)
    categories_: list[str] = field(default_factory=list)

    # -- fit ---------------------------------------------------------------------
    def fit(self, train_orders: pd.DataFrame, lines: pd.DataFrame) -> "FeatureBuilder":
        """Learn reference prices and OOF category risk from the TRAIN slice only."""
        train_ids = set(train_orders["order_id"])
        train_lines = lines[lines["invoice"].isin(train_ids)]

        stat = self.cfg["features"]["discount_reference"]
        self.ref_price_ = train_lines.groupby("stock_code")["unit_price"].agg(stat)

        self.global_rate_ = float(train_orders["returned"].mean())
        self.category_rate_, self.category_count_ = self._fit_category_rate(train_orders)

        top_n = int(self.cfg["features"]["top_n_countries"])
        self.countries_ = (
            train_orders["country"].value_counts().head(top_n).index.astype(str).tolist()
        )
        self.categories_ = (
            train_orders["top_category"].value_counts().head(60).index.astype(str).tolist()
        )
        return self

    def _fit_category_rate(self, train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Smoothed category return rate over the whole train slice (used for calib/test)."""
        m = float(self.cfg["features"]["target_encoding_smoothing"])
        grp = train.groupby("top_category")["returned"].agg(["sum", "count"])
        rate = (grp["sum"] + m * self.global_rate_) / (grp["count"] + m)
        return rate.astype("float64"), grp["count"].astype("float64")

    def oof_category_rate(self, train: pd.DataFrame) -> pd.Series:
        """Out-of-fold encoding for TRAIN rows, so no row sees its own label."""
        m = float(self.cfg["features"]["target_encoding_smoothing"])
        folds = int(self.cfg["features"]["target_encoding_folds"])
        seed = int(self.cfg["seed"])
        out = pd.Series(np.nan, index=train.index, dtype="float64")
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit_idx, enc_idx in kf.split(train):
            part = train.iloc[fit_idx]
            g_rate = float(part["returned"].mean())
            grp = part.groupby("top_category")["returned"].agg(["sum", "count"])
            rate = (grp["sum"] + m * g_rate) / (grp["count"] + m)
            keys = train.iloc[enc_idx]["top_category"]
            out.iloc[enc_idx] = keys.map(rate).fillna(g_rate).to_numpy()
        return out

    # -- transform ---------------------------------------------------------------
    def transform(
        self, orders: pd.DataFrame, lines: pd.DataFrame, *, is_train: bool = False
    ) -> pd.DataFrame:
        """Attach the fitted artefacts and return a frame carrying FEATURE_COLUMNS."""
        if self.ref_price_ is None:
            raise RuntimeError("FeatureBuilder.fit must be called before transform")
        df = orders.copy()

        df["discount_pct"] = self._discount(df, lines)

        if is_train:
            df["category_return_rate"] = self.oof_category_rate(df).to_numpy()
        else:
            df["category_return_rate"] = (
                df["top_category"].map(self.category_rate_).fillna(self.global_rate_)
            )
        df["category_order_count"] = (
            df["top_category"].map(self.category_count_).fillna(0.0)
        )

        df["country"] = pd.Categorical(
            df["country"].astype(str).where(df["country"].astype(str).isin(self.countries_), "OTHER"),
            categories=[*self.countries_, "OTHER"],
        )
        df["region"] = pd.Categorical(df["region"].astype(str), categories=["UK", "EU", "ROW"])
        df["top_category"] = pd.Categorical(
            df["top_category"].astype(str).where(
                df["top_category"].astype(str).isin(self.categories_), "OTHER"
            ),
            categories=[*self.categories_, "OTHER"],
        )
        return df

    def _discount(self, orders: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
        """Discount vs the train-set reference price for the same stock code.

        Online Retail II has no discount column, but the same stock code genuinely
        sells at different unit prices. `1 - paid/reference` recovers a usable
        promo-depth proxy -- and gives the naive baseline rule something real to fire on.
        """
        sub = lines[lines["invoice"].isin(set(orders["order_id"]))][
            ["invoice", "stock_code", "quantity", "unit_price", "line_value"]
        ].copy()
        sub["ref"] = sub["stock_code"].map(self.ref_price_)
        sub = sub.dropna(subset=["ref"])
        sub["expected_value"] = sub["quantity"] * sub["ref"]
        agg = sub.groupby("invoice")[["line_value", "expected_value"]].sum()
        disc = 1.0 - (agg["line_value"] / agg["expected_value"].replace(0, np.nan))
        return orders["order_id"].map(disc).fillna(0.0).clip(-2.0, 1.0)


def build_feature_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """Order-intrinsic + customer-history stage, shared by every split."""
    return add_customer_history(add_order_features(orders))


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the model input matrix, after the allow-list check."""
    assert_no_post_checkout_features(FEATURE_COLUMNS)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature frame is missing {missing}")
    return df[FEATURE_COLUMNS]


__all__ = [
    "CATEGORICAL_COLUMNS",
    "FEATURE_COLUMNS",
    "FeatureBuilder",
    "PASSTHROUGH_COLUMNS",
    "POST_CHECKOUT_TERMS",
    "add_customer_history",
    "add_order_features",
    "assert_no_post_checkout_features",
    "build_feature_frame",
    "select_features",
]
