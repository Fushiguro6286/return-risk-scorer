"""Leakage tests -- the ones that would actually catch a mistake (Tier 0.2).

Three hazards, three tests:

1. A feature that is not knowable at checkout reaches the model.
2. Customer history looks forward in time -- either at later orders, or at a return
   that had not happened yet when the order was placed.
3. A label-derived aggregate is fitted on data the model is then scored against.

The second is the one that is easy to get wrong and impossible to spot in a metric,
so it is asserted row by row against a hand-built table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from returnrisk.features import (
    FEATURE_COLUMNS,
    POST_CHECKOUT_TERMS,
    FeatureBuilder,
    add_customer_history,
    assert_no_post_checkout_features,
    build_feature_frame,
    select_features,
)
from returnrisk.split import temporal_split


# --------------------------------------------------------------------------- 1. allow-list
def test_no_post_checkout_feature_names():
    """No modelled column may look like information learned after checkout."""
    assert_no_post_checkout_features(FEATURE_COLUMNS)


def test_label_columns_are_not_features():
    """The label and its timestamp must never be model inputs."""
    for banned in ("returned", "return_event_date", "returned_units", "is_cancellation"):
        assert banned not in FEATURE_COLUMNS


def test_allow_list_actually_rejects_a_leak():
    """The guard must fail on a planted post-checkout column, or it proves nothing."""
    with pytest.raises(AssertionError, match="post-checkout"):
        assert_no_post_checkout_features([*FEATURE_COLUMNS, "days_to_delivery"])
    with pytest.raises(AssertionError):
        assert_no_post_checkout_features([*FEATURE_COLUMNS, "refund_amount"])


def test_post_checkout_terms_cover_the_obvious_leaks():
    for term in ("delivery", "refund", "review", "chargeback", "cancel"):
        assert term in POST_CHECKOUT_TERMS


# ------------------------------------------------------- 2. strictly-prior customer history
def test_prior_history_is_row_by_row_correct(toy_orders):
    """Hand-checked expectations for every row of the toy table.

    The critical row is a2: customer A's first order WAS returned, but the return did
    not land until 2024-03-20, ten days after a2 was placed. A naive implementation
    reports a 100% prior return rate here. The correct answer is 0 observed returns.
    """
    out = add_customer_history(toy_orders).set_index("order_id")

    # prior_orders counts only strictly earlier orders by the same customer.
    assert out.loc["a1", "prior_orders"] == 0
    assert out.loc["a2", "prior_orders"] == 1
    assert out.loc["a3", "prior_orders"] == 2
    assert out.loc["a4", "prior_orders"] == 3
    assert out.loc["b1", "prior_orders"] == 0
    assert out.loc["b2", "prior_orders"] == 1

    # prior_returns_observed counts only returns whose EVENT DATE precedes this order.
    assert out.loc["a1", "prior_returns_observed"] == 0
    assert out.loc["a2", "prior_returns_observed"] == 0, (
        "a1's return happened on 2024-03-20, after a2 was placed on 2024-03-10"
    )
    assert out.loc["a3", "prior_returns_observed"] == 1  # a1's return is now visible
    assert out.loc["a4", "prior_returns_observed"] == 2  # a1 and a3 both visible

    assert out.loc["a2", "prior_return_rate"] == pytest.approx(0.0)
    assert out.loc["a3", "prior_return_rate"] == pytest.approx(0.5)
    assert out.loc["a4", "prior_return_rate"] == pytest.approx(2 / 3)

    # A customer's first order has no history at all.
    assert bool(out.loc["a1", "is_new_customer"])
    assert np.isnan(out.loc["a1", "prior_return_rate"])
    assert np.isnan(out.loc["a1", "days_since_last_order"])


def test_prior_history_never_exceeds_prior_orders(synthetic_orders):
    """Observed returns can never outnumber the orders they came from."""
    out = add_customer_history(synthetic_orders)
    assert (out["prior_returns_observed"] <= out["prior_orders"]).all()
    rate = out["prior_return_rate"].dropna()
    assert ((rate >= 0) & (rate <= 1)).all()


def test_prior_history_matches_a_brute_force_recomputation(synthetic_orders):
    """Independent O(n^2) recomputation over a sample -- different code, same answer."""
    out = add_customer_history(synthetic_orders)
    rng = np.random.default_rng(0)
    sample = out.sample(min(300, len(out)), random_state=0)

    for row in sample.itertuples():
        peers = out[(out["customer_id"] == row.customer_id)]
        earlier = peers[peers["order_date"] < row.order_date]
        assert row.prior_orders == len(earlier), f"prior_orders wrong for {row.order_id}"

        observed = earlier["return_event_date"].dropna()
        expected = int((observed < row.order_date).sum())
        assert row.prior_returns_observed == expected, (
            f"prior_returns_observed wrong for {row.order_id}: "
            f"got {row.prior_returns_observed}, expected {expected}"
        )


def test_history_ignores_the_future_entirely(toy_orders):
    """Deleting every order after a cut-off must not change features before it.

    This is the strongest possible statement of 'no lookahead': if the future is
    genuinely unused, removing it is a no-op.
    """
    full = add_customer_history(toy_orders).set_index("order_id")
    truncated = add_customer_history(
        toy_orders[toy_orders["order_date"] <= "2024-04-01"]
    ).set_index("order_id")

    cols = ["prior_orders", "prior_returns_observed", "prior_avg_order_value", "customer_tenure_days"]
    for oid in truncated.index:
        for c in cols:
            a, b = full.loc[oid, c], truncated.loc[oid, c]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (
                f"{c} for {oid} changed when future orders were removed"
            )


# ------------------------------------------------- 3. train-only, out-of-fold aggregates
def test_target_encoding_is_fitted_on_train_only(synthetic_orders, synthetic_txn, cfg):
    """Corrupting test labels must not change the encoding test rows receive."""
    feat = build_feature_frame(synthetic_orders)
    sp = temporal_split(feat, cfg)
    lines = synthetic_txn[~synthetic_txn["is_cancellation"]]

    builder = FeatureBuilder(cfg).fit(sp.train, lines)
    baseline = builder.transform(sp.test, lines)["category_return_rate"].to_numpy()

    # Flip every test label; a train-only encoder cannot notice.
    poisoned_test = sp.test.copy()
    poisoned_test["returned"] = 1 - poisoned_test["returned"]
    after = builder.transform(poisoned_test, lines)["category_return_rate"].to_numpy()

    np.testing.assert_allclose(baseline, after, err_msg="category encoding saw test labels")


def test_reference_prices_are_fitted_on_train_only(synthetic_orders, synthetic_txn, cfg):
    """The discount proxy's reference prices must come from the training slice alone."""
    feat = build_feature_frame(synthetic_orders)
    sp = temporal_split(feat, cfg)
    lines = synthetic_txn[~synthetic_txn["is_cancellation"]]

    builder = FeatureBuilder(cfg).fit(sp.train, lines)
    train_ids = set(sp.train["order_id"])
    train_codes = set(lines[lines["invoice"].isin(train_ids)]["stock_code"])
    assert set(builder.ref_price_.index) <= train_codes


def test_oof_encoding_differs_from_in_fold(synthetic_orders, synthetic_txn, cfg):
    """Out-of-fold encoding must not equal the full-train encoding on train rows.

    If they matched, every training row would be seeing its own label through the
    encoder -- the exact overfit OOF exists to prevent.
    """
    feat = build_feature_frame(synthetic_orders)
    sp = temporal_split(feat, cfg)
    lines = synthetic_txn[~synthetic_txn["is_cancellation"]]
    builder = FeatureBuilder(cfg).fit(sp.train, lines)

    oof = builder.oof_category_rate(sp.train).to_numpy()
    in_fold = sp.train["top_category"].map(builder.category_rate_).to_numpy()
    assert not np.allclose(oof, in_fold), "OOF encoding collapsed to the in-fold encoding"


# ------------------------------------------------------------------ 4. end-to-end guard
def test_model_matrix_contains_only_allow_listed_columns(synthetic_orders, synthetic_txn, cfg):
    feat = build_feature_frame(synthetic_orders)
    sp = temporal_split(feat, cfg)
    lines = synthetic_txn[~synthetic_txn["is_cancellation"]]
    builder = FeatureBuilder(cfg).fit(sp.train, lines)
    X = select_features(builder.transform(sp.test, lines))
    assert list(X.columns) == FEATURE_COLUMNS
