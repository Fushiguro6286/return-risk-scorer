"""Temporal split and cancellation-linking tests."""
from __future__ import annotations

import pandas as pd
import pytest

from returnrisk.data.labeling import build_labelled_orders, derive_category, link_cancellations
from returnrisk.data.loader import clean
from returnrisk.features import build_feature_frame
from returnrisk.split import temporal_split


# ------------------------------------------------------------------------- split
def test_split_is_strictly_chronological(synthetic_orders, cfg):
    sp = temporal_split(build_feature_frame(synthetic_orders), cfg)
    assert sp.train["order_date"].max() <= sp.calib["order_date"].min()
    assert sp.calib["order_date"].max() < sp.test["order_date"].min()


def test_split_has_no_overlapping_orders(synthetic_orders, cfg):
    sp = temporal_split(build_feature_frame(synthetic_orders), cfg)
    ids = [set(part["order_id"]) for part in (sp.train, sp.calib, sp.test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_embargo_gap_is_actually_enforced(synthetic_orders, cfg):
    """No fitting order may sit within `embargo_days` of the deployment moment."""
    sp = temporal_split(build_feature_frame(synthetic_orders), cfg)
    embargo = pd.Timedelta(days=int(cfg["split"]["embargo_days"]))
    deploy = pd.Timestamp(sp.meta["deployment_moment"])
    assert sp.calib["order_date"].max() <= deploy - embargo
    assert sp.train["order_date"].max() <= deploy - embargo
    assert (deploy - sp.calib["order_date"].max()) >= embargo


def test_split_describe_reports_all_three_date_ranges(synthetic_orders, cfg):
    """Acceptance criterion 0.3: the split must print its date ranges."""
    text = temporal_split(build_feature_frame(synthetic_orders), cfg).describe()
    for name in ("train", "calib", "test"):
        assert name in text
    assert "base rate" in text
    assert "embargo" in text


def test_split_rejects_an_impossible_embargo(synthetic_orders, cfg):
    patched = cfg.as_dict()
    patched["split"] = {**patched["split"], "embargo_days": 100000}
    bad = type(cfg)(_data=patched, path=cfg.path)
    with pytest.raises(ValueError, match="embargo"):
        temporal_split(build_feature_frame(synthetic_orders), bad)


# ----------------------------------------------------------------------- labels
def test_label_is_binary_and_matches_the_event_date(synthetic_orders):
    assert set(synthetic_orders["returned"].unique()) <= {0, 1}
    returned = synthetic_orders[synthetic_orders["returned"] == 1]
    assert returned["return_event_date"].notna().all()
    not_returned = synthetic_orders[synthetic_orders["returned"] == 0]
    assert not_returned["return_event_date"].isna().all()


def test_return_event_never_precedes_the_order(synthetic_orders):
    r = synthetic_orders[synthetic_orders["returned"] == 1]
    assert (r["return_event_date"] >= r["order_date"]).all()


def test_return_event_is_inside_the_lookahead_window(synthetic_orders, cfg):
    r = synthetic_orders[synthetic_orders["returned"] == 1]
    window = pd.Timedelta(days=int(cfg["label"]["lookahead_days"]))
    assert ((r["return_event_date"] - r["order_date"]) <= window).all()


def test_right_censored_orders_are_dropped(synthetic_txn, synthetic_orders, cfg):
    """Orders whose 90-day window runs past the end of the data must not be labelled 0."""
    window = pd.Timedelta(days=int(cfg["label"]["lookahead_days"]))
    cutoff = synthetic_txn["invoice_date"].max() - window
    assert synthetic_orders["order_date"].max() <= cutoff


def test_cancellation_quantity_is_consumed_once(cfg):
    """One returned unit may not mark two different orders as returned.

    Customer A buys the same item twice, then cancels a quantity matching only the
    later purchase. Greedy most-recent-first matching must attribute it to that order
    alone and leave the earlier one clean.

    The trailing filler order exists only to push the dataset's end date out, so the
    two orders under test survive right-censoring.
    """
    txn = clean(
        pd.DataFrame(
            {
                "invoice": ["1", "2", "C9", "999"],
                "stock_code": ["20001"] * 4,
                "description": ["ITEM"] * 4,
                "quantity": [5.0, 3.0, -3.0, 1.0],
                "invoice_date": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-02-10", "2024-12-31"]
                ),
                "unit_price": [10.0] * 4,
                "customer_id": [1.0, 1.0, 1.0, 2.0],
                "country": ["United Kingdom"] * 4,
            }
        ),
        cfg,
    )
    from returnrisk.data.labeling import build_orders

    orders = build_orders(txn)
    labelled, report = link_cancellations(txn, orders, cfg)
    labelled = labelled.set_index("order_id")
    assert labelled.loc["2", "returned"] == 1, "the matching-quantity order should be flagged"
    assert labelled.loc["1", "returned"] == 0, "the earlier order must not also be flagged"
    assert report.n_matched_lines == 1


def test_unmatchable_cancellation_is_reported_not_silently_dropped(cfg):
    """A cancellation for an item the customer never bought must lower the match rate."""
    txn = clean(
        pd.DataFrame(
            {
                "invoice": ["1", "C9"],
                "stock_code": ["20001", "29999"],
                "description": ["ITEM", "OTHER"],
                "quantity": [5.0, -2.0],
                "invoice_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
                "unit_price": [10.0, 10.0],
                "customer_id": [1.0, 1.0],
                "country": ["United Kingdom"] * 2,
            }
        ),
        cfg,
    )
    from returnrisk.data.labeling import build_orders

    _, report = link_cancellations(txn, build_orders(txn), cfg)
    assert report.n_cancellation_lines == 1
    assert report.n_matched_lines == 0
    assert report.match_rate == 0.0


def test_cancellation_outside_the_window_does_not_match(cfg):
    txn = clean(
        pd.DataFrame(
            {
                "invoice": ["1", "C9"],
                "stock_code": ["20001", "20001"],
                "description": ["ITEM", "ITEM"],
                "quantity": [5.0, -2.0],
                "invoice_date": pd.to_datetime(["2024-01-01", "2024-09-01"]),  # 244 days later
                "unit_price": [10.0, 10.0],
                "customer_id": [1.0, 1.0],
                "country": ["United Kingdom"] * 2,
            }
        ),
        cfg,
    )
    from returnrisk.data.labeling import build_orders

    _, report = link_cancellations(txn, build_orders(txn), cfg)
    assert report.n_matched_lines == 0


def test_label_report_states_the_definition(synthetic_txn, cfg):
    _, report = build_labelled_orders(synthetic_txn, cfg)
    d = report.to_dict()
    assert "returned=1" in d["label_definition"]
    assert 0.0 <= d["base_rate"] <= 1.0
    assert d["n_orders_analysed"] > 0


def test_category_derivation_is_a_stable_prefix():
    s = pd.Series(["20001", "85123A", "22423", "DOT"], dtype="string")
    assert derive_category(s).tolist() == ["200", "851", "224", "DOT"]
