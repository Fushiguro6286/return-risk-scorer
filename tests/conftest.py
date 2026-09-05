"""Shared fixtures. Puts `src/` on the path so tests run from a clean clone."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from returnrisk.config import load_config  # noqa: E402
from returnrisk.data.labeling import build_labelled_orders  # noqa: E402
from returnrisk.data.loader import clean, make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def synthetic_txn(cfg):
    """Small generated transaction frame -- fast, and enough to exercise every path."""
    return clean(make_synthetic(cfg, n_orders=2500), cfg)


@pytest.fixture(scope="session")
def synthetic_orders(synthetic_txn, cfg):
    orders, _ = build_labelled_orders(synthetic_txn, cfg)
    return orders


@pytest.fixture
def toy_orders() -> pd.DataFrame:
    """A hand-built order table with outcomes we can reason about by eye.

    Customer A places four orders. The first is returned, but not until 2024-03-20 --
    which is AFTER their second order on 2024-03-10. So at checkout on 10 March the
    merchant knows about zero returns; only from the third order onward is that return
    part of the customer's observable history.
    """
    return pd.DataFrame(
        {
            "order_id": ["a1", "a2", "a3", "a4", "b1", "b2"],
            "customer_id": ["A", "A", "A", "A", "B", "B"],
            "order_date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-03-10",
                    "2024-04-01",
                    "2024-06-01",
                    "2024-02-01",
                    "2024-05-01",
                ]
            ),
            "return_event_date": pd.to_datetime(
                [
                    "2024-03-20",  # a1 returned, but only observed on 20 March
                    None,
                    "2024-04-15",  # a3 returned, observed before a4
                    None,
                    None,
                    None,
                ]
            ),
            "returned": [1, 0, 1, 0, 0, 0],
            "order_value_gbp": [100.0, 200.0, 300.0, 400.0, 50.0, 60.0],
            "country": ["United Kingdom"] * 6,
            "n_lines": [2, 3, 4, 5, 1, 2],
            "total_quantity": [10, 12, 14, 16, 3, 4],
            "max_line_value_gbp": [60.0, 120.0, 180.0, 240.0, 30.0, 36.0],
            "avg_unit_price": [5.0, 6.0, 7.0, 8.0, 4.0, 4.5],
            "max_unit_price": [9.0, 10.0, 11.0, 12.0, 5.0, 6.0],
            "min_unit_price": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            "top_stock_code": ["20001"] * 6,
            "top_category": ["200"] * 6,
            "top_description": ["ITEM"] * 6,
            "returned_units": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
