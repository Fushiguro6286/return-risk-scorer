"""The bring-your-own-data front door.

The dangerous failure here is not a crash -- it is a file that maps *almost* right,
trains without complaint, and produces a confident model built on the wrong column. So
these tests care most about two things: that detection actually works on files that look
nothing like Online Retail II, and that when it cannot work, the run stops instead of
guessing.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from returnrisk.data.ingest import (
    REQUIRED_FIELDS,
    ColumnMapping,
    list_tables,
    profile,
    read_any,
    suggest_mapping,
    to_canonical,
    validate,
    with_file_currency,
)


def _foreign_frame(n_orders: int = 900, seed: int = 3) -> pd.DataFrame:
    """An export sharing no column names with the canonical schema."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    rows = []
    for i in range(n_orders):
        when = start + pd.Timedelta(days=float(rng.uniform(0, 640)))
        returned = bool(rng.random() < 0.25)
        for _ in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "Order Number": f"ORD{100000 + i}",
                    "Placed On": when.strftime("%Y-%m-%d %H:%M:%S"),
                    "Item SKU": f"SKU-{rng.integers(1000, 1040)}",
                    "Item Name": "Widget",
                    "Product Category": rng.choice(["Apparel", "Home"]),
                    "Units": int(rng.integers(1, 6)),
                    "Rate (INR)": round(float(rng.uniform(90, 2400)), 2),
                    "Buyer Email": f"user{int(rng.integers(0, 120)):03d}@example.com",
                    "Ship City": "Mumbai",
                    "Order Status": "RETURNED" if returned else "DELIVERED",
                    "Returned On": (
                        (when + pd.Timedelta(days=12)).strftime("%Y-%m-%d") if returned else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def foreign() -> pd.DataFrame:
    return _foreign_frame()


# --------------------------------------------------------------------------- sniffing
def test_every_required_field_is_found_in_a_foreign_export(foreign):
    """Not one of these column names matches the canonical schema."""
    mapping, _ = suggest_mapping(foreign)
    assert mapping.missing_required() == []


def test_columns_map_to_the_right_source(foreign):
    mapping, _ = suggest_mapping(foreign)
    assert mapping.columns["invoice"] == "Order Number"
    assert mapping.columns["invoice_date"] == "Placed On"
    assert mapping.columns["stock_code"] == "Item SKU"
    assert mapping.columns["quantity"] == "Units"
    assert mapping.columns["unit_price"] == "Rate (INR)"
    assert mapping.columns["customer_id"] == "Buyer Email"
    assert mapping.columns["category"] == "Product Category"
    assert mapping.columns["returned"] == "Order Status"
    assert mapping.columns["return_date"] == "Returned On"


def test_multi_word_headers_match_underscored_synonyms(foreign):
    """`Order Status` and the synonym `order_status` must normalise to the same thing --
    keeping underscores in the normaliser silently broke every multi-word synonym."""
    mapping, _ = suggest_mapping(foreign)
    assert mapping.columns["returned"] == "Order Status"


def test_a_status_column_selects_flag_mode(foreign):
    mapping, _ = suggest_mapping(foreign)
    assert mapping.return_mode == "flag"


def test_negative_quantities_select_cancellation_mode(foreign):
    uci_like = foreign.drop(columns=["Order Status", "Returned On"]).copy()
    uci_like.loc[uci_like.index[:40], "Units"] *= -1
    mapping, _ = suggest_mapping(uci_like)
    assert mapping.return_mode == "cancellation"


def test_a_file_with_no_return_information_is_refused(foreign):
    """No flag column and no negative quantities means there is nothing to learn from."""
    blind = foreign.drop(columns=["Order Status", "Returned On"])
    _, issues = suggest_mapping(blind)
    assert any(i.level == "error" for i in issues)


def test_a_missing_required_column_is_an_error(foreign):
    _, issues = suggest_mapping(foreign.drop(columns=["Placed On"]))
    assert any(i.level == "error" and "invoice_date" in i.message for i in issues)


# ------------------------------------------------------------------------ validation
def test_a_tiny_file_is_rejected(foreign):
    mapping, _ = suggest_mapping(foreign)
    issues = validate(foreign.head(50), mapping)
    assert any(i.level == "error" and "rows" in i.message for i in issues)


def test_a_short_history_is_rejected(foreign):
    """A temporal split plus a 90-day embargo cannot come out of three months."""
    mapping, _ = suggest_mapping(foreign)
    short = foreign.copy()
    short["Placed On"] = "2023-01-05 10:00:00"
    issues = validate(short, mapping)
    assert any(i.level == "error" and "spans" in i.message for i in issues)


def test_a_collapsing_sku_prefix_is_warned_about(foreign):
    """Every SKU here starts with 'SKU', so the prefix heuristic yields one family and
    would silently delete all product-level signal."""
    mapping, _ = suggest_mapping(foreign)
    mapping.columns["category"] = None
    issues = validate(foreign, mapping)
    assert any(i.level == "warning" and "prefix" in i.message for i in issues)


def test_a_clean_file_has_no_errors(foreign):
    mapping, _ = suggest_mapping(foreign)
    assert [i for i in validate(foreign, mapping) if i.level == "error"] == []


def test_missing_return_date_warns_about_the_assumed_lag(foreign):
    mapping, _ = suggest_mapping(foreign)
    mapping.columns["return_date"] = None
    issues = validate(foreign, mapping)
    assert any(i.level == "warning" and "lag" in i.message for i in issues)


# ------------------------------------------------------------------------ conversion
def test_canonical_frame_has_the_expected_shape(foreign):
    mapping, _ = suggest_mapping(foreign)
    out, _ = to_canonical(foreign, mapping)
    for col in ("invoice", "stock_code", "quantity", "invoice_date", "unit_price",
                "customer_id", "country", "description"):
        assert col in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["invoice_date"])


def test_flag_mode_carries_ground_truth_rather_than_re_deriving_it(foreign):
    """The label must travel with the order it belongs to. Re-deriving it by matching
    cancellations would pin a return to the wrong order whenever a customer buys the
    same SKU twice."""
    mapping, _ = suggest_mapping(foreign)
    out, _ = to_canonical(foreign, mapping)
    assert "_returned_flag" in out.columns
    assert "_return_event_date" in out.columns

    purchases = out[~out["invoice"].str.startswith("C")]
    truth = (
        foreign.assign(r=foreign["Order Status"].eq("RETURNED"))
        .groupby("Order Number")["r"]
        .max()
    )
    got = purchases.groupby("invoice")["_returned_flag"].max()
    assert got.reindex(truth.index).fillna(False).equals(truth.astype(bool))


def test_flag_mode_still_emits_visible_cancellation_rows(foreign):
    mapping, _ = suggest_mapping(foreign)
    out, _ = to_canonical(foreign, mapping)
    cancels = out[out["invoice"].str.startswith("C")]
    assert len(cancels) > 0
    assert (cancels["quantity"] < 0).all()
    assert not cancels["_returned_flag"].any(), "a cancellation row must not carry the label"


def test_non_numeric_customer_ids_become_stable_integers(foreign):
    """Emails and UUIDs are common; the downstream cast expects something numeric."""
    mapping, _ = suggest_mapping(foreign)
    out, _ = to_canonical(foreign, mapping)
    assert pd.api.types.is_numeric_dtype(out["customer_id"])
    n_unique_source = foreign["Buyer Email"].nunique()
    purchases = out[~out["invoice"].str.startswith("C")]
    assert purchases["customer_id"].nunique() == n_unique_source


def test_missing_optional_columns_are_filled(foreign):
    mapping, _ = suggest_mapping(foreign)
    mapping.columns["country"] = None
    mapping.columns["description"] = None
    out, _ = to_canonical(foreign, mapping)
    assert (out["country"] == "Unknown").all()
    assert out["description"].notna().all()


def test_conversion_refuses_an_incomplete_mapping(foreign):
    mapping = ColumnMapping(columns={f: None for f in REQUIRED_FIELDS})
    with pytest.raises(ValueError, match="unmapped required fields"):
        to_canonical(foreign, mapping)


def test_return_dated_before_its_order_is_repaired(foreign):
    mapping, _ = suggest_mapping(foreign)
    broken = foreign.copy()
    mask = broken["Order Status"].eq("RETURNED")
    broken.loc[mask, "Returned On"] = "2020-01-01"
    _, notes = to_canonical(broken, mapping)
    assert any("before their order date" in n.message for n in notes)


# --------------------------------------------------------------------------- readers
def test_reads_csv(tmp_path, foreign):
    p = tmp_path / "x.csv"
    foreign.to_csv(p, index=False)
    assert len(read_any(p)) == len(foreign)


def test_reads_semicolon_delimited(tmp_path, foreign):
    p = tmp_path / "x.csv"
    foreign.to_csv(p, index=False, sep=";")
    assert read_any(p).shape[1] == foreign.shape[1]


def test_reads_parquet(tmp_path, foreign):
    p = tmp_path / "x.parquet"
    foreign.to_parquet(p, index=False)
    assert len(read_any(p)) == len(foreign)


def test_reads_jsonl(tmp_path, foreign):
    p = tmp_path / "x.jsonl"
    foreign.to_json(p, orient="records", lines=True)
    assert len(read_any(p)) == len(foreign)


def test_reads_sqlite_and_picks_the_largest_table(tmp_path, foreign):
    p = tmp_path / "shop.db"
    with sqlite3.connect(p) as conn:
        foreign.to_sql("order_lines", conn, index=False)
        foreign.head(5).to_sql("settings", conn, index=False)
    assert set(list_tables(p)) == {"order_lines", "settings"}
    assert len(read_any(p)) == len(foreign)  # not the 5-row table


def test_reads_a_zipped_csv(tmp_path, foreign):
    import zipfile

    csv = tmp_path / "inner.csv"
    foreign.to_csv(csv, index=False)
    z = tmp_path / "bundle.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(csv, arcname="inner.csv")
    assert len(read_any(z)) == len(foreign)


def test_unsupported_extension_is_a_readable_error(tmp_path):
    p = tmp_path / "notes.docx"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported file type"):
        read_any(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_any(tmp_path / "absent.csv")


# -------------------------------------------------------------------------- currency
def test_file_runs_do_not_inherit_the_uci_exchange_rate(cfg):
    """money.gbp_to_inr is 105 because the reference data is British. Applying it to a
    file already in rupees inflates every order 105-fold and drives t* to zero."""
    payload = cfg.as_dict()
    payload["data"] = {**payload["data"], "source": "file",
                       "file": {**(payload["data"].get("file") or {}), "currency_to_inr": 1.0}}
    patched = with_file_currency(type(cfg)(_data=payload, path=cfg.path))
    assert float(patched["money"]["gbp_to_inr"]) == 1.0


def test_uci_runs_keep_their_exchange_rate(cfg):
    assert float(with_file_currency(cfg)["money"]["gbp_to_inr"]) == 105.0


# -------------------------------------------------------------------------- profiling
def test_profile_reports_what_the_confirmation_screen_shows(foreign):
    mapping, _ = suggest_mapping(foreign)
    info = profile(foreign, mapping)
    assert info["n_rows"] == len(foreign)
    assert info["n_orders"] == foreign["Order Number"].nunique()
    assert info["n_customers"] == foreign["Buyer Email"].nunique()
    assert info["span_days"] > 300
    assert 0 < info["flagged_return_rate"] < 1
