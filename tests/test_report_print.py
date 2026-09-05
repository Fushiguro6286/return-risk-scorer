"""The printable report: self-contained, section-gated, and injection-safe.

The one property worth defending hardest is that the document has no external
references. A report emailed to a merchant's auditor, or opened from an archive in two
years, must render identically -- which it cannot do if the charts are `<img src>`
pointing at a `reports/` folder that has since been regenerated.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

import pandas as pd
import pytest

from returnrisk.report_print import SECTIONS, build_report_html


def _tiny_png() -> bytes:
    """A real 1x1 PNG, so the base64 path is exercised on genuine bytes."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    d = tmp_path / "reports"
    d.mkdir()
    (d / "money_confusion_matrix.png").write_bytes(_tiny_png())
    (d / "pr_curve.png").write_bytes(_tiny_png())
    (d / "baseline_comparison.csv").write_text("rule,cost\ndo nothing,100\n", encoding="utf-8")
    (d / "scored_test_orders.csv").write_text("order_id,x\n1,2\n", encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps(
            {
                "metrics_test": {
                    "auc_pr": 0.3409, "base_rate": 0.1757, "roc_auc": 0.7046,
                    "brier": 0.1318, "precision": 0.2524, "recall": 0.7871,
                },
                "money": {
                    "action": "remove_discount", "threshold": 0.1219,
                    "do_nothing_cost": 28763670.0, "total_cost": 21106568.0,
                    "savings": 7657101.0, "n_flagged": 4116, "flag_rate": 0.548,
                },
                "label": {"label_definition": "returned=1 if ...", "n_orders_analysed": 30047},
                "split": {
                    "train": {"n": 15694}, "calib": {"n": 3923}, "test": {"n": 7511},
                    "embargo_days": 90, "n_embargoed": 2919,
                },
                "leakage_audit": {"verdict": "No leakage indicators."},
            }
        ),
        encoding="utf-8",
    )
    return d


# ------------------------------------------------------------------ self-contained
def test_every_image_is_inlined_not_linked(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert html.count("data:image/png;base64,") == 2
    # No reference that would need a file alongside the document.
    assert 'src="reports/' not in html
    assert ".png\"" not in html.replace("data:image/png", "")


def test_inlined_bytes_decode_back_to_the_original_png(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    payload = html.split("data:image/png;base64,")[1].split('"')[0]
    assert base64.b64decode(payload).startswith(b"\x89PNG")


def test_carries_its_own_stylesheet_and_print_rules(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert "<style>" in html
    assert "@media print" in html
    assert "@page" in html
    assert "window.print()" in html  # the button the user actually presses


def test_toolbar_is_hidden_when_printed(reports_dir):
    """The Print button must not appear on the paper it prints."""
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    printed = html.split("@media print")[1]
    assert ".bar{display:none" in printed.replace(" ", "")


# ----------------------------------------------------------------------- content
def test_headline_numbers_come_from_summary_json(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert "0.341" in html          # AUC-PR
    assert "17.57%" in html         # base rate
    assert "1.94x" in html          # lift, computed from the two above


def test_savings_percentage_is_derived_not_invented(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert "26.6%" in html  # 7,657,101 / 28,763,670


def test_split_block_reads_the_nested_slice_counts(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert "15,694" in html and "7,511" in html and "2,919" in html


def test_dataset_label_appears_in_title_and_body(reports_dir):
    html = build_report_html(dataset_label="Acme Q3", reports_dir=reports_dir)
    assert "<title>" in html and "Acme Q3" in html


def test_charts_follow_narrative_order_not_alphabetical(reports_dir):
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert html.index("Money confusion matrix") < html.index("Pr curve")


def test_bulky_tables_are_left_out(reports_dir):
    """A 7,500-row scored book is a CSV, not a printed page."""
    html = build_report_html(dataset_label="D", reports_dir=reports_dir)
    assert "Baseline comparison" in html
    assert "Scored test orders" not in html


def test_missing_summary_is_reported_not_crashed(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    html = build_report_html(dataset_label="Nothing", reports_dir=empty)
    assert "No <code>summary.json</code>" in html


# ---------------------------------------------------------------------- sections
@pytest.mark.parametrize("section", [s.key for s in SECTIONS])
def test_each_section_can_be_switched_off(reports_dir, section):
    everything = {s.key for s in SECTIONS}
    without = build_report_html(
        dataset_label="D",
        reports_dir=reports_dir,
        sections=everything - {section},
        ledger_stats={"n_decisions": 3, "by_final_action": {"none": 1}},
        policy={"min_order_value_inr": 500},
    )
    with_it = build_report_html(
        dataset_label="D",
        reports_dir=reports_dir,
        sections=everything,
        ledger_stats={"n_decisions": 3, "by_final_action": {"none": 1}},
        policy={"min_order_value_inr": 500},
    )
    assert len(without) < len(with_it)


def test_charts_section_off_drops_every_image(reports_dir):
    html = build_report_html(
        dataset_label="D", reports_dir=reports_dir, sections=("summary",)
    )
    assert "data:image/png" not in html


# -------------------------------------------------------------------- governance
def test_governance_renders_ledger_and_chain_state(reports_dir):
    html = build_report_html(
        dataset_label="D",
        reports_dir=reports_dir,
        ledger_stats={
            "n_decisions": 12, "n_outcomes_recorded": 4,
            "by_final_action": {"none": 5, "remove_cod": 7},
            "guardrails_fired": {"min_order_value": 2},
            "expected_saving_inr_total": 9100.0,
        },
        ledger_rows=pd.DataFrame(
            [{"seq": 1, "recorded_at": "2026-09-05", "risk_score": 0.4,
              "final_action": "remove_cod", "outcome": "actioned", "entry_hash": "ab12"}]
        ),
        chain_ok=True,
        policy={"min_order_value_inr": 500, "loyalty_shield": {"min_prior_orders": 4}},
    )
    assert "Governance and audit trail" in html
    assert "chain-ok" in html
    assert "min_order_value" in html
    assert "remove_cod" in html


def test_broken_chain_is_stated_loudly(reports_dir):
    html = build_report_html(
        dataset_label="D", reports_dir=reports_dir,
        ledger_stats={"n_decisions": 1, "by_final_action": {}}, chain_ok=False,
    )
    assert "chain-bad" in html and "BROKEN" in html


def test_unchecked_chain_is_not_claimed_intact(reports_dir):
    html = build_report_html(
        dataset_label="D", reports_dir=reports_dir,
        ledger_stats={"n_decisions": 1, "by_final_action": {}}, chain_ok=None,
    )
    assert "not checked" in html
    # The class is defined in the stylesheet either way; what matters is that it is
    # never applied to an element, which would assert an integrity check that never ran.
    body = html.split("</style>")[1]
    assert "chain-ok" not in body and "chain-bad" not in body


# ------------------------------------------------------------------------ safety
def test_dataset_label_is_escaped(reports_dir):
    """Labels are user input from the upload tab."""
    html = build_report_html(
        dataset_label="<script>alert(1)</script>", reports_dir=reports_dir
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_table_cells_are_escaped(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "lift_table.csv").write_text(
        "decile,note\n1,<img src=x onerror=alert(1)>\n", encoding="utf-8"
    )
    html = build_report_html(dataset_label="D", reports_dir=d)
    assert "onerror=alert(1)>" not in html
    assert "&lt;img" in html


def test_unreadable_table_is_noted_not_fatal(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "lift_table.csv").write_bytes(b"\xff\xfe\x00binary garbage\x00")
    html = build_report_html(dataset_label="D", reports_dir=d)
    assert "Lift table" in html


def test_long_tables_are_truncated_with_a_note(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    rows = "\n".join(f"{i},{i * 2}" for i in range(500))
    (d / "lift_table.csv").write_text(f"a,b\n{rows}\n", encoding="utf-8")
    html = build_report_html(dataset_label="D", reports_dir=d)
    assert "of 500 rows" in html
