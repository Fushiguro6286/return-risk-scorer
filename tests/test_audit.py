"""The ledger's whole claim is that a past decision cannot be quietly altered.

That claim is worth exactly as much as the test that tries to alter one, so most of
this file is tampering: edit a field, renumber a row, splice the chain, delete an
entry, and assert `verify()` names the break.
"""
from __future__ import annotations

import json

import pytest

from returnrisk.audit import GENESIS, DecisionLedger, fingerprint, record_hash


@pytest.fixture
def ledger(tmp_path):
    return DecisionLedger(tmp_path / "audit")


def _body(**overrides):
    body = {
        "risk_score": 0.31,
        "threshold": 0.12,
        "outcome": "acted",
        "final_action": "remove_discount",
        "requires_human_review": False,
        "expected_saving_inr": 420.0,
        "rules_fired": [],
    }
    body.update(overrides)
    return body


def _rewrite(ledger, index, mutate):
    """Edit one line of the ledger file, the way an insider actually would."""
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[index])
    mutate(row)
    lines[index] = json.dumps(row, sort_keys=True)
    ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ the basics
def test_empty_ledger_verifies(ledger):
    assert ledger.verify()["ok"] is True
    assert len(ledger) == 0


def test_append_returns_a_receipt(ledger):
    entry = ledger.append(_body())
    assert entry.seq == 0
    assert entry.prev_hash == GENESIS
    assert len(entry.entry_hash) == 64
    assert ledger.get(entry.decision_id) is not None


def test_entries_are_chained_in_order(ledger):
    entries = [ledger.append(_body(risk_score=i / 10)) for i in range(5)]
    rows = list(ledger.entries())
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    for prev, row in zip(entries, rows[1:]):
        assert row["prev_hash"] == prev.entry_hash
    assert ledger.verify()["ok"] is True


def test_decision_ids_are_unique(ledger):
    ids = {ledger.append(_body()).decision_id for _ in range(50)}
    assert len(ids) == 50


def test_ledger_survives_a_reopen(ledger, tmp_path):
    """A restart must continue the chain, not start a second one at seq 0."""
    first = ledger.append(_body())
    reopened = DecisionLedger(tmp_path / "audit")
    second = reopened.append(_body())
    assert second.seq == 1
    assert second.prev_hash == first.entry_hash
    assert reopened.verify()["ok"] is True


# -------------------------------------------------------------------------- tampering
def test_editing_a_field_breaks_the_chain(ledger):
    for _ in range(4):
        ledger.append(_body())
    _rewrite(ledger, 1, lambda r: r.update(expected_saving_inr=9_999_999.0))
    result = ledger.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 1
    assert "edited" in result["reason"]


def test_flipping_an_action_breaks_the_chain(ledger):
    """The realistic attack: quietly change what the system did, not what it saved."""
    ledger.append(_body())
    ledger.append(_body(final_action="remove_cod"))
    _rewrite(ledger, 1, lambda r: r.update(final_action="none", outcome="suppressed"))
    assert ledger.verify()["ok"] is False


def test_deleting_an_entry_breaks_the_chain(ledger):
    for _ in range(4):
        ledger.append(_body())
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    del lines[2]
    ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ledger.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 2


def test_rehashing_an_edited_row_still_breaks_the_next_link(ledger):
    """A sophisticated edit -- fix the row's own hash after changing it -- must still
    fail, because the following row's `prev_hash` no longer matches."""
    for _ in range(4):
        ledger.append(_body())

    def repair(row):
        row["expected_saving_inr"] = 1_234_567.0
        body = {k: v for k, v in row.items() if k != "entry_hash"}
        row["entry_hash"] = record_hash(body)

    _rewrite(ledger, 1, repair)
    result = ledger.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 2
    assert "prev_hash" in result["reason"]


def test_renumbering_is_caught(ledger):
    for _ in range(3):
        ledger.append(_body())
    _rewrite(ledger, 1, lambda r: r.update(seq=7))
    result = ledger.verify()
    assert result["ok"] is False
    assert "sequence" in result["reason"]


# --------------------------------------------------------------------------- outcomes
def test_outcomes_are_stored_separately_from_decisions(ledger):
    """Recording what happened must never edit the decision row."""
    entry = ledger.append(_body())
    before = ledger.path.read_text(encoding="utf-8")
    ledger.record_outcome(entry.decision_id, returned=True, note="came back")
    assert ledger.path.read_text(encoding="utf-8") == before
    assert ledger.verify()["ok"] is True
    assert ledger.outcomes()[entry.decision_id]["returned"] is True


def test_latest_outcome_wins(ledger):
    entry = ledger.append(_body())
    ledger.record_outcome(entry.decision_id, returned=False)
    ledger.record_outcome(entry.decision_id, returned=True, note="corrected")
    assert ledger.outcomes()[entry.decision_id]["returned"] is True


# ------------------------------------------------------------------------- aggregates
def test_action_rate_counts_only_automatic_actions(ledger):
    """Orders queued for a human are not actioned, and must not eat the daily budget."""
    ledger.append(_body(final_action="remove_cod"))
    ledger.append(_body(final_action="none", outcome="suppressed"))
    ledger.append(_body(final_action="remove_cod", requires_human_review=True))
    ledger.append(_body(final_action="none", outcome="holdout"))
    assert ledger.action_rate() == pytest.approx(0.25)


def test_stats_reports_guardrail_hits_and_chain_state(ledger):
    ledger.append(
        _body(
            final_action="none",
            outcome="suppressed",
            rules_fired=[{"rule": "loyalty_shield", "effect": "suppress", "explanation": "x"}],
        )
    )
    ledger.append(_body(final_action="remove_discount"))
    stats = ledger.stats()
    assert stats["n_decisions"] == 2
    assert stats["guardrails_fired"]["loyalty_shield"] == 1
    assert stats["by_outcome"]["suppressed"] == 1
    assert stats["chain"]["ok"] is True


def test_corrupt_line_does_not_crash_reading(ledger):
    """Reading a ledger with a torn write must degrade, not explode -- you read a
    ledger precisely when something has already gone wrong."""
    ledger.append(_body())
    with open(ledger.path, "a", encoding="utf-8") as fh:
        fh.write("{not json at all\n")
    assert len(list(ledger.entries())) == 1


# ----------------------------------------------------------------------- fingerprints
def test_fingerprint_is_stable_and_order_independent():
    a = fingerprint({"order_value_gbp": 100.0, "customer_id": "X"})
    b = fingerprint({"customer_id": "X", "order_value_gbp": 100.0})
    assert a == b


def test_fingerprint_changes_with_content():
    a = fingerprint({"order_value_gbp": 100.0})
    b = fingerprint({"order_value_gbp": 100.5})
    assert a != b
