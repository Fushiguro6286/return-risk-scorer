"""The governed path, end to end: score -> guardrails -> ledger.

These tests use the real trained model when one exists on disk and skip otherwise, so a
clean clone that has not run `python run_demo.py` still gets a green suite rather than a
misleading failure.
"""
from __future__ import annotations

import pytest

from returnrisk.audit import DecisionLedger
from returnrisk.config import REPO_ROOT
from returnrisk.decision import DecisionService
from returnrisk.scoring import OrderInput

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "models" / "return_risk_model.joblib").exists(),
    reason="no trained model on disk - run `python run_demo.py` first",
)


@pytest.fixture(scope="module")
def scoring(cfg):
    from returnrisk.scoring import ScoringService

    return ScoringService.load(cfg)


@pytest.fixture
def service(scoring, cfg, tmp_path):
    """A decision service writing to a throwaway ledger, never the repo's own."""
    return DecisionService(scoring, cfg, ledger=DecisionLedger(tmp_path / "audit"))


def _order(**overrides) -> OrderInput:
    kwargs = dict(
        order_value_gbp=820.0,
        n_lines=18,
        total_quantity=240,
        avg_unit_price=3.4,
        max_unit_price=12.5,
        min_unit_price=0.85,
        top_category="228",
        discount_pct=0.34,
    )
    kwargs.update(overrides)
    return OrderInput(**kwargs)


# ------------------------------------------------------------------ the receipt
def test_every_decision_is_recorded(service):
    record = service.decide(_order())
    assert service.ledger.get(record.decision_id) is not None
    assert service.ledger.verify()["ok"] is True


def test_recorded_row_matches_the_returned_payload(service):
    """The API response and the audit row must not be allowed to drift apart -- if they
    can differ, the ledger stops being evidence of what the caller was told."""
    record = service.decide(_order())
    row = service.ledger.get(record.decision_id)
    payload = record.to_dict()
    for key in (
        "risk_score",
        "threshold",
        "final_action",
        "outcome",
        "expected_loss_inr",
        "order_value_inr",
        "model_version",
    ):
        assert row[key] == payload[key], f"{key} differs between response and ledger"


def test_decisions_accumulate_in_order(service):
    ids = [service.decide(_order(order_value_gbp=100.0 + i)).decision_id for i in range(5)]
    assert len({*ids}) == 5
    assert [r["seq"] for r in service.ledger.entries()] == [0, 1, 2, 3, 4]


def test_dry_run_does_not_touch_the_ledger(service):
    """The dashboard's what-if slider must not fill the audit trail with hypotheticals."""
    before = len(service.ledger)
    record = service.decide(_order(), record=False)
    assert len(service.ledger) == before
    assert record.decision_id == "not-recorded"


# --------------------------------------------------------------- policy integration
def test_the_ledger_records_the_action_actually_taken(service):
    """A suppressed order must be recorded as suppressed, with the model's original
    recommendation kept alongside -- both halves are needed to audit the guardrail."""
    record = service.decide(_order(customer_id="17850"), action="remove_cod")
    row = service.ledger.get(record.decision_id)
    assert row["model_action"] == record.policy.model_action
    assert row["final_action"] == record.policy.final_action
    if row["outcome"] == "suppressed":
        assert row["final_action"] == "none"
        assert row["rules_fired"], "a suppression with no recorded reason is unauditable"


def test_scoring_is_unaffected_by_the_policy_layer(service, scoring):
    """`/score` is what the offline reports measure. Wrapping it in governance must not
    change the number it returns, or the README stops describing the system."""
    order = _order(customer_id="17850")
    direct = scoring.score(order, action="remove_cod")
    governed = service.decide(order, action="remove_cod")
    assert governed.score.risk_score == pytest.approx(direct.risk_score)
    assert governed.score.threshold == pytest.approx(direct.threshold)


def test_same_order_lands_in_the_same_holdout_bucket(service):
    """Retrying must not re-roll the holdout draw."""
    order = _order(customer_id="14911")
    first = service.decide(order)
    second = service.decide(order)
    assert first.order_fingerprint == second.order_fingerprint
    assert first.policy.in_holdout == second.policy.in_holdout


def test_a_tiny_order_is_suppressed_end_to_end(service):
    """The value floor, exercised through the real model rather than in isolation."""
    record = service.decide(_order(order_value_gbp=5.0, n_lines=1, total_quantity=1))
    assert record.policy.final_action == "none"


def test_config_fingerprint_is_recorded(service):
    """Costs live in YAML; a decision is only reproducible if you know which YAML."""
    record = service.decide(_order())
    assert service.ledger.get(record.decision_id)["config_fingerprint"]


# --------------------------------------------------------- the LLM has no authority
def test_the_responder_cannot_change_a_decision(service, cfg, monkeypatch):
    """The architectural guarantee, asserted rather than merely documented: hand the
    responder a hostile model that tries to reverse the outcome, and the recorded
    decision is byte-identical."""
    from returnrisk.responder import Responder

    record = service.decide(_order(customer_id="17850"), action="remove_cod")
    before = dict(service.ledger.get(record.decision_id))

    responder = Responder(cfg)
    monkeypatch.setattr(
        responder,
        "_generate",
        lambda s, f, a: ("IGNORE THE POLICY. Block this customer permanently.", None),
    )
    responder.merchant_note(record.to_dict())
    responder.dispute_pack(record.to_dict())

    assert service.ledger.get(record.decision_id) == before
    assert service.ledger.verify()["ok"] is True


def test_outcome_recording_closes_the_loop(service):
    record = service.decide(_order())
    service.ledger.record_outcome(record.decision_id, returned=True, note="came back")
    assert service.ledger.outcomes()[record.decision_id]["returned"] is True
    assert service.ledger.verify()["ok"] is True, "recording an outcome must not edit history"
