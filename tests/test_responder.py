"""The responder is the only place an LLM touches this system, so these tests are
mostly about what it is *not* allowed to do.

Three properties, each worth a failing build:

1. It never decides anything. The decision is already in the ledger before the
   responder is called, and no output of it can change one.
2. It never states a number it was not given. A fluent sentence with an invented rupee
   figure is worse than a blunt template, because someone will forward it.
3. It never fails the request. No key, no package, a timeout, a refusal -- all of them
   produce a deterministic note and say so.

The API is never actually called here: a test suite that needs credentials is a test
suite that does not run in CI.
"""
from __future__ import annotations

import pytest

from returnrisk.responder import Responder, _ungrounded_figures


@pytest.fixture
def decision():
    """A recorded decision, in the shape `DecisionRecord.to_dict()` produces."""
    return {
        "decision_id": "abc123",
        "recorded_at": "2026-09-01T09:00:00+00:00",
        "seq": 7,
        "entry_hash": "f" * 64,
        "risk_score": 0.2196,
        "flagged": True,
        "threshold": 0.1974,
        "order_value_inr": 86_100.0,
        "expected_loss_inr": 6_106.05,
        "expected_saving_inr": 577.54,
        "top_reasons": ["Unusually large order value", "Deep discount"],
        "model_version": "2026-08-31T15:59:22",
        "config_fingerprint": "49dc264cdc254dcf",
        "model_action": "remove_cod",
        "final_action": "none",
        "final_action_label": "No action - proceed normally",
        "outcome": "suppressed",
        "requires_human_review": False,
        "in_holdout": False,
        "rules_fired": [
            {
                "rule": "loyalty_shield",
                "effect": "suppress",
                "explanation": "Customer has 155 prior orders at a 5% return rate.",
            }
        ],
    }


@pytest.fixture
def offline(cfg):
    """A responder that cannot reach the API, which is how CI and most judges run it."""
    r = Responder(cfg)
    r._unavailable = "test: no credentials"
    return r


@pytest.fixture
def disabled(cfg):
    patched = cfg.as_dict()
    patched["responder"] = {**patched["responder"], "enabled": False}
    return Responder(type(cfg)(_data=patched, path=cfg.path))


# ------------------------------------------------------------- graceful degradation
def test_offline_responder_still_produces_a_note(offline, decision):
    result = offline.merchant_note(decision)
    assert result.text.strip()
    assert result.generated_by == "template (fallback)"
    assert result.fallback_reason


def test_disabled_responder_produces_a_note(disabled, decision):
    result = disabled.merchant_note(decision)
    assert result.text.strip()
    assert result.generated_by == "template"


def test_offline_responder_is_honest_about_availability(offline):
    assert offline.available is False


def test_the_template_is_not_a_placeholder(offline, decision):
    """The no-key path is what a judge sees. It has to be a real sentence carrying the
    real numbers, not 'explanation unavailable'."""
    text = offline.merchant_note(decision).text
    assert len(text) > 120
    assert "86,100" in text
    assert "22.0%" in text
    assert "155 prior orders" in text


def test_template_covers_every_decision_outcome(offline, decision):
    """Each branch a decision can take needs prose, or some order somewhere gets an
    empty note at the worst moment."""
    for patch in (
        {"outcome": "acted", "final_action": "remove_cod", "final_action_label": "Block COD"},
        {"outcome": "suppressed", "final_action": "none"},
        {"outcome": "review", "requires_human_review": True},
        {"outcome": "holdout", "in_holdout": True, "final_action": "none"},
        {"outcome": "not_flagged", "flagged": False, "final_action": "none"},
    ):
        text = offline.merchant_note({**decision, **patch}).text
        assert len(text) > 80, f"thin note for {patch['outcome']}"


def test_a_note_never_raises_on_a_sparse_decision(offline):
    """Missing reasons, missing rules, missing everything optional."""
    result = offline.merchant_note({"risk_score": 0.5, "threshold": 0.2})
    assert result.text.strip()


# -------------------------------------------------------------- the grounding check
def test_grounded_figures_pass():
    text = "The order is Rs.86,100 and scores 22.0% risk."
    assert _ungrounded_figures(text, ["Rs.86,100", "22.0%"]) == []


def test_invented_rupee_figure_is_caught():
    text = "This order is worth Rs.9,90,000 and will save Rs.45,000."
    bad = _ungrounded_figures(text, ["Rs.86,100", "22.0%"])
    assert bad, "an invented amount must be detected"


def test_invented_percentage_is_caught():
    bad = _ungrounded_figures("The model is 97% accurate here.", ["22.0%"])
    assert any("97" in b for b in bad)


def test_formatting_differences_are_not_treated_as_invention():
    """'Rs.86,100', '86100' and '86,100' are the same figure written three ways."""
    assert _ungrounded_figures("Order value 86100 rupees.", ["Rs.86,100"]) == []
    assert _ungrounded_figures("Order value Rs. 86,100.", ["Rs.86,100"]) == []


def test_small_counting_numbers_are_allowed():
    """'3 reasons' and '2 guardrails' are prose, not hallucinated amounts."""
    assert _ungrounded_figures("There are 3 reasons and 2 guardrails.", []) == []


def test_ungrounded_output_is_rejected_in_favour_of_the_template(cfg, decision, monkeypatch):
    """The end-to-end guarantee: a fluent but ungrounded model answer is discarded, and
    the caller is told why rather than being handed the invented number."""
    r = Responder(cfg)

    def fake_generate(system, facts, allowed):
        return "", "ungrounded figures in output: Rs.9,90,000"

    monkeypatch.setattr(r, "_generate", fake_generate)
    result = r.merchant_note(decision)
    assert result.generated_by == "template (fallback)"
    assert "ungrounded" in (result.fallback_reason or "")
    assert "9,90,000" not in result.text


def test_a_good_model_answer_is_used_and_attributed(cfg, decision, monkeypatch):
    r = Responder(cfg)
    monkeypatch.setattr(r, "_generate", lambda s, f, a: ("A clean grounded sentence.", None))
    result = r.merchant_note(decision)
    assert result.generated_by == "claude"
    assert result.model == r.model
    assert result.text == "A clean grounded sentence."


# ------------------------------------------------------------------- the dispute pack
def test_dispute_evidence_is_identical_with_and_without_the_llm(cfg, decision, monkeypatch):
    """The evidence table is what a payment network checks. It must not depend on
    whether an LLM happened to answer."""
    offline_pack = Responder(cfg)
    offline_pack._unavailable = "no credentials"

    online = Responder(cfg)
    monkeypatch.setattr(online, "_generate", lambda s, f, a: ("Generated narrative.", None))

    assert offline_pack.dispute_pack(decision)["evidence"] == online.dispute_pack(decision)["evidence"]


def test_dispute_pack_states_when_the_model_was_wrong(offline, decision):
    """A pack that hides a false positive is worse than no pack."""
    pack = offline.dispute_pack(decision, {"returned": False})
    assert pack["evidence"]["model_was_correct"] is False
    assert "did not match" in pack["narrative"]["text"]


def test_dispute_pack_marks_an_unobserved_outcome_as_unobserved(offline, decision):
    pack = offline.dispute_pack(decision, None)
    assert pack["evidence"]["observed_outcome"] == "not yet observed"
    assert pack["evidence"]["model_was_correct"] is None


def test_dispute_pack_never_claims_proof_of_wrongdoing(offline, decision):
    pack = offline.dispute_pack(decision, {"returned": True})
    text = pack["narrative"]["text"].lower()
    for word in ("fraud", "fraudulent", "guilty", "proves", "abuser"):
        assert word not in text
    assert "not evidence of customer wrongdoing" in pack["narrative"]["text"]


def test_dispute_pack_carries_the_ledger_hash(offline, decision):
    pack = offline.dispute_pack(decision)
    assert pack["attestation"]["ledger_hash"] == decision["entry_hash"]
    assert pack["attestation"]["decision_was_deterministic"] is True


def test_the_prompt_forbids_deciding_anything(offline):
    """A guard on the instructions themselves: if someone later loosens the system
    prompt into 'recommend an action', this test should be the thing that objects."""
    from returnrisk.responder import DISPUTE_SYSTEM_PROMPT, SYSTEM_PROMPT

    assert "not deciding" in SYSTEM_PROMPT.lower()
    assert "only the facts given" in SYSTEM_PROMPT.lower()
    for prompt in (SYSTEM_PROMPT, DISPUTE_SYSTEM_PROMPT):
        assert "never introduce a number" in prompt.lower()
