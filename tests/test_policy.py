"""The guardrails are the part that spends -- or refuses to spend -- the merchant's
goodwill, so each rule gets a test that would fail if the rule silently stopped firing.

A guardrail nobody has watched fire is indistinguishable from a guardrail that does
nothing, which is the same argument `test_leakage.py` makes about the leakage guard.
"""
from __future__ import annotations

import pytest

from returnrisk.policy import PolicyEngine


@pytest.fixture
def engine(cfg):
    return PolicyEngine(cfg)


def _outside_holdout(engine) -> str:
    """A fingerprint that is definitely *not* in the holdout.

    Found rather than hardcoded: a literal string is a coin flip against the 5% draw,
    and a baseline that silently lands in the holdout would make every other rule's
    test pass or fail for the wrong reason.
    """
    return next(f"baseline-{i}" for i in range(1000) if not engine.is_holdout(f"baseline-{i}"))


def _apply(engine, **overrides):
    """A flagged, mid-size order from an unremarkable repeat customer.

    Every test overrides exactly the field its rule cares about, so a failure names the
    rule that broke rather than "something in policy".
    """
    kwargs = dict(
        model_action="remove_cod",
        flagged=True,
        order_value_inr=50_000.0,
        prior_orders=2,
        prior_return_rate=0.5,
        is_new_customer=False,
        fingerprint=_outside_holdout(engine),
        current_action_rate=0.0,
    )
    kwargs.update(overrides)
    return engine.apply(**kwargs)


def _rules(decision) -> set[str]:
    return {r.rule for r in decision.rules_fired}


# --------------------------------------------------------------------- the base case
def test_unflagged_order_is_never_actioned(engine):
    d = _apply(engine, flagged=False)
    assert d.final_action == "none"
    assert d.outcome == "not_flagged"
    assert d.rules_fired == []


def test_ordinary_flagged_order_is_actioned_as_the_model_asked(engine):
    d = _apply(engine)
    assert d.final_action == "remove_cod"
    assert d.outcome == "acted"
    assert d.acted is True


# ------------------------------------------------------------------ rule 1: value floor
def test_tiny_order_is_suppressed(engine, cfg):
    floor = float(cfg["policy"]["min_order_value_inr"])
    d = _apply(engine, order_value_inr=floor - 1)
    assert d.final_action == "none"
    assert d.outcome == "suppressed"
    assert "min_order_value" in _rules(d)


def test_the_floor_is_inclusive_at_the_boundary(engine, cfg):
    """An order exactly at the floor is worth acting on -- the rule is `<`, not `<=`,
    and a boundary that drifts is a boundary nobody notices drifting."""
    floor = float(cfg["policy"]["min_order_value_inr"])
    assert _apply(engine, order_value_inr=floor).outcome == "acted"


# --------------------------------------------------------------- rule 2: loyalty shield
def test_loyal_customer_is_protected_however_high_the_score(engine, cfg):
    shield = cfg["policy"]["loyalty_shield"]
    d = _apply(
        engine,
        prior_orders=int(shield["min_prior_orders"]) + 10,
        prior_return_rate=float(shield["max_prior_return_rate"]) / 2,
    )
    assert d.final_action == "none"
    assert "loyalty_shield" in _rules(d)


def test_loyal_but_returny_customer_is_not_protected(engine, cfg):
    """Long tenure alone is not the shield -- a frequent returner still gets actioned."""
    shield = cfg["policy"]["loyalty_shield"]
    d = _apply(
        engine,
        prior_orders=int(shield["min_prior_orders"]) + 10,
        prior_return_rate=float(shield["max_prior_return_rate"]) + 0.2,
    )
    assert "loyalty_shield" not in _rules(d)
    assert d.final_action == "remove_cod"


def test_unknown_history_does_not_earn_the_shield(engine):
    """A customer we have never seen must not be protected by an absent return rate."""
    d = _apply(engine, prior_orders=99, prior_return_rate=None)
    assert "loyalty_shield" not in _rules(d)


# ------------------------------------------------------------ rule 3: new-customer cap
def test_new_customer_action_is_downgraded_not_suppressed(engine, cfg):
    """The model card names new customers as the weakest segment. The system must not
    be able to hit them with the harshest action; it may still do something gentle."""
    d = _apply(engine, is_new_customer=True, prior_orders=0, prior_return_rate=None)
    cap = cfg["policy"]["new_customer_max_action"]
    assert d.final_action == cap
    assert "new_customer_cap" in _rules(d)
    assert d.final_action != "none", "a downgrade is not a suppression"


def test_new_customer_cap_does_not_upgrade_a_gentler_action(engine):
    """The cap is a ceiling, never a floor -- a pack check stays a pack check."""
    d = _apply(
        engine,
        model_action="manual_pack_check",
        is_new_customer=True,
        prior_orders=0,
        prior_return_rate=None,
    )
    assert d.final_action == "manual_pack_check"
    assert "new_customer_cap" not in _rules(d)


# --------------------------------------------------------------- rule 4: human review
def test_large_order_goes_to_a_human(engine, cfg):
    limit = float(cfg["policy"]["human_review_above_inr"])
    d = _apply(engine, order_value_inr=limit + 1)
    assert d.requires_human_review is True
    assert d.outcome == "review"
    assert d.acted is False, "queued for review is not the same as actioned"
    assert "human_review" in _rules(d)


# ------------------------------------------------------------------ rule 5: daily cap
def test_action_rate_at_the_cap_escalates_instead_of_acting(engine, cfg):
    cap = float(cfg["policy"]["max_daily_action_rate"])
    d = _apply(engine, current_action_rate=cap)
    assert d.requires_human_review is True
    assert "daily_action_cap" in _rules(d)


def test_action_rate_below_the_cap_acts_normally(engine, cfg):
    cap = float(cfg["policy"]["max_daily_action_rate"])
    assert _apply(engine, current_action_rate=cap - 0.01).outcome == "acted"


# -------------------------------------------------------------------- rule 6: holdout
def test_holdout_membership_is_deterministic(engine):
    """Same order, same answer, always. A random draw would mean a retry changes the
    decision -- an audit problem and a way to shop for a better answer."""
    picks = [engine.is_holdout(f"order-{i}") for i in range(200)]
    again = [engine.is_holdout(f"order-{i}") for i in range(200)]
    assert picks == again


def test_holdout_rate_is_close_to_the_configured_fraction(engine, cfg):
    frac = float(cfg["policy"]["holdout_frac"])
    n = 20_000
    hits = sum(engine.is_holdout(f"order-{i}") for i in range(n))
    assert abs(hits / n - frac) < 0.01, "holdout draw is not uniform enough to trust"


def test_a_holdout_order_is_let_through_unactioned(engine):
    fingerprint = next(
        f"order-{i}" for i in range(5000) if engine.is_holdout(f"order-{i}")
    )
    d = _apply(engine, fingerprint=fingerprint)
    assert d.in_holdout is True
    assert d.final_action == "none"
    assert "selective_labels_holdout" in _rules(d)


# ---------------------------------------------------------------------- ordering rules
def test_severity_order_is_derived_from_cost_not_hardcoded(engine):
    """Severity comes from each action's conversion loss, so a new intervention
    declared in YAML slots into the ordering without a code change."""
    order = engine.severity_order
    losses = [engine.cost.actions[k].conversion_loss_prob for k in order]
    assert losses == sorted(losses)
    assert engine.severity("manual_pack_check") < engine.severity("remove_cod")


def test_suppression_beats_downgrade(engine, cfg):
    """A new customer who is also loyalty-shielded must end up suppressed, not merely
    downgraded -- the stronger protection has to win."""
    shield = cfg["policy"]["loyalty_shield"]
    d = _apply(
        engine,
        is_new_customer=True,
        prior_orders=int(shield["min_prior_orders"]) + 1,
        prior_return_rate=0.0,
    )
    assert d.final_action == "none"


def test_disabling_policy_passes_the_model_action_through(cfg):
    """The escape hatch has to actually be an escape hatch: with guardrails off, the
    system does exactly what the model said and says so."""
    patched = cfg.as_dict()
    patched["policy"] = {**patched["policy"], "enabled": False}
    engine = PolicyEngine(type(cfg)(_data=patched, path=cfg.path))
    d = _apply(engine, order_value_inr=1.0, prior_orders=999, prior_return_rate=0.0)
    assert d.final_action == "remove_cod"
    assert d.rules_fired == []


def test_every_rule_that_fires_carries_a_readable_explanation(engine, cfg):
    """Guardrails end up in a merchant's audit trail. A rule that fires without saying
    why in words is a rule that will be argued about later."""
    cases = [
        dict(order_value_inr=1.0),
        dict(prior_orders=99, prior_return_rate=0.0),
        dict(is_new_customer=True, prior_orders=0, prior_return_rate=None),
        dict(order_value_inr=float(cfg["policy"]["human_review_above_inr"]) + 1),
        dict(current_action_rate=1.0),
    ]
    for case in cases:
        for hit in _apply(engine, **case).rules_fired:
            assert hit.explanation.strip(), f"{hit.rule} fired with no explanation"
            assert len(hit.explanation) > 25, f"{hit.rule} explanation is too terse"
            assert hit.effect in {"suppress", "downgrade", "review", "holdout"}
