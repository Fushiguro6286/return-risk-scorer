"""Policy guardrails: the model recommends, this layer decides.

A calibrated score and a cost-optimal threshold answer "is this order risky enough to
be worth intervening on *on average*". They do not answer "may we intervene on *this*
order, right now, given everything else we have already done today". That second
question is the one an operations team actually has to answer, and it is the one the
brief is asking about when it says every money action must be **bounded and gated**.

So the serving path is deliberately two stages:

    model  ->  recommended_action        (statistics: what usually pays off)
    policy ->  final_action + reasons    (governance: what we are allowed to do)

Six rules, all declared in `config.yaml`, each able to suppress, downgrade, or escalate
the model's recommendation. Every rule that fires records a merchant-readable sentence,
and those sentences go into the audit ledger next to the score -- so a decision can be
reconstructed months later without rerunning the model.

Two of the rules deserve their own note, because they are where the model card's
admissions become enforced code rather than prose:

* **new_customer_max_action** -- new customers are the model's weakest segment (1.48x
  lift) and the segment a COD block hurts most. The model card says so; this rule means
  the system cannot act as if it did not.
* **holdout** -- README section 9 argues that acting on every flagged order destroys the
  labels you need to retrain. A deterministic slice of flagged orders is therefore let
  through unactioned. Arguing for it in a README is cheap; doing it in the serving path
  costs real money and is the only version that counts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .money import ActionSpec, CostModel

#: A rule may do one of four things to the model's recommendation.
SUPPRESS = "suppress"      # take no action at all
DOWNGRADE = "downgrade"    # take a gentler action than recommended
ESCALATE = "review"        # hand to a human instead of acting automatically
HOLDOUT = "holdout"        # deliberately unactioned, to keep the labels honest


@dataclass(frozen=True)
class RuleHit:
    """One guardrail firing, in the words a merchant would want to read."""

    rule: str
    effect: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "effect": self.effect, "explanation": self.explanation}


@dataclass
class PolicyDecision:
    """The final, governed decision -- what actually happens to the order."""

    model_action: str
    final_action: str
    final_action_label: str
    outcome: str
    requires_human_review: bool
    in_holdout: bool
    rules_fired: list[RuleHit] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return self.final_action != "none" and not self.requires_human_review

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_action": self.model_action,
            "final_action": self.final_action,
            "final_action_label": self.final_action_label,
            "outcome": self.outcome,
            "requires_human_review": self.requires_human_review,
            "in_holdout": self.in_holdout,
            "rules_fired": [r.to_dict() for r in self.rules_fired],
        }


class PolicyEngine:
    """Applies the config-declared guardrails to one model recommendation.

    Stateless except for the portfolio cap, which needs to know what share of today's
    orders already carry an action -- the caller passes that in (the audit ledger keeps
    it), so the engine itself stays a pure function and is trivial to test.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cost = CostModel(cfg)
        self._p: dict[str, Any] = dict(cfg.get("policy") or {})
        self.enabled = bool(self._p.get("enabled", True))

    # -- action severity ---------------------------------------------------------
    @property
    def severity_order(self) -> list[str]:
        """Actions from gentlest to harshest, ordered by the friction they impose.

        Derived from the cost model rather than hardcoded, so a new intervention
        declared in YAML slots into the ordering without a code change.
        """
        return sorted(
            self.cost.actions,
            key=lambda k: (
                self.cost.actions[k].conversion_loss_prob,
                self.cost.actions[k].ops_cost_inr,
            ),
        )

    def severity(self, action: str) -> int:
        order = self.severity_order
        return order.index(action) if action in order else len(order)

    def _label(self, action: str) -> str:
        if action == "none":
            return "No action - proceed normally"
        spec: ActionSpec = self.cost.resolve(action)
        return spec.label

    # -- the holdout -------------------------------------------------------------
    def is_holdout(self, fingerprint: str) -> bool:
        """Deterministic membership, so the same order is always treated the same way.

        A random draw would mean a retry flips the decision, which is both an audit
        problem and a way for a caller to shop for the answer they want.
        """
        frac = float(self._p.get("holdout_frac", 0.0) or 0.0)
        if frac <= 0:
            return False
        salt = str(self._p.get("holdout_salt", ""))
        digest = hashlib.sha256(f"{salt}:{fingerprint}".encode("utf-8")).digest()
        # First 8 bytes as a uniform [0, 1) draw.
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return draw < frac

    # -- the main entry point ----------------------------------------------------
    def apply(
        self,
        *,
        model_action: str,
        flagged: bool,
        order_value_inr: float,
        prior_orders: int,
        prior_return_rate: float | None,
        is_new_customer: bool,
        fingerprint: str,
        current_action_rate: float | None = None,
    ) -> PolicyDecision:
        """Run every guardrail and return the decision that will actually be executed."""
        hits: list[RuleHit] = []

        if not flagged or model_action == "none":
            return PolicyDecision(
                model_action="none",
                final_action="none",
                final_action_label=self._label("none"),
                outcome="not_flagged",
                requires_human_review=False,
                in_holdout=False,
            )

        if not self.enabled:
            return PolicyDecision(
                model_action=model_action,
                final_action=model_action,
                final_action_label=self._label(model_action),
                outcome="acted",
                requires_human_review=False,
                in_holdout=False,
            )

        action = model_action

        # (1) Below this value the friction costs more than the return it prevents.
        floor = float(self._p.get("min_order_value_inr", 0.0) or 0.0)
        if order_value_inr < floor:
            hits.append(
                RuleHit(
                    "min_order_value",
                    SUPPRESS,
                    f"Order is {order_value_inr:,.0f} rupees, below the "
                    f"{floor:,.0f} floor where intervening is worth the friction.",
                )
            )
            return self._final(model_action, "none", "suppressed", False, False, hits)

        # (2) A long clean history outranks a 25%-precision signal.
        shield = dict(self._p.get("loyalty_shield") or {})
        min_orders = int(shield.get("min_prior_orders", 0) or 0)
        max_rate = float(shield.get("max_prior_return_rate", 1.0))
        if (
            min_orders
            and prior_orders >= min_orders
            and prior_return_rate is not None
            and prior_return_rate <= max_rate
        ):
            hits.append(
                RuleHit(
                    "loyalty_shield",
                    SUPPRESS,
                    f"Customer has {prior_orders} prior orders at a "
                    f"{prior_return_rate:.0%} return rate - protected from friction "
                    f"regardless of score.",
                )
            )
            return self._final(model_action, "none", "suppressed", False, False, hits)

        # (3) Cap the severity on the segment the model is worst at.
        cap = self._p.get("new_customer_max_action")
        if is_new_customer and cap and self.severity(action) > self.severity(str(cap)):
            hits.append(
                RuleHit(
                    "new_customer_cap",
                    DOWNGRADE,
                    f"New customer: the model's weakest segment. "
                    f"'{self._label(action)}' downgraded to '{self._label(str(cap))}'.",
                )
            )
            action = str(cap)

        # (4) Big money is a human's call. Bounded autonomy, stated as a number.
        review_above = float(self._p.get("human_review_above_inr", float("inf")))
        if order_value_inr >= review_above:
            hits.append(
                RuleHit(
                    "human_review",
                    ESCALATE,
                    f"Order is {order_value_inr:,.0f} rupees, at or above the "
                    f"{review_above:,.0f} limit for automatic action - queued for a human.",
                )
            )
            return self._final(model_action, action, "review", True, False, hits)

        # (5) Portfolio cap. One drift event must not action the whole book.
        cap_rate = self._p.get("max_daily_action_rate")
        if cap_rate is not None and current_action_rate is not None:
            if float(current_action_rate) >= float(cap_rate):
                hits.append(
                    RuleHit(
                        "daily_action_cap",
                        ESCALATE,
                        f"{float(current_action_rate):.1%} of today's orders already "
                        f"carry an action, at the {float(cap_rate):.0%} cap - queued for "
                        f"a human rather than actioned automatically.",
                    )
                )
                return self._final(model_action, action, "review", True, False, hits)

        # (6) The always-approve holdout, so the labels stay unbiased.
        if self.is_holdout(fingerprint):
            hits.append(
                RuleHit(
                    "selective_labels_holdout",
                    HOLDOUT,
                    f"In the {float(self._p.get('holdout_frac', 0)):.0%} always-approve "
                    f"holdout: deliberately let through unactioned so the outcome stays "
                    f"observable and the model can be retrained on unbiased labels.",
                )
            )
            return self._final(model_action, "none", "holdout", False, True, hits)

        return self._final(model_action, action, "acted", False, False, hits)

    def _final(
        self,
        model_action: str,
        final_action: str,
        outcome: str,
        review: bool,
        holdout: bool,
        hits: list[RuleHit],
    ) -> PolicyDecision:
        return PolicyDecision(
            model_action=model_action,
            final_action=final_action,
            final_action_label=self._label(final_action),
            outcome=outcome,
            requires_human_review=review,
            in_holdout=holdout,
            rules_fired=hits,
        )


__all__ = [
    "DOWNGRADE",
    "ESCALATE",
    "HOLDOUT",
    "SUPPRESS",
    "PolicyDecision",
    "PolicyEngine",
    "RuleHit",
]
