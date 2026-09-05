"""The governed decision path: score, gate, record.

`ScoringService.score` answers a statistical question and is deliberately left alone --
it is what the offline evaluation measures, and it must keep meaning exactly what the
README says it means. This module wraps it in the two things a production risk decision
also needs:

    score  ->  policy guardrails  ->  append-only ledger  ->  DecisionRecord

Keeping this separate from `scoring.py` is the point. The metrics in `reports/` describe
the model; the ledger describes what the *system* did, and those two are allowed to
differ -- a guardrail suppressing an action is a case where they should. Collapsing them
into one call would quietly make the reported savings unachievable, because they would
no longer account for the orders policy refuses to act on.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audit import DecisionLedger, config_fingerprint, fingerprint
from .config import Config, load_config
from .money import CostModel
from .policy import PolicyDecision, PolicyEngine
from .scoring import OrderInput, ScoreResult, ScoringService

#: The order fields that define "the same basket". Timestamp and customer are excluded
#: from the *holdout* fingerprint on purpose -- see `DecisionService._fingerprint`.
FINGERPRINT_FIELDS = (
    "order_value_gbp",
    "n_lines",
    "total_quantity",
    "avg_unit_price",
    "max_unit_price",
    "min_unit_price",
    "country",
    "top_category",
    "discount_pct",
    "customer_id",
)


@dataclass
class DecisionRecord:
    """One governed decision: the score, the gate, and the receipt."""

    score: ScoreResult
    policy: PolicyDecision
    order_value_inr: float
    decision_id: str
    seq: int
    entry_hash: str
    recorded_at: str
    config_fingerprint: str
    order_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Flat, because this is both the API response and the ledger row."""
        s = self.score
        return {
            "decision_id": self.decision_id,
            "seq": self.seq,
            "entry_hash": self.entry_hash,
            "recorded_at": self.recorded_at,
            "risk_score": round(s.risk_score, 6),
            "flagged": s.flagged,
            "threshold": round(s.threshold, 6),
            "order_value_inr": round(self.order_value_inr, 2),
            "expected_loss_inr": round(s.expected_loss_inr, 2),
            "expected_saving_inr": round(s.expected_saving_inr, 2),
            "top_reasons": s.top_reasons,
            "reason_detail": s.reason_detail,
            "model_version": s.model_version,
            "config_fingerprint": self.config_fingerprint,
            "order_fingerprint": self.order_fingerprint,
            **self.policy.to_dict(),
        }


class DecisionService:
    """Composes scoring, policy and the ledger into the call an integrator makes.

    Built from an already-loaded `ScoringService` so the API keeps its single cold start
    -- the model is loaded once at boot, not once per decision.
    """

    def __init__(
        self,
        scoring: ScoringService,
        cfg: Config | None = None,
        ledger: DecisionLedger | None = None,
    ) -> None:
        self.cfg = cfg or scoring.cfg
        self.scoring = scoring
        self.policy = PolicyEngine(self.cfg)
        self.ledger = ledger if ledger is not None else DecisionLedger.from_config(self.cfg)
        self.cost = CostModel(self.cfg)
        self._config_fp = config_fingerprint(self.cfg)

    @classmethod
    def load(cls, cfg: Config | None = None) -> "DecisionService":
        cfg = cfg or load_config()
        return cls(ScoringService.load(cfg), cfg)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(order: OrderInput) -> str:
        """Identity of the basket, for holdout membership and duplicate spotting.

        Includes `customer_id` so the holdout is per customer-basket rather than per
        basket shape; excludes the timestamp so a retry of the same order lands in the
        same holdout bucket instead of re-rolling the dice.
        """
        payload = {f: getattr(order, f, None) for f in FINGERPRINT_FIELDS}
        return fingerprint(payload)

    def _profile_for(self, order: OrderInput) -> dict[str, Any]:
        """The customer history the policy layer reasons about.

        Reuses the scoring service's own lookup so policy and model never see different
        views of the same customer -- a disagreement there would be invisible and
        extremely annoying to debug.
        """
        as_of = order.order_date or dt.datetime.now()
        profile = self.scoring.lookup_profile(order.customer_id, as_of)
        for key in ("prior_orders", "prior_return_rate"):
            override = getattr(order, key, None)
            if override is not None:
                profile[key] = override
        return profile

    # -- the entry point ----------------------------------------------------------
    def decide(
        self,
        order: OrderInput,
        action: str | None = None,
        record: bool = True,
    ) -> DecisionRecord:
        """Score an order, gate it, and write the receipt.

        `record=False` exists for the dashboard's what-if slider, which needs to show
        what *would* happen without filling the ledger with hypotheticals. A decision
        that is not recorded is not a decision anybody may act on, and the API never
        uses it.
        """
        order = order.defaults()
        score = self.scoring.score(order, action=action)
        value_inr = float(self.cost.to_inr(np.array([order.order_value_gbp]))[0])
        profile = self._profile_for(order)

        prior_rate = profile.get("prior_return_rate")
        if prior_rate is not None and isinstance(prior_rate, float) and np.isnan(prior_rate):
            prior_rate = None

        fp = self._fingerprint(order)
        verdict = self.policy.apply(
            model_action=score.recommended_action,
            flagged=score.flagged,
            order_value_inr=value_inr,
            prior_orders=int(profile.get("prior_orders", 0) or 0),
            prior_return_rate=None if prior_rate is None else float(prior_rate),
            is_new_customer=int(profile.get("prior_orders", 0) or 0) == 0,
            fingerprint=fp,
            current_action_rate=self.ledger.action_rate() if record else None,
        )

        body = {
            "risk_score": round(score.risk_score, 6),
            "flagged": bool(score.flagged),
            "threshold": round(score.threshold, 6),
            "order_value_inr": round(value_inr, 2),
            "expected_loss_inr": round(score.expected_loss_inr, 2),
            "expected_saving_inr": round(score.expected_saving_inr, 2),
            "top_reasons": list(score.top_reasons),
            "model_version": score.model_version,
            "config_fingerprint": self._config_fp,
            "order_fingerprint": fp,
            **verdict.to_dict(),
        }

        if record:
            entry = self.ledger.append(body)
            decision_id, seq, entry_hash, recorded_at = (
                entry.decision_id,
                entry.seq,
                entry.entry_hash,
                entry.recorded_at,
            )
        else:
            decision_id, seq, entry_hash = "not-recorded", -1, ""
            recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()

        return DecisionRecord(
            score=score,
            policy=verdict,
            order_value_inr=value_inr,
            decision_id=decision_id,
            seq=seq,
            entry_hash=entry_hash,
            recorded_at=recorded_at,
            config_fingerprint=self._config_fp,
            order_fingerprint=fp,
        )


__all__ = ["FINGERPRINT_FIELDS", "DecisionRecord", "DecisionService"]
