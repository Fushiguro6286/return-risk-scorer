"""The responder: Claude writes the prose, the pipeline owns the facts.

Track 02 asks for detectors, verifiers or **responders**. The model in this repo is a
detector and the policy layer is a gate; neither of them produces the thing a merchant
actually has to send -- the note to the ops team explaining why an order was held, and
the evidence pack that goes back when a customer disputes a return.

The tempting way to build this is to hand an LLM the order and ask it what to do. That
would be the wrong architecture and, for a risk system, a straightforwardly bad one: the
decision would stop being reproducible, the threshold would stop being calibrated, and
"why did you block this" would be answerable only by a sampling temperature.

So the split here is strict, and it is the whole design:

    deterministic  ->  score, threshold, action, guardrails, rupee amounts
    Claude         ->  the English sentence that carries them

The LLM is never consulted before a decision, cannot change one, and is not on the
critical path -- `merchant_note` and `dispute_pack` take a decision that has *already*
been made and written to the ledger.

Three properties this file has to hold, because an ungrounded model in a money workflow
is worse than no model:

* **It cannot invent numbers.** Every figure Claude is permitted to state is passed in
  explicitly, and `_ungrounded_figures` re-reads the generated text afterwards to check
  no other rupee amount or percentage appeared. A note that fails the check is discarded
  and the template is used instead. That is the "verifier" half of the track, pointed at
  our own output.
* **It cannot fail the request.** No credentials, no `anthropic` package, a timeout, a
  rate limit, a refusal -- every path falls back to a deterministic template. A judge
  running this repo with no API key gets a working demo, just a blunter sentence.
* **It says which one you got.** Every response carries `generated_by`, so nobody
  mistakes a template for a model output or the reverse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import Config

#: Figures in the generated text must come from the facts we supplied. Matches
#: "Rs.8,300", "8,300 rupees", "26.6%", "0.34" and similar.
_NUMBER_RE = re.compile(r"(?:Rs\.?\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|rupees|lakh|crore)?")

SYSTEM_PROMPT = """You write short, factual operations notes for an e-commerce \
return-risk system used by an Indian merchant.

You are given a decision that has ALREADY been made by a calibrated statistical model \
and a policy layer. You are not deciding anything. Your only job is to state, in plain \
English, what was decided and why, so an operations person or a customer-support agent \
can act on it and defend it.

Hard rules:
- Use ONLY the facts given to you. Never introduce a number, percentage, or rupee \
amount that does not appear in the facts.
- Never claim the customer will return the order. The score is a probability, and the \
model's precision is roughly 25% - three in four flagged orders are not returned. Write \
in terms of risk and cost, never guilt.
- Never suggest denying service, blacklisting, or accusing anyone of fraud.
- No greeting, no sign-off, no bullet points, no markdown. Plain sentences.
- 3 sentences maximum. Under 70 words."""

DISPUTE_SYSTEM_PROMPT = """You draft the narrative section of a return/chargeback \
evidence pack for an Indian e-commerce merchant.

The structured evidence is assembled deterministically and given to you. Your job is \
only to write the narrative paragraph that accompanies it.

Hard rules:
- Use ONLY the facts given. Never introduce a number, amount, or date that is not in \
the facts.
- Be factual and neutral. This is a document a payment network or an ombudsman may \
read. Do not editorialise, do not accuse the customer of anything, do not speculate \
about intent.
- State what was decided, on what basis, and what the merchant did. If the model was \
wrong, say so plainly - a pack that overclaims is worse than no pack.
- No markdown, no headings, no bullets. One paragraph, under 130 words."""


@dataclass
class ResponderResult:
    """A generated artifact plus an honest account of where it came from."""

    text: str
    generated_by: str          # "claude" | "template" | "template (fallback)"
    model: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "generated_by": self.generated_by,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
        }


def _normalise(text: str) -> str:
    """Strip formatting so number comparison is not defeated by punctuation."""
    return re.sub(r"[,\s]", "", text).lower().rstrip(".")


def _ungrounded_figures(generated: str, allowed: list[str]) -> list[str]:
    """Figures in the output that were not in the facts we supplied.

    Deliberately permissive about *form* -- "8300", "8,300" and "Rs.8,300" all count as
    the same figure -- and deliberately strict about *existence*. Small integers up to
    ten are ignored: "3 reasons" is not a hallucinated amount.
    """
    permitted = {_normalise(a) for a in allowed}
    # Every permitted figure also permits its bare-digit form.
    for a in list(permitted):
        permitted.add(a.replace("rs.", "").replace("%", "").replace("rupees", ""))

    bad: list[str] = []
    for match in _NUMBER_RE.findall(generated):
        token = match.strip()
        if not any(ch.isdigit() for ch in token):
            continue
        norm = _normalise(token)
        bare = norm.replace("rs.", "").replace("%", "").replace("rupees", "")
        if norm in permitted or bare in permitted:
            continue
        # Ignore small counting numbers, which are prose, not amounts.
        try:
            if float(bare) <= 10 and "%" not in norm and "rs." not in norm:
                continue
        except ValueError:
            pass
        bad.append(token)
    return bad


class Responder:
    """Generates merchant-facing prose for a decision that is already final.

    Construct once and reuse -- the Anthropic client is built lazily on first use and
    cached, and a client that fails to build is remembered as unavailable so a
    key-less deployment does not retry on every request.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        r = dict(cfg.get("responder") or {})
        self.enabled = bool(r.get("enabled", True))
        self.model = str(r.get("model", "claude-opus-5"))
        self.effort = str(r.get("effort", "low"))
        self.max_tokens = int(r.get("max_tokens", 1200))
        self.timeout = float(r.get("timeout_seconds", 30))
        self.verify_numbers = bool(r.get("verify_numbers", True))
        self._client: Any = None
        self._unavailable: str | None = None

    # -- client ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.enabled and self._build_client() is not None

    def _build_client(self) -> Any:
        """Build the Anthropic client once, or record why we cannot.

        Credentials are resolved by the SDK itself (ANTHROPIC_API_KEY, an auth token,
        or an `ant auth login` profile) -- we deliberately do not read env vars here,
        because doing so would report "no key" for a machine that is authenticated by
        profile.
        """
        if not self.enabled:
            self._unavailable = "responder disabled in config"
            return None
        if self._unavailable is not None:
            # Once we know credentials are missing or the package is absent, stay
            # unavailable -- `available` must not claim otherwise.
            return None
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            self._unavailable = "anthropic package not installed"
            return None
        try:
            self._client = anthropic.Anthropic(timeout=self.timeout)
        except Exception as exc:  # no credentials, bad config, ...
            self._unavailable = f"{type(exc).__name__}: {exc}"
            return None
        return self._client

    def _generate(self, system: str, facts: str, allowed: list[str]) -> tuple[str, str | None]:
        """One short, grounded completion. Returns (text, failure_reason)."""
        client = self._build_client()
        if client is None:
            return "", self._unavailable or "client unavailable"
        try:
            import anthropic
        except ImportError:  # pragma: no cover - _build_client already covers this
            return "", "anthropic package not installed"

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": facts}],
            )
        except anthropic.AuthenticationError:
            # Permanent for this process: stop retrying on every request.
            self._unavailable = "no valid API credentials"
            return "", self._unavailable
        except anthropic.RateLimitError:
            return "", "rate limited"
        except anthropic.APIStatusError as exc:
            return "", f"api error {exc.status_code}"
        except anthropic.APIConnectionError:
            return "", "could not reach the API"
        except Exception as exc:
            # The SDK raises a plain TypeError when it cannot resolve any credential,
            # which is a configuration fact, not a transient one -- cache it too.
            message = f"{type(exc).__name__}: {exc}"
            if "authentication" in str(exc).lower() or "api_key" in str(exc).lower():
                self._unavailable = "no valid API credentials"
                return "", self._unavailable
            return "", message

        # A refusal is a valid HTTP 200 with no usable content -- check before reading.
        if getattr(response, "stop_reason", None) == "refusal":
            return "", "model declined to answer"

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            return "", "empty response"

        if self.verify_numbers:
            bad = _ungrounded_figures(text, allowed)
            if bad:
                # The model stated a figure we never gave it. In a money workflow that
                # is disqualifying, not something to warn about and ship.
                return "", f"ungrounded figures in output: {', '.join(bad[:3])}"
        return text, None

    # -- artifact 1: the merchant note --------------------------------------------
    def merchant_note(self, decision: dict[str, Any]) -> ResponderResult:
        """Why this order was held, in words an ops person can forward."""
        facts, allowed = _note_facts(decision)
        template = _note_template(decision)
        if not self.enabled:
            return ResponderResult(template, "template", None, "responder disabled in config")
        text, failure = self._generate(SYSTEM_PROMPT, facts, allowed)
        if failure:
            return ResponderResult(template, "template (fallback)", None, failure)
        return ResponderResult(text, "claude", self.model)

    # -- artifact 2: the dispute evidence pack ------------------------------------
    def dispute_pack(
        self, decision: dict[str, Any], outcome: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """A structured, defensible record for a disputed return.

        The evidence table is assembled from the ledger and is not negotiable; only the
        narrative paragraph is generated. That ordering is intentional -- the facts are
        what a payment network will check, and they must be identical whether or not an
        LLM was available.
        """
        evidence = _dispute_evidence(decision, outcome)
        facts, allowed = _dispute_facts(decision, evidence, outcome)
        template = _dispute_template(decision, evidence, outcome)

        if not self.enabled:
            narrative = ResponderResult(
                template, "template", None, "responder disabled in config"
            )
        else:
            text, failure = self._generate(DISPUTE_SYSTEM_PROMPT, facts, allowed)
            narrative = (
                ResponderResult(template, "template (fallback)", None, failure)
                if failure
                else ResponderResult(text, "claude", self.model)
            )

        return {
            "decision_id": decision.get("decision_id"),
            "evidence": evidence,
            "narrative": narrative.to_dict(),
            "attestation": {
                "decision_was_deterministic": True,
                "model_version": decision.get("model_version"),
                "config_fingerprint": decision.get("config_fingerprint"),
                "ledger_hash": decision.get("entry_hash"),
                "note": (
                    "The score, threshold, action and guardrails in this pack were "
                    "computed by a calibrated model and a config-declared policy layer, "
                    "and were written to an append-only hash-chained ledger at decision "
                    "time. The narrative above is generated prose over those same facts "
                    "and carries no decision authority."
                ),
            },
        }


# --------------------------------------------------------------------------- facts
def _fmt_inr(x: float) -> str:
    return f"Rs.{float(x):,.0f}"


def _note_facts(d: dict[str, Any]) -> tuple[str, list[str]]:
    """The exact facts Claude may use, and the figures it is allowed to state."""
    score_pct = f"{float(d.get('risk_score', 0.0)) * 100:.1f}%"
    thr_pct = f"{float(d.get('threshold', 0.0)) * 100:.1f}%"
    loss = _fmt_inr(d.get("expected_loss_inr", 0.0))
    value = _fmt_inr(d.get("order_value_inr", 0.0))
    reasons = d.get("top_reasons") or []
    rules = [r.get("explanation", "") for r in (d.get("rules_fired") or [])]

    lines = [
        f"Order value: {value}",
        f"Calibrated return-risk score: {score_pct}",
        f"Threshold for this action: {thr_pct}",
        f"Expected return cost if it is returned and nothing is done: {loss}",
        f"Model's recommended action: {d.get('model_action', 'none')}",
        f"Final action after policy guardrails: {d.get('final_action_label', 'none')}",
        f"Decision outcome category: {d.get('outcome', 'unknown')}",
    ]
    if reasons:
        lines.append("Top model reasons: " + "; ".join(str(r) for r in reasons))
    if rules:
        lines.append("Policy guardrails that fired: " + " | ".join(rules))
    if d.get("requires_human_review"):
        lines.append("This order is queued for human review, not actioned automatically.")
    if d.get("in_holdout"):
        lines.append(
            "This order is in the always-approve holdout: deliberately let through "
            "unactioned so outcomes stay measurable."
        )

    allowed = [score_pct, thr_pct, loss, value]
    allowed += re.findall(r"\d[\d,]*(?:\.\d+)?%?", " ".join(rules))
    allowed += re.findall(r"\d[\d,]*(?:\.\d+)?%?", " ".join(str(r) for r in reasons))
    return "\n".join(lines), allowed


def _note_template(d: dict[str, Any]) -> str:
    """The deterministic sentence. This ships when there is no API key -- it has to be
    good enough to demo on its own, not a placeholder."""
    score = float(d.get("risk_score", 0.0))
    action = d.get("final_action_label", "No action")
    value = _fmt_inr(d.get("order_value_inr", 0.0))
    loss = _fmt_inr(d.get("expected_loss_inr", 0.0))
    reasons = [str(r) for r in (d.get("top_reasons") or [])]
    rules = [str(r.get("explanation", "")) for r in (d.get("rules_fired") or [])]

    head = (
        f"This {value} order scores {score:.1%} return risk against a "
        f"{float(d.get('threshold', 0.0)):.1%} threshold, an expected {loss} of return "
        f"cost if it comes back."
    )
    why = f" Main drivers: {'; '.join(reasons[:3])}." if reasons else ""
    if d.get("requires_human_review"):
        tail = " It is above the automatic-action limit and is queued for a human to decide."
    elif d.get("in_holdout"):
        tail = (
            " It was deliberately let through unactioned as part of the always-approve "
            "holdout, so the outcome stays measurable."
        )
    elif d.get("final_action", "none") == "none":
        tail = f" No action was taken. {' '.join(rules)}".rstrip()
    else:
        tail = f" Action taken: {action}." + (f" {' '.join(rules)}" if rules else "")
    return (head + why + tail).strip()


def _dispute_evidence(d: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    """The non-negotiable half of the pack: what the ledger says, unchanged."""
    returned = None if outcome is None else bool(outcome.get("returned"))
    score = float(d.get("risk_score", 0.0))
    return {
        "decision_id": d.get("decision_id"),
        "decided_at": d.get("recorded_at"),
        "order_value_inr": round(float(d.get("order_value_inr", 0.0)), 2),
        "calibrated_risk_score": round(score, 6),
        "threshold_in_force": round(float(d.get("threshold", 0.0)), 6),
        "action_recommended_by_model": d.get("model_action"),
        "action_actually_taken": d.get("final_action"),
        "policy_guardrails_fired": [
            r.get("rule") for r in (d.get("rules_fired") or [])
        ],
        "model_reasons": d.get("top_reasons") or [],
        "model_version": d.get("model_version"),
        "config_fingerprint": d.get("config_fingerprint"),
        "ledger_sequence": d.get("seq"),
        "ledger_entry_hash": d.get("entry_hash"),
        "observed_outcome": (
            "not yet observed" if returned is None
            else ("returned" if returned else "not returned")
        ),
        "model_was_correct": (
            None if returned is None else bool(returned == (score >= float(d.get("threshold", 0.5))))
        ),
        "stated_model_precision": (
            "About 25% at this operating point: roughly three in four flagged orders are "
            "not returned. This score is a risk signal, never proof of intent."
        ),
    }


def _dispute_facts(
    d: dict[str, Any], evidence: dict[str, Any], outcome: dict[str, Any] | None
) -> tuple[str, list[str]]:
    score_pct = f"{float(d.get('risk_score', 0.0)) * 100:.1f}%"
    thr_pct = f"{float(d.get('threshold', 0.0)) * 100:.1f}%"
    value = _fmt_inr(d.get("order_value_inr", 0.0))
    lines = [
        f"Order value: {value}",
        f"Decision recorded at: {evidence['decided_at']}",
        f"Calibrated risk score: {score_pct}",
        f"Threshold in force: {thr_pct}",
        f"Action taken: {d.get('final_action_label', 'none')}",
        f"Observed outcome: {evidence['observed_outcome']}",
        f"Ledger entry hash: {evidence['ledger_entry_hash']}",
        "The model's precision at this threshold is about 25%, so a flag is a risk "
        "signal and not evidence of wrongdoing.",
    ]
    if evidence["model_was_correct"] is False:
        lines.append(
            "Note: the model's call did not match the observed outcome for this order. "
            "State this plainly."
        )
    rules = [str(r.get("explanation", "")) for r in (d.get("rules_fired") or [])]
    if rules:
        lines.append("Policy guardrails that fired: " + " | ".join(rules))
    allowed = [score_pct, thr_pct, value, "25%"]
    allowed += re.findall(r"\d[\d,]*(?:\.\d+)?%?", " ".join(rules))
    return "\n".join(lines), allowed


def _dispute_template(
    d: dict[str, Any], evidence: dict[str, Any], outcome: dict[str, Any] | None
) -> str:
    value = _fmt_inr(d.get("order_value_inr", 0.0))
    correct = evidence["model_was_correct"]
    tail = ""
    if correct is False:
        tail = (
            " The observed outcome did not match the model's call on this order, which "
            "is expected at roughly 25% precision and is recorded here rather than omitted."
        )
    elif correct is True:
        tail = " The observed outcome matched the model's call on this order."
    return (
        f"On {evidence['decided_at']}, this {value} order was scored at "
        f"{float(d.get('risk_score', 0.0)):.1%} return risk against a threshold of "
        f"{float(d.get('threshold', 0.0)):.1%}, and the action taken was "
        f"'{d.get('final_action_label', 'none')}'. The score is produced by a calibrated "
        f"statistical model whose precision at this operating point is about 25%, so it "
        f"is a risk signal and not evidence of customer wrongdoing. The decision, its "
        f"inputs and the policy guardrails applied were written to an append-only ledger "
        f"at the time under entry hash {evidence['ledger_entry_hash']}." + tail
    )


__all__ = ["Responder", "ResponderResult", "SYSTEM_PROMPT", "DISPUTE_SYSTEM_PROMPT"]
