"""FastAPI scoring and decision service.

    uvicorn app.api:app --reload

Two families of endpoint, and the difference between them is the point of this service:

* ``POST /score``   -- what the model thinks. Stateless, side-effect free, and exactly
  what `reports/` measures. Unchanged since the first version, deliberately.
* ``POST /decide``  -- what the system does. Runs the same score through the policy
  guardrails and writes an append-only, hash-chained ledger entry. This is the endpoint
  a merchant integrates against, because it is the only one whose answer can be
  defended six months later.

Plus the surfaces that make that defensible in practice: ``/audit/*`` to replay and
verify decisions, ``/policy`` to read the guardrails currently in force, and
``/respond/*`` for the merchant-facing note and the dispute evidence pack.

The trained model is loaded from joblib **once at startup** and reused -- retraining per
request would be slow and, worse, would mean the score depends on when you asked.

**Startup never hard-fails.** If the model is missing the service comes up degraded:
``/health`` says so, the scoring routes return a 503 that tells you to run
``python run_demo.py``, and the routes that do not need a model keep working. A risk
service that refuses to boot tells an on-call engineer nothing; one that reports which
component is down tells them everything.
"""
from __future__ import annotations

import datetime as dt
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from returnrisk.audit import DecisionLedger  # noqa: E402
from returnrisk.config import load_config  # noqa: E402
from returnrisk.decision import DecisionService  # noqa: E402
from returnrisk.responder import Responder  # noqa: E402
from returnrisk.scoring import OrderInput, ScoringService  # noqa: E402

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load what we can, record what we could not, and come up either way."""
    cfg = load_config()
    _state["cfg"] = cfg
    _state["degraded"] = []

    try:
        _state["service"] = ScoringService.load(cfg)
    except FileNotFoundError:
        _state["service"] = None
        _state["degraded"].append(
            "no trained model on disk - run `python run_demo.py` to train one"
        )
    except Exception as exc:  # a corrupt joblib should not take the process down
        _state["service"] = None
        _state["degraded"].append(f"model failed to load: {type(exc).__name__}: {exc}")

    # The ledger is independent of the model: audit endpoints must keep working even
    # when scoring cannot, which is exactly when someone is investigating.
    try:
        _state["ledger"] = DecisionLedger.from_config(cfg)
    except Exception as exc:
        _state["ledger"] = None
        _state["degraded"].append(f"ledger unavailable: {type(exc).__name__}: {exc}")

    if _state.get("service") is not None:
        _state["decisions"] = DecisionService(
            _state["service"], cfg, ledger=_state.get("ledger")
        )
    else:
        _state["decisions"] = None

    _state["responder"] = Responder(cfg)
    yield
    _state.clear()


app = FastAPI(
    title="Return-Risk Scorer",
    version="2.0.0",
    description=(
        "Defense-only return-risk scoring for e-commerce orders. Returns a calibrated "
        "probability, a cost-optimal recommended action, merchant-readable reasons, and "
        "-- through /decide -- a policy-gated decision written to an append-only, "
        "hash-chained audit ledger."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- models
class ScoreRequest(BaseModel):
    """A checkout-time order. Only `order_value_gbp` is strictly required."""

    order_value_gbp: float = Field(..., gt=0, description="Basket value in GBP")
    n_lines: int = Field(1, ge=1, description="Distinct SKUs in the basket")
    total_quantity: int = Field(1, ge=1, description="Total units")
    avg_unit_price: float = Field(0.0, ge=0)
    max_unit_price: float = Field(0.0, ge=0)
    min_unit_price: float = Field(0.0, ge=0)
    max_line_value_gbp: float | None = None
    country: str = "United Kingdom"
    top_category: str = Field("200", description="Product family (stock-code prefix)")
    discount_pct: float = Field(0.0, ge=-2.0, le=1.0, description="Discount off reference price")
    order_date: dt.datetime | None = None
    customer_id: str | None = Field(None, description="Looked up in the customer profile store")
    action: str | None = Field(
        None, description="Intervention to price against; defaults to config's default_action"
    )
    # Optional history overrides for callers holding their own customer state.
    prior_orders: int | None = Field(None, ge=0)
    prior_return_rate: float | None = Field(None, ge=0.0, le=1.0)
    prior_returns_observed: int | None = Field(None, ge=0)
    days_since_last_order: float | None = Field(None, ge=0)
    prior_avg_order_value: float | None = Field(None, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "order_value_gbp": 820.0,
                "n_lines": 18,
                "total_quantity": 240,
                "avg_unit_price": 3.4,
                "max_unit_price": 12.5,
                "min_unit_price": 0.85,
                "country": "United Kingdom",
                "top_category": "228",
                "discount_pct": 0.34,
                "customer_id": "17850",
                "action": "remove_cod",
            }
        }
    }


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool
    threshold: float
    recommended_action: str
    recommended_action_label: str
    top_reasons: list[str]
    expected_loss_inr: float
    expected_saving_inr: float
    model_version: str


class OutcomeRequest(BaseModel):
    """What actually happened to an order we decided on."""

    returned: bool = Field(..., description="Did the order come back?")
    note: str = Field("", max_length=500)


class ExplainRequest(BaseModel):
    """Ask for prose over a decision that has already been made and recorded."""

    decision_id: str


# --------------------------------------------------------------------------- helpers
def _service() -> ScoringService:
    svc = _state.get("service")
    if svc is None:
        raise HTTPException(
            503,
            {
                "error": "model not loaded",
                "detail": _state.get("degraded") or ["unknown"],
                "fix": "run `python run_demo.py` to train and persist a model, then restart",
            },
        )
    return svc


def _decisions() -> DecisionService:
    svc = _state.get("decisions")
    if svc is None:
        _service()  # raises the informative 503
        raise HTTPException(503, "decision service unavailable")
    return svc


def _ledger() -> DecisionLedger:
    led = _state.get("ledger")
    if led is None:
        raise HTTPException(503, "audit ledger unavailable")
    return led


def _order_from(req: ScoreRequest) -> OrderInput:
    return OrderInput(**req.model_dump(exclude={"action"}))


# --------------------------------------------------------------------------- health
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus enough metadata to know *which* model is answering.

    Reports `degraded` rather than failing, and names the component that is down --
    a boolean `ok` would hide the difference between "no model" and "no ledger".
    """
    svc = _state.get("service")
    responder: Responder | None = _state.get("responder")
    base: dict[str, Any] = {
        "status": "ok" if svc is not None and not _state.get("degraded") else "degraded",
        "degraded": _state.get("degraded") or [],
        "components": {
            "model": "ok" if svc is not None else "unavailable",
            "ledger": "ok" if _state.get("ledger") is not None else "unavailable",
            "responder": (
                "claude" if responder is not None and responder.available else "template-only"
            ),
        },
    }
    if svc is None:
        return base
    base.update(
        {
            "model_version": svc.model.metadata.get("trained_at"),
            "label_definition": svc.model.metadata.get("label_definition"),
            "test_auc_pr": svc.model.metadata.get("test_auc_pr"),
            "test_base_rate": svc.model.metadata.get("test_base_rate"),
            "thresholds": svc.model.thresholds,
            "n_customer_profiles": 0 if svc.profiles is None else int(len(svc.profiles)),
        }
    )
    led = _state.get("ledger")
    if led is not None:
        base["n_decisions_recorded"] = len(led)
    return base


@app.get("/actions")
def actions() -> dict[str, Any]:
    """The interventions available, each with its own cost-optimal threshold."""
    svc = _service()
    return {
        key: {
            "label": spec.label,
            "ops_cost_inr": spec.ops_cost_inr,
            "conversion_loss_prob": spec.conversion_loss_prob,
            "effectiveness": spec.effectiveness,
            "threshold": svc.threshold_for(key),
        }
        for key, spec in svc.cost.actions.items()
    }


@app.get("/policy")
def policy() -> dict[str, Any]:
    """The guardrails currently in force, in the order they are applied.

    Published deliberately: a merchant integrating against `/decide` needs to know the
    bounds on the system's autonomy without reading our source.
    """
    svc = _decisions()
    engine = svc.policy
    p = dict(svc.cfg.get("policy") or {})
    shield = dict(p.get("loyalty_shield") or {})
    return {
        "enabled": engine.enabled,
        "action_severity_order": engine.severity_order,
        "rules": [
            {
                "rule": "min_order_value",
                "effect": "suppress",
                "bound": f"{float(p.get('min_order_value_inr', 0)):,.0f} INR",
                "description": "Below this order value no intervention is worth its friction.",
            },
            {
                "rule": "loyalty_shield",
                "effect": "suppress",
                "bound": (
                    f">= {shield.get('min_prior_orders')} prior orders at "
                    f"<= {float(shield.get('max_prior_return_rate', 1)):.0%} return rate"
                ),
                "description": "Proven-good customers are never actioned automatically.",
            },
            {
                "rule": "new_customer_cap",
                "effect": "downgrade",
                "bound": str(p.get("new_customer_max_action")),
                "description": (
                    "New customers are the model's weakest segment; their action "
                    "severity is capped."
                ),
            },
            {
                "rule": "human_review",
                "effect": "escalate",
                "bound": f"{float(p.get('human_review_above_inr', 0)):,.0f} INR",
                "description": "At or above this value a human decides, not the model.",
            },
            {
                "rule": "daily_action_cap",
                "effect": "escalate",
                "bound": f"{float(p.get('max_daily_action_rate', 1)):.0%} of daily orders",
                "description": "Caps how much of the book the system may action in a day.",
            },
            {
                "rule": "selective_labels_holdout",
                "effect": "holdout",
                "bound": f"{float(p.get('holdout_frac', 0)):.0%} of flagged orders",
                "description": (
                    "Deterministically let through unactioned so outcomes stay "
                    "observable and retraining labels stay unbiased."
                ),
            },
        ],
    }


# --------------------------------------------------------------------------- scoring
@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> dict[str, Any]:
    """Score one order and recommend an action. No side effects, nothing recorded."""
    svc = _service()
    if req.action is not None and req.action not in svc.cost.actions:
        raise HTTPException(
            422, f"unknown action {req.action!r}; expected one of {list(svc.cost.actions)}"
        )
    result = svc.score(_order_from(req), action=req.action)
    return result.to_dict()


@app.post("/score/batch")
def score_batch(reqs: list[ScoreRequest]) -> dict[str, Any]:
    """Score up to 500 orders in one call, for a portfolio sweep."""
    svc = _service()
    if len(reqs) > 500:
        raise HTTPException(413, "batch limited to 500 orders")
    results = [svc.score(_order_from(r), action=r.action).to_dict() for r in reqs]
    flagged = sum(1 for r in results if r["flagged"])
    return {
        "n": len(results),
        "n_flagged": flagged,
        "total_expected_loss_inr": round(sum(r["expected_loss_inr"] for r in results), 2),
        "results": results,
    }


# -------------------------------------------------------------------------- deciding
@app.post("/decide")
def decide(req: ScoreRequest, explain: bool = False) -> dict[str, Any]:
    """Score, gate, and record. The endpoint a merchant actually integrates against.

    Returns the model's recommendation *and* the action policy will permit -- those two
    differ often enough that collapsing them would be dishonest. Every call writes one
    hash-chained ledger entry; the returned `decision_id` and `entry_hash` are the
    receipt.

    `explain=true` additionally attaches a merchant-readable note. Generating it cannot
    change the decision and cannot fail the request.
    """
    svc = _decisions()
    if req.action is not None and req.action not in svc.cost.actions:
        raise HTTPException(
            422, f"unknown action {req.action!r}; expected one of {list(svc.cost.actions)}"
        )
    record = svc.decide(_order_from(req), action=req.action)
    payload = record.to_dict()
    if explain:
        responder: Responder = _state["responder"]
        payload["merchant_note"] = responder.merchant_note(payload).to_dict()
    return payload


@app.post("/decide/batch")
def decide_batch(reqs: list[ScoreRequest]) -> dict[str, Any]:
    """Decide up to 500 orders, with the portfolio-level effect of the guardrails."""
    svc = _decisions()
    if len(reqs) > 500:
        raise HTTPException(413, "batch limited to 500 orders")
    records = [svc.decide(_order_from(r), action=r.action).to_dict() for r in reqs]
    acted = [r for r in records if r["final_action"] != "none" and not r["requires_human_review"]]
    suppressed = [r for r in records if r["flagged"] and r["final_action"] == "none"]
    return {
        "n": len(records),
        "n_flagged_by_model": sum(1 for r in records if r["flagged"]),
        "n_actioned_after_policy": len(acted),
        "n_suppressed_by_policy": len(suppressed),
        "n_sent_to_human_review": sum(1 for r in records if r["requires_human_review"]),
        "n_in_holdout": sum(1 for r in records if r["in_holdout"]),
        "expected_saving_inr": round(sum(r["expected_saving_inr"] for r in acted), 2),
        "results": records,
    }


# ----------------------------------------------------------------------- audit trail
@app.get("/audit/verify")
def audit_verify() -> dict[str, Any]:
    """Walk the hash chain and report the first entry that does not line up."""
    return _ledger().verify()


@app.get("/audit/stats")
def audit_stats() -> dict[str, Any]:
    """What the ledger knows about itself: volumes, guardrail hit counts, chain state."""
    return _ledger().stats()


@app.get("/audit/recent")
def audit_recent(limit: int = 50) -> dict[str, Any]:
    """The most recent decisions, newest first."""
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit must be between 1 and 500")
    return {"entries": _ledger().tail(limit)}


@app.get("/audit/{decision_id}")
def audit_get(decision_id: str) -> dict[str, Any]:
    """Replay one decision exactly as it was recorded, plus its observed outcome."""
    led = _ledger()
    row = led.get(decision_id)
    if row is None:
        raise HTTPException(404, f"no decision {decision_id!r} in the ledger")
    return {"decision": row, "outcome": led.outcomes().get(decision_id)}


@app.post("/audit/{decision_id}/outcome")
def audit_outcome(decision_id: str, req: OutcomeRequest) -> dict[str, Any]:
    """Close the loop: record what actually happened.

    Written to a separate file rather than mutating the decision, because a ledger whose
    rows change after the fact is not a ledger.
    """
    led = _ledger()
    if led.get(decision_id) is None:
        raise HTTPException(404, f"no decision {decision_id!r} in the ledger")
    return led.record_outcome(decision_id, req.returned, req.note)


# ------------------------------------------------------------------------ responding
@app.post("/respond/note")
def respond_note(req: ExplainRequest) -> dict[str, Any]:
    """Merchant-readable prose for a recorded decision.

    Generated from the ledger entry, never from a fresh score -- so the note explains
    what was actually decided rather than what the current model would decide today.
    """
    led = _ledger()
    row = led.get(req.decision_id)
    if row is None:
        raise HTTPException(404, f"no decision {req.decision_id!r} in the ledger")
    responder: Responder = _state["responder"]
    return {"decision_id": req.decision_id, **responder.merchant_note(row).to_dict()}


@app.post("/respond/dispute")
def respond_dispute(req: ExplainRequest) -> dict[str, Any]:
    """A structured evidence pack for a disputed return.

    The evidence table comes from the ledger and is identical with or without an LLM;
    only the narrative paragraph is generated, and it is checked for figures we did not
    supply before it is returned.
    """
    led = _ledger()
    row = led.get(req.decision_id)
    if row is None:
        raise HTTPException(404, f"no decision {req.decision_id!r} in the ledger")
    responder: Responder = _state["responder"]
    return responder.dispute_pack(row, led.outcomes().get(req.decision_id))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
