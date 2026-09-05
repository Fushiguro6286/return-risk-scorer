"""Serving path: one order in, a score and a recommendation out.

Shared by the API and the dashboard so there is exactly one definition of "what the
model does in production" -- a second copy in the UI is how a demo and its API quietly
start disagreeing.

The awkward part of serving this model is customer history: `prior_return_rate` and
friends are computed by walking a customer's past, which a request payload does not
carry. So the pipeline persists a **customer profile store** -- each known customer's
state as of the end of the training data -- and the service looks the caller up. An
unknown customer is scored as genuinely new, which is the honest default and also the
riskier one.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPO_ROOT, Config, load_config
from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, _region
from .model import TrainedModel
from .money import ActionSpec, CostModel

PROFILE_FILE = "customer_profiles.parquet"

#: Neutral fallbacks for a customer we have never seen. Deliberately not zeros --
#: `prior_return_rate` for a new customer is *unknown*, and LightGBM handles NaN
#: natively, which is a truer statement than "0% return rate".
NEW_CUSTOMER_PROFILE: dict[str, Any] = {
    "prior_orders": 0,
    "prior_returns_observed": 0,
    "prior_return_rate": np.nan,
    "prior_avg_order_value": np.nan,
    "days_since_last_order": np.nan,
    "customer_tenure_days": 0.0,
}


@dataclass
class OrderInput:
    """A checkout-time order, as the merchant's system would describe it."""

    order_value_gbp: float
    n_lines: int = 1
    total_quantity: int = 1
    avg_unit_price: float = 0.0
    max_unit_price: float = 0.0
    min_unit_price: float = 0.0
    max_line_value_gbp: float | None = None
    country: str = "United Kingdom"
    top_category: str = "200"
    discount_pct: float = 0.0
    order_date: dt.datetime | None = None
    customer_id: str | None = None
    # Optional explicit history, for callers that keep their own customer state.
    prior_orders: int | None = None
    prior_return_rate: float | None = None
    prior_returns_observed: int | None = None
    days_since_last_order: float | None = None
    prior_avg_order_value: float | None = None
    customer_tenure_days: float | None = None

    def defaults(self) -> "OrderInput":
        """Fill the price fields a caller may reasonably omit."""
        if self.avg_unit_price <= 0 and self.total_quantity:
            self.avg_unit_price = self.order_value_gbp / max(self.total_quantity, 1)
        if self.max_unit_price <= 0:
            self.max_unit_price = self.avg_unit_price
        if self.min_unit_price <= 0:
            self.min_unit_price = self.avg_unit_price
        if self.max_line_value_gbp is None:
            self.max_line_value_gbp = self.order_value_gbp / max(self.n_lines, 1)
        if self.order_date is None:
            self.order_date = dt.datetime.now()
        return self


@dataclass
class ScoreResult:
    """Exactly what `POST /score` returns."""

    risk_score: float
    flagged: bool
    threshold: float
    recommended_action: str
    recommended_action_label: str
    top_reasons: list[str]
    expected_loss_inr: float
    expected_saving_inr: float
    reason_detail: list[dict[str, Any]] = field(default_factory=list)
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 6),
            "flagged": self.flagged,
            "threshold": round(self.threshold, 6),
            "recommended_action": self.recommended_action,
            "recommended_action_label": self.recommended_action_label,
            "top_reasons": self.top_reasons,
            "expected_loss_inr": round(self.expected_loss_inr, 2),
            "expected_saving_inr": round(self.expected_saving_inr, 2),
            "reason_detail": self.reason_detail,
            "model_version": self.model_version,
        }


def build_profile_store(orders_with_history: pd.DataFrame) -> pd.DataFrame:
    """Latest known history per customer, for the serving path.

    Takes the last row per customer from the feature frame, then rolls it forward by
    one order -- because at serving time that last order is itself now in the past.
    """
    df = orders_with_history.sort_values(["customer_id", "order_date"], kind="stable")
    last = df.groupby("customer_id", as_index=False).last()
    out = pd.DataFrame(
        {
            "customer_id": last["customer_id"].astype(str),
            "prior_orders": last["prior_orders"].astype("int64") + 1,
            "prior_returns_observed": (
                last["prior_returns_observed"].astype("int64") + last["returned"].astype("int64")
            ),
            "prior_avg_order_value": last["prior_avg_order_value"].fillna(last["order_value_gbp"]),
            "last_order_date": last["order_date"],
            "first_order_date": last["order_date"] - pd.to_timedelta(
                last["customer_tenure_days"].fillna(0.0), unit="D"
            ),
        }
    )
    out["prior_return_rate"] = out["prior_returns_observed"] / out["prior_orders"].clip(lower=1)
    return out


class ScoringService:
    """Loads the persisted artifacts once and scores orders without retraining."""

    def __init__(
        self,
        model: TrainedModel,
        cfg: Config,
        profiles: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.cost = CostModel(cfg)
        self.profiles = (
            profiles.set_index("customer_id") if profiles is not None and len(profiles) else None
        )
        self._explainer: Any = None

    # -- construction -------------------------------------------------------------
    @classmethod
    def load(cls, cfg: Config | None = None) -> "ScoringService":
        """Cold start: read the joblib model and the profile store from disk."""
        cfg = cfg or load_config()
        model = TrainedModel.load(cfg.get_path("model_file"))
        profile_path = REPO_ROOT / cfg["paths"]["models_dir"] / PROFILE_FILE
        profiles = pd.read_parquet(profile_path) if profile_path.exists() else None
        return cls(model, cfg, profiles)

    @property
    def explainer(self) -> Any:
        """Built lazily -- SHAP costs ~1s to set up and most callers want a score first."""
        if self._explainer is None:
            from .explain import Explainer

            self._explainer = Explainer(self.model)
        return self._explainer

    # -- feature assembly ---------------------------------------------------------
    def lookup_profile(self, customer_id: str | None, as_of: dt.datetime) -> dict[str, Any]:
        """Customer history from the store, or the new-customer defaults."""
        if customer_id is None or self.profiles is None or str(customer_id) not in self.profiles.index:
            return dict(NEW_CUSTOMER_PROFILE)
        row = self.profiles.loc[str(customer_id)]
        last = pd.Timestamp(row["last_order_date"])
        first = pd.Timestamp(row["first_order_date"])
        ts = pd.Timestamp(as_of)
        return {
            "prior_orders": int(row["prior_orders"]),
            "prior_returns_observed": int(row["prior_returns_observed"]),
            "prior_return_rate": float(row["prior_return_rate"]),
            "prior_avg_order_value": float(row["prior_avg_order_value"]),
            "days_since_last_order": max((ts - last).total_seconds() / 86400.0, 0.0),
            "customer_tenure_days": max((ts - first).total_seconds() / 86400.0, 0.0),
        }

    def to_feature_row(self, order: OrderInput) -> pd.DataFrame:
        """Assemble a single-row frame carrying exactly FEATURE_COLUMNS."""
        o = order.defaults()
        ts = pd.Timestamp(o.order_date)
        profile = self.lookup_profile(o.customer_id, ts)
        for key in NEW_CUSTOMER_PROFILE:
            override = getattr(o, key, None)
            if override is not None:
                profile[key] = override

        fb = self.model.feature_builder
        category = str(o.top_category)
        doy = int(ts.dayofyear)

        row: dict[str, Any] = {
            "order_value_gbp": float(o.order_value_gbp),
            "log_order_value": float(np.log1p(max(o.order_value_gbp, 0.0))),
            "n_lines": int(o.n_lines),
            "total_quantity": int(o.total_quantity),
            "avg_qty_per_line": o.total_quantity / max(o.n_lines, 1),
            "avg_unit_price": float(o.avg_unit_price),
            "max_unit_price": float(o.max_unit_price),
            "min_unit_price": float(o.min_unit_price),
            "price_spread": o.max_unit_price / max(o.min_unit_price, 0.01),
            "basket_concentration": (o.max_line_value_gbp or 0.0) / max(o.order_value_gbp, 0.01),
            "discount_pct": float(o.discount_pct),
            "hour": int(ts.hour),
            "day_of_week": int(ts.dayofweek),
            "is_weekend": int(ts.dayofweek >= 5),
            "month": int(ts.month),
            "days_to_christmas": int(min((359 - doy) % 365, (doy + 6) % 365)),
            "prior_orders": profile["prior_orders"],
            "is_new_customer": int(profile["prior_orders"] == 0),
            "days_since_last_order": profile["days_since_last_order"],
            "customer_tenure_days": profile["customer_tenure_days"],
            "prior_avg_order_value": profile["prior_avg_order_value"],
            "prior_return_rate": profile["prior_return_rate"],
            "prior_returns_observed": profile["prior_returns_observed"],
            "category_return_rate": float(
                fb.category_rate_.get(category, fb.global_rate_) if fb.category_rate_ is not None
                else fb.global_rate_
            ),
            "category_order_count": float(
                fb.category_count_.get(category, 0.0) if fb.category_count_ is not None else 0.0
            ),
        }
        df = pd.DataFrame([row])
        country = str(o.country)
        df["country"] = pd.Categorical(
            [country if country in fb.countries_ else "OTHER"],
            categories=[*fb.countries_, "OTHER"],
        )
        df["region"] = pd.Categorical(
            _region(pd.Series([country])).tolist(), categories=["UK", "EU", "ROW"]
        )
        df["top_category"] = pd.Categorical(
            [category if category in fb.categories_ else "OTHER"],
            categories=[*fb.categories_, "OTHER"],
        )
        return df[FEATURE_COLUMNS]

    # -- scoring ------------------------------------------------------------------
    def threshold_for(self, action: str) -> float:
        """The persisted cost-optimal t* for one intervention."""
        return float(self.model.thresholds.get(action, 0.5))

    def score(
        self, order: OrderInput, action: str | None = None, with_reasons: bool = True
    ) -> ScoreResult:
        """Score one order and recommend an action, with merchant-readable reasons."""
        action = action or self.cost.default_action
        spec: ActionSpec = self.cost.resolve(action)
        X = self.to_feature_row(order)
        p = float(self.model.predict_proba(X)[0])
        t = self.threshold_for(action)

        value_inr = self.cost.to_inr(np.array([order.order_value_gbp]))
        fn_cost = float(self.cost.fn_cost(value_inr)[0])
        expected_loss = p * fn_cost
        # Acting is worth it when the prevented loss exceeds the friction it creates.
        act_cost = spec.ops_cost_inr + (1.0 - p) * float(
            spec.conversion_loss_prob * self.cost.margin_inr(value_inr)[0]
        )
        expected_saving = p * spec.effectiveness * fn_cost - act_cost

        reasons: list[dict[str, Any]] = []
        if with_reasons:
            try:
                reasons = self.explainer.reason_codes(X, max_reasons=3)[0]
            except Exception:  # pragma: no cover - never fail a score over an explanation
                reasons = []

        return ScoreResult(
            risk_score=p,
            flagged=p >= t,
            threshold=t,
            recommended_action=action if p >= t else "none",
            recommended_action_label=spec.label if p >= t else "No action - proceed normally",
            top_reasons=[r["reason"] for r in reasons],
            expected_loss_inr=expected_loss,
            expected_saving_inr=expected_saving,
            reason_detail=reasons,
            model_version=str(self.model.metadata.get("trained_at", "unknown")),
        )

    def score_batch(self, orders: pd.DataFrame, action: str | None = None) -> pd.DataFrame:
        """Score a pre-built feature frame (the dashboard's portfolio view)."""
        action = action or self.cost.default_action
        spec = self.cost.resolve(action)
        X = orders[FEATURE_COLUMNS].copy()
        for c in CATEGORICAL_COLUMNS:
            if not isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].astype("category")
        p = self.model.predict_proba(X)
        t = self.threshold_for(action)
        value_inr = self.cost.to_inr(orders["order_value_gbp"])
        fn_cost = self.cost.fn_cost(value_inr)

        out = orders.copy()
        out["risk_score"] = p
        out["flagged"] = p >= t
        out["threshold"] = t
        out["order_value_inr"] = value_inr
        out["expected_loss_inr"] = p * fn_cost
        out["expected_saving_inr"] = (
            p * spec.effectiveness * fn_cost
            - spec.ops_cost_inr
            - (1 - p) * spec.conversion_loss_prob * self.cost.margin_inr(value_inr)
        )
        return out.sort_values("expected_loss_inr", ascending=False)


__all__ = [
    "NEW_CUSTOMER_PROFILE",
    "PROFILE_FILE",
    "OrderInput",
    "ScoreResult",
    "ScoringService",
    "build_profile_store",
]
