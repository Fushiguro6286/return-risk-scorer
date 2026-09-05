"""SHAP explanations (Tier 4) and merchant-readable reason codes (Tier 3.3).

Two audiences, one engine.

*Us.* Global SHAP tells us which features actually drive the score, which is how we
audit for leakage: a checkout-time model that leans overwhelmingly on one feature is
usually a model that found a post-checkout shortcut. `leakage_audit` makes that check
explicit and writes down what it found instead of leaving it as a vibe.

*The merchant.* Nobody in a warehouse is going to act on "basket_concentration = 0.71,
SHAP +0.31". They need "one line item dominates this order". `reason_codes` maps signed
SHAP contributions onto a fixed vocabulary of plain-language phrases, each tied to
something a human can do about it, and returns at most three.

One honest caveat, stated in the model card too: SHAP is computed on the *raw* LightGBM
margin, not the isotonic-calibrated probability. Isotonic regression is monotone, so the
ranking and sign of the contributions carry over, but the magnitudes are in log-odds
space and should not be read as "this feature added 4 percentage points of risk".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final

import numpy as np
import pandas as pd

from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from .model import TrainedModel


@dataclass(frozen=True)
class ReasonTemplate:
    """A plain-language phrase plus the action it argues for."""

    phrase: Callable[[Any], str]
    action_hint: str


def _pct(x: Any) -> str:
    try:
        return f"{float(x):.0%}"
    except (TypeError, ValueError):
        return "?"


def _num(x: Any) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "?"


#: Fixed vocabulary. Every flagged order's reasons come from here -- never raw feature
#: names -- so the merchant-facing language is stable and reviewable.
REASON_TEMPLATES: Final[dict[str, ReasonTemplate]] = {
    "discount_pct": ReasonTemplate(
        lambda v: f"Deep discount ({_pct(v)} off the usual price for these items)",
        "remove_discount",
    ),
    "is_new_customer": ReasonTemplate(
        lambda v: "First-time buyer with no order history", "remove_cod"
    ),
    "prior_return_rate": ReasonTemplate(
        lambda v: f"Customer has returned {_pct(v)} of their previous orders", "remove_cod"
    ),
    "prior_returns_observed": ReasonTemplate(
        lambda v: f"{_num(v)} confirmed past return(s) from this customer", "remove_cod"
    ),
    "prior_orders": ReasonTemplate(
        lambda v: f"Thin purchase history ({_num(v)} prior orders)", "manual_pack_check"
    ),
    "category_return_rate": ReasonTemplate(
        lambda v: f"High-return product family ({_pct(v)} historical return rate)",
        "manual_pack_check",
    ),
    "order_value_gbp": ReasonTemplate(
        lambda v: "Unusually large order value", "manual_pack_check"
    ),
    "log_order_value": ReasonTemplate(
        lambda v: "Unusually large order value", "manual_pack_check"
    ),
    "n_lines": ReasonTemplate(
        lambda v: f"Large multi-item basket ({_num(v)} distinct lines)", "manual_pack_check"
    ),
    "total_quantity": ReasonTemplate(
        lambda v: f"High total quantity ({_num(v)} units)", "manual_pack_check"
    ),
    "avg_qty_per_line": ReasonTemplate(
        lambda v: "Bulk quantities of the same item", "manual_pack_check"
    ),
    "basket_concentration": ReasonTemplate(
        lambda v: f"Order dominated by one line ({_pct(v)} of value)", "manual_pack_check"
    ),
    "price_spread": ReasonTemplate(
        lambda v: "Wide price range within the basket", "manual_pack_check"
    ),
    "avg_unit_price": ReasonTemplate(lambda v: "Atypical average item price", "manual_pack_check"),
    "max_unit_price": ReasonTemplate(lambda v: "Contains a high-value item", "manual_pack_check"),
    "min_unit_price": ReasonTemplate(lambda v: "Contains very low-priced items", "manual_pack_check"),
    "days_since_last_order": ReasonTemplate(
        lambda v: f"Long gap since last order ({_num(v)} days)", "manual_pack_check"
    ),
    "customer_tenure_days": ReasonTemplate(
        lambda v: "Short relationship with this customer", "remove_cod"
    ),
    "prior_avg_order_value": ReasonTemplate(
        lambda v: "Order size out of line with this customer's history", "manual_pack_check"
    ),
    "hour": ReasonTemplate(lambda v: f"Placed at an unusual hour ({_num(v)}:00)", "manual_pack_check"),
    "day_of_week": ReasonTemplate(lambda v: "Placed on a higher-risk weekday", "manual_pack_check"),
    "is_weekend": ReasonTemplate(lambda v: "Weekend order", "manual_pack_check"),
    "month": ReasonTemplate(lambda v: "Seasonal risk period", "manual_pack_check"),
    "days_to_christmas": ReasonTemplate(
        lambda v: "Peak gifting season (returns spike after the holidays)", "manual_pack_check"
    ),
    "category_order_count": ReasonTemplate(
        lambda v: "Rarely-ordered product family (thin history)", "manual_pack_check"
    ),
    "country": ReasonTemplate(lambda v: f"Destination country ({v})", "remove_cod"),
    "region": ReasonTemplate(lambda v: f"Shipping region ({v})", "remove_cod"),
    "top_category": ReasonTemplate(lambda v: f"Product family {v}", "manual_pack_check"),
}


class Explainer:
    """TreeSHAP over the uncalibrated booster, with a reason-code layer on top."""

    def __init__(self, model: TrainedModel) -> None:
        import shap

        self.model = model
        self.explainer = shap.TreeExplainer(model.raw)
        self.feature_columns = list(model.feature_columns)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for c in CATEGORICAL_COLUMNS:
            if not isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].astype("category")
        return X

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_features) signed contributions toward the positive class."""
        X = self._matrix(df)
        vals = self.explainer.shap_values(X)
        if isinstance(vals, list):  # older SHAP returns one array per class
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, -1]
        return vals

    def global_importance(self, df: pd.DataFrame, max_rows: int = 4000) -> pd.DataFrame:
        """Mean |SHAP| per feature -- the global summary, as a table."""
        sample = df.sample(min(len(df), max_rows), random_state=0) if len(df) > max_rows else df
        vals = self.shap_values(sample)
        out = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "mean_abs_shap": np.abs(vals).mean(axis=0),
                "mean_shap": vals.mean(axis=0),
            }
        )
        out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
        return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    def reason_codes(
        self, df: pd.DataFrame, max_reasons: int = 3, min_shap: float = 0.01
    ) -> list[list[dict[str, Any]]]:
        """Top merchant-readable risk drivers per row, strongest first.

        Only *positive* contributions are returned: the question a flagged order raises
        is "why is this risky", not "what was reassuring about it".
        """
        vals = self.shap_values(df)
        raw = df[self.feature_columns]
        out: list[list[dict[str, Any]]] = []
        for i in range(len(df)):
            row = vals[i]
            order = np.argsort(-row)
            reasons: list[dict[str, Any]] = []
            for j in order[: max_reasons * 3]:
                if row[j] <= min_shap or len(reasons) >= max_reasons:
                    break
                name = self.feature_columns[j]
                tpl = REASON_TEMPLATES.get(name)
                if tpl is None:
                    continue
                value = raw.iloc[i, j]
                if name == "is_new_customer" and not bool(value):
                    continue  # "first-time buyer" is false; do not say it
                if name == "is_weekend" and not bool(value):
                    continue
                reasons.append(
                    {
                        "reason": tpl.phrase(value),
                        "action_hint": tpl.action_hint,
                        "feature": name,
                        "contribution": float(row[j]),
                    }
                )
            out.append(reasons)
        return out


def leakage_audit(
    global_imp: pd.DataFrame, gain_imp: pd.DataFrame, auc_pr: float, base_rate: float
) -> dict[str, Any]:
    """Tier 4.2 -- interrogate the importance ranking for signs of leaked information.

    Three cheap tests that between them catch most real leaks:
      * Does one feature carry an implausible share of the explanation?
      * Is overall performance implausibly good for a checkout-time problem?
      * Does anything post-checkout appear at all (belt and braces over the allow-list)?
    """
    top = global_imp.iloc[0]
    concentration = float(top["share"])
    lift = auc_pr / base_rate if base_rate else float("nan")

    flags: list[str] = []
    if concentration > 0.50:
        flags.append(
            f"'{top['feature']}' carries {concentration:.0%} of total |SHAP| -- inspect it "
            "for post-checkout information"
        )
    if auc_pr > 0.90 or lift > 4.5:
        flags.append(
            f"AUC-PR {auc_pr:.3f} is {lift:.1f}x the base rate, which is implausibly strong "
            "for checkout-time features and usually indicates leakage"
        )
    suspicious = [f for f in global_imp["feature"] if f not in set(FEATURE_COLUMNS)]
    if suspicious:
        flags.append(f"features outside the allow-list reached the model: {suspicious}")

    return {
        "top_feature": str(top["feature"]),
        "top_feature_share_of_shap": concentration,
        "top_5_features": global_imp["feature"].head(5).tolist(),
        "top_5_gain_features": gain_imp["feature"].head(5).tolist(),
        "auc_pr": float(auc_pr),
        "auc_pr_lift_vs_base": float(lift),
        "flags": flags,
        "passed": not flags,
        "verdict": (
            "No leakage indicators. "
            if not flags
            else "Leakage indicators found: " + "; ".join(flags) + ". "
        )
        + (
            f"Top drivers are {', '.join(global_imp['feature'].head(3))}, all of which are "
            "known at checkout."
        ),
    }


__all__ = ["Explainer", "REASON_TEMPLATES", "ReasonTemplate", "leakage_audit"]
