"""What the model must beat, priced in rupees (Tier 1.3).

The comparison set is deliberately unkind to the model:

* **Do nothing.** Absorb every return. The merchant's status quo, and the denominator
  for every "rupees saved" figure in this repo.
* **Three real merchant rules**, config-driven, each firing on a meaningful slice:
  deep discount, big-ticket order, and known repeat returner. These are not strawmen --
  "hand-check anything over five hundred quid" is the single strongest one-line
  heuristic we could find on this dataset, and pricing it in rupees flatters it further,
  because big orders are exactly where a prevented return is worth most.
* **Flag everything.** The other degenerate corner. It exposes rules that buy recall
  with no precision: intervening on all 7.5k test orders is affordable for a ~Rs.40
  pack check and ruinous for a COD block, which is the whole per-action argument.

The model is then shown at its cost-optimal t*, and -- as a comparison, not a headline --
under the per-order value-aware rule from `money.CostModel.value_aware_flags`.
"""
from __future__ import annotations

import operator
from typing import Any, Callable, Final

import numpy as np
import pandas as pd

from .config import Config
from .money import CostModel

_OPS: Final[dict[str, Callable[[Any, Any], Any]]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}


def rule_flags(features: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    """Evaluate one config-declared rule against the feature frame."""
    op = _OPS.get(str(rule["op"]))
    if op is None:
        raise ValueError(f"unsupported rule operator {rule['op']!r}; expected one of {list(_OPS)}")
    col = str(rule["feature"])
    if col not in features.columns:
        raise KeyError(f"rule '{rule['name']}' references unknown feature {col!r}")
    return np.asarray(op(features[col].to_numpy(), rule["value"])).astype(bool)


def rule_diagnostics(features: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    """Prove each rule is real: how often it fires, and how well it discriminates."""
    y = features["returned"].to_numpy().astype(bool)
    base = float(y.mean())
    out = []
    for rule in cfg["baselines"]["rules"]:
        flags = rule_flags(features, rule)
        n = int(flags.sum())
        precision = float((flags & y).sum() / n) if n else 0.0
        out.append(
            {
                "rule": str(rule["name"]),
                "condition": f"{rule['feature']} {rule['op']} {rule['value']}",
                "orders_flagged": n,
                "flag_rate": float(flags.mean()),
                "precision": precision,
                "lift_vs_base": precision / base if base else float("nan"),
                "recall": float((flags & y).sum() / max(int(y.sum()), 1)),
                "base_rate": base,
            }
        )
    return out


def compare_baselines(
    features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    cfg: Config,
    t_star: float,
    action: str | None = None,
) -> pd.DataFrame:
    """One row per policy, all priced with the same cost model and the same action."""
    action = action or str(cfg["baselines"]["action"])
    y = features["returned"].to_numpy()
    value_inr = cost.to_inr(features["order_value_gbp"])
    n = len(y)

    policies: list[tuple[str, np.ndarray]] = [("Do nothing (status quo)", np.zeros(n, dtype=bool))]
    for rule in cfg["baselines"]["rules"]:
        policies.append((f"Rule -- {rule['name']}", rule_flags(features, rule)))
    policies += [
        ("Flag every order", np.ones(n, dtype=bool)),
        (f"Model @ t*={t_star:.3f}", np.asarray(proba) >= t_star),
        ("Model, per-order value-aware rule", cost.value_aware_flags(proba, value_inr, action)),
    ]

    rows = []
    for label, flags in policies:
        bd = cost.evaluate_policy(y, flags, value_inr, action, label=label)
        rows.append(
            {
                "policy": label,
                "orders_flagged": bd.n_flagged,
                "flag_rate": bd.flag_rate,
                "precision": bd.precision,
                "recall": bd.recall,
                "total_cost_inr": bd.total_cost,
                "savings_vs_do_nothing_inr": bd.savings,
                "savings_pct": bd.savings / bd.do_nothing_cost if bd.do_nothing_cost else 0.0,
            }
        )
    out = pd.DataFrame(rows)

    model_cost = float(out.loc[out["policy"].str.startswith("Model @"), "total_cost_inr"].iloc[0])
    out["model_saves_vs_this_inr"] = out["total_cost_inr"] - model_cost
    out.attrs["action"] = action
    return out


def headline_comparison(table: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """The two head-to-head numbers the README quotes, extracted once."""
    headline = str(cfg["baselines"]["headline_rule"])
    row_model = table[table["policy"].str.startswith("Model @")].iloc[0]
    row_none = table[table["policy"].str.startswith("Do nothing")].iloc[0]
    match = table[table["policy"] == f"Rule -- {headline}"]
    row_rule = match.iloc[0] if len(match) else table.iloc[1]

    best_rule = table[table["policy"].str.startswith("Rule --")].sort_values("total_cost_inr").iloc[0]
    return {
        "model_total_cost_inr": float(row_model["total_cost_inr"]),
        "do_nothing_cost_inr": float(row_none["total_cost_inr"]),
        "savings_vs_do_nothing_inr": float(row_model["savings_vs_do_nothing_inr"]),
        "savings_vs_do_nothing_pct": float(row_model["savings_pct"]),
        "headline_rule": headline,
        "headline_rule_cost_inr": float(row_rule["total_cost_inr"]),
        "savings_vs_headline_rule_inr": float(row_rule["total_cost_inr"] - row_model["total_cost_inr"]),
        "best_rule": str(best_rule["policy"]),
        "best_rule_cost_inr": float(best_rule["total_cost_inr"]),
        "savings_vs_best_rule_inr": float(best_rule["total_cost_inr"] - row_model["total_cost_inr"]),
        "model_beats_all_rules": bool(row_model["total_cost_inr"] < best_rule["total_cost_inr"]),
        "model_beats_do_nothing": bool(row_model["savings_vs_do_nothing_inr"] > 0),
    }


__all__ = ["compare_baselines", "headline_comparison", "rule_diagnostics", "rule_flags"]
