"""The money layer: price every decision in rupees, then pick the threshold that hurts least.

A return-risk score is worthless on its own. What a merchant needs is: *given what a
missed return costs me and what annoying a good customer costs me, at what score should
I actually intervene?* That is a cost-minimisation, not an F1-maximisation, and the
answer differs per intervention -- which is why `optimal_threshold_per_action` exists.

Cost algebra (all amounts in rupees, per order)
-----------------------------------------------
Let `V` be order value, `M = margin_pct * V` the booked margin, and

    FN_cost(V) = reverse_logistics + restocking_pct * V + (margin_pct * V if lost_margin)

the cost of a return that we did not prevent: the reverse leg, the condition write-down,
and the margin reversed by the refund.

For an intervention `a` with ops cost `c_a`, conversion-loss probability `l_a` and
effectiveness `e_a` (the share of true returns it actually prevents):

    flagged & returned      (TP) : c_a + (1 - e_a) * FN_cost(V)
    flagged & not returned  (FP) : c_a + l_a * M          <-- the false-positive cost
    not flagged & returned  (FN) : FN_cost(V)
    not flagged & not ret.  (TN) : 0

Two deliberate modelling choices, both stated so a reviewer can disagree with them:

* `e_a < 1`. An intervention is not a cure. A pack check catches picking errors, not
  buyer's remorse. Crediting a flag with the full return cost would inflate savings by
  ~2-4x and is the single most common way these demos lie.
* No conversion loss is charged on a true positive. If a would-be returner abandons the
  cart, that is the outcome we wanted; it is already priced into `e_a`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class ActionSpec:
    """One intervention the merchant can take on a flagged order."""

    key: str
    label: str
    ops_cost_inr: float
    conversion_loss_prob: float
    effectiveness: float

    @classmethod
    def from_config(cls, key: str, cfg: Config) -> "ActionSpec":
        spec = cfg["money"]["actions"][key]
        return cls(
            key=key,
            label=str(spec["label"]),
            ops_cost_inr=float(spec["ops_cost_inr"]),
            conversion_loss_prob=float(spec["conversion_loss_prob"]),
            effectiveness=float(spec["effectiveness"]),
        )


def load_actions(cfg: Config) -> dict[str, ActionSpec]:
    return {k: ActionSpec.from_config(k, cfg) for k in cfg["money"]["actions"]}


@dataclass
class CostBreakdown:
    """Rupee-denominated confusion matrix plus the counts behind it."""

    action: str
    threshold: float
    n: int
    n_tp: int
    n_fp: int
    n_fn: int
    n_tn: int
    cost_tp: float
    cost_fp: float
    cost_fn: float
    cost_tn: float
    #: What the caught orders WOULD have cost had we not intervened. Without this the
    #: true-positive cell reads as a loss, when it is the cell doing the work.
    cost_tp_counterfactual: float
    total_cost: float
    do_nothing_cost: float
    savings: float
    n_flagged: int
    flag_rate: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)

    @property
    def matrix(self) -> np.ndarray:
        """2x2 of rupee cost, rows = actual (no-return, return), cols = predicted."""
        return np.array([[self.cost_tn, self.cost_fp], [self.cost_fn, self.cost_tp]])

    @property
    def counts(self) -> np.ndarray:
        return np.array([[self.n_tn, self.n_fp], [self.n_fn, self.n_tp]])


class CostModel:
    """Turns order values and a chosen action into rupee costs."""

    def __init__(self, cfg: Config) -> None:
        m = cfg["money"]
        self.cfg = cfg
        self.symbol: str = str(m["currency_symbol"])
        self.gbp_to_inr: float = float(m["gbp_to_inr"])
        self.margin_pct: float = float(m["gross_margin_pct"])
        fn = m["false_negative"]
        self.reverse_logistics: float = float(fn["reverse_logistics_inr"])
        self.restocking_pct: float = float(fn["restocking_pct"])
        self.lost_margin: bool = bool(fn["lost_margin"])
        self.actions: dict[str, ActionSpec] = load_actions(cfg)
        self.default_action: str = str(m["default_action"])
        # Sensitivity analysis scales the two cost families without touching config.
        self.fn_multiplier: float = 1.0
        self.fp_multiplier: float = 1.0

    # -- primitives ---------------------------------------------------------------
    def to_inr(self, order_value_gbp: np.ndarray | pd.Series) -> np.ndarray:
        return np.asarray(order_value_gbp, dtype="float64") * self.gbp_to_inr

    def margin_inr(self, value_inr: np.ndarray) -> np.ndarray:
        return value_inr * self.margin_pct

    def fn_cost(self, value_inr: np.ndarray) -> np.ndarray:
        """Rupee cost of a return we did not prevent."""
        cost = self.reverse_logistics + self.restocking_pct * value_inr
        if self.lost_margin:
            cost = cost + self.margin_pct * value_inr
        return cost * self.fn_multiplier

    def fp_cost(self, value_inr: np.ndarray, action: ActionSpec) -> np.ndarray:
        """Rupee cost of intervening on an order that was never going to come back."""
        friction = action.conversion_loss_prob * self.margin_inr(value_inr)
        return (action.ops_cost_inr + friction) * self.fp_multiplier

    def tp_cost(self, value_inr: np.ndarray, action: ActionSpec) -> np.ndarray:
        """Residual cost when we intervene on a genuine return: ops + what leaks through."""
        return action.ops_cost_inr * self.fp_multiplier + (
            1.0 - action.effectiveness
        ) * self.fn_cost(value_inr)

    def resolve(self, action: str | ActionSpec | None) -> ActionSpec:
        if isinstance(action, ActionSpec):
            return action
        return self.actions[action or self.default_action]

    # -- policies -----------------------------------------------------------------
    def do_nothing_cost(self, y: np.ndarray, value_inr: np.ndarray) -> float:
        """Every return lands, uncontested. The number every policy must beat."""
        y = np.asarray(y).astype(bool)
        return float(self.fn_cost(value_inr)[y].sum())

    def evaluate_policy(
        self,
        y: np.ndarray | pd.Series,
        flagged: np.ndarray | pd.Series,
        value_inr: np.ndarray,
        action: str | ActionSpec | None = None,
        *,
        threshold: float = float("nan"),
        label: str | None = None,
    ) -> CostBreakdown:
        """Price an arbitrary flag vector. Used for the model, the rules and the baselines."""
        a = self.resolve(action)
        y = np.asarray(y).astype(bool)
        f = np.asarray(flagged).astype(bool)
        value_inr = np.asarray(value_inr, dtype="float64")

        tp, fp = y & f, ~y & f
        fn, tn = y & ~f, ~y & ~f

        c_tp = float(self.tp_cost(value_inr[tp], a).sum()) if tp.any() else 0.0
        c_fp = float(self.fp_cost(value_inr[fp], a).sum()) if fp.any() else 0.0
        c_fn = float(self.fn_cost(value_inr[fn]).sum()) if fn.any() else 0.0
        c_tp_cf = float(self.fn_cost(value_inr[tp]).sum()) if tp.any() else 0.0
        total = c_tp + c_fp + c_fn
        baseline = self.do_nothing_cost(y, value_inr)
        n_flagged = int(f.sum())

        return CostBreakdown(
            action=label or a.key,
            threshold=float(threshold),
            n=int(len(y)),
            n_tp=int(tp.sum()),
            n_fp=int(fp.sum()),
            n_fn=int(fn.sum()),
            n_tn=int(tn.sum()),
            cost_tp=c_tp,
            cost_fp=c_fp,
            cost_fn=c_fn,
            cost_tn=0.0,
            cost_tp_counterfactual=c_tp_cf,
            total_cost=total,
            do_nothing_cost=baseline,
            savings=baseline - total,
            n_flagged=n_flagged,
            flag_rate=n_flagged / max(len(y), 1),
            precision=float(tp.sum() / n_flagged) if n_flagged else 0.0,
            recall=float(tp.sum() / max(int(y.sum()), 1)),
        )

    # -- threshold search ---------------------------------------------------------
    def sweep(
        self,
        y: np.ndarray | pd.Series,
        proba: np.ndarray | pd.Series,
        value_inr: np.ndarray,
        action: str | ActionSpec | None = None,
        grid: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        """Total rupee cost at every candidate threshold, for one action."""
        a = self.resolve(action)
        p = np.asarray(proba, dtype="float64")
        thresholds = np.asarray(list(grid) if grid is not None else default_grid(p))
        rows = [
            self.evaluate_policy(y, p >= t, value_inr, a, threshold=float(t)).to_dict()
            for t in thresholds
        ]
        return pd.DataFrame(rows)

    def optimal_threshold(
        self,
        y: np.ndarray | pd.Series,
        proba: np.ndarray | pd.Series,
        value_inr: np.ndarray,
        action: str | ActionSpec | None = None,
        grid: Iterable[float] | None = None,
    ) -> tuple[float, CostBreakdown, pd.DataFrame]:
        """Return (t*, the breakdown at t*, the full sweep)."""
        sweep = self.sweep(y, proba, value_inr, action, grid)
        best = sweep.loc[sweep["total_cost"].idxmin()]
        t_star = float(best["threshold"])
        bd = self.evaluate_policy(
            y, np.asarray(proba) >= t_star, value_inr, action, threshold=t_star
        )
        return t_star, bd, sweep

    def optimal_threshold_per_action(
        self,
        y: np.ndarray | pd.Series,
        proba: np.ndarray | pd.Series,
        value_inr: np.ndarray,
        grid: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        """Tier 1.2 -- solve t* separately for each intervention.

        Expect a monotone table: the cheaper the false positive, the lower t*, and the
        more orders get flagged. `analytic_threshold` explains why.
        """
        rows = []
        for key, a in self.actions.items():
            t_star, bd, _ = self.optimal_threshold(y, proba, value_inr, a, grid)
            rows.append(
                {
                    "action": key,
                    "action_label": a.label,
                    "ops_cost_inr": a.ops_cost_inr,
                    "conversion_loss_prob": a.conversion_loss_prob,
                    "effectiveness": a.effectiveness,
                    "fp_cost_mean_inr": float(
                        self.fp_cost(np.asarray(value_inr, dtype="float64"), a).mean()
                    ),
                    "t_star": t_star,
                    "orders_flagged": bd.n_flagged,
                    "flag_rate": bd.flag_rate,
                    "precision": bd.precision,
                    "recall": bd.recall,
                    "total_cost_inr": bd.total_cost,
                    "savings_inr": bd.savings,
                    "savings_pct": bd.savings / bd.do_nothing_cost if bd.do_nothing_cost else 0.0,
                }
            )
        out = pd.DataFrame(rows).sort_values("fp_cost_mean_inr", ignore_index=True)
        return out

    def analytic_threshold(self, value_inr: float, action: str | ActionSpec) -> float:
        """Closed-form break-even score for one order, for sanity-checking the sweep.

        Intervene when  p * e * FN  >  ops + (1-p) * l * M, i.e.

            p* = (ops + l*M) / (e*FN + l*M)

        which is increasing in the false-positive cost -- the monotonicity Tier 1.2 asks
        us to demonstrate.
        """
        a = self.resolve(action)
        v = np.asarray([value_inr], dtype="float64")
        fn = float(self.fn_cost(v)[0])
        friction = float(a.conversion_loss_prob * self.margin_inr(v)[0])
        num = a.ops_cost_inr + friction
        den = a.effectiveness * fn + friction
        return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0

    def value_aware_flags(
        self, proba: np.ndarray, value_inr: np.ndarray, action: str | ActionSpec | None = None
    ) -> np.ndarray:
        """Per-order expected-value rule: compare costs order by order, no single cut.

        This is the decision-theoretically correct policy *if* the probabilities are
        perfectly calibrated -- a Rs.500 order and a Rs.50,000 order plainly deserve
        different cuts. In practice it does not always beat the global t*, because t* is
        chosen by exhaustive search on the very data it is scored against and so enjoys
        an in-sample advantage the analytic rule does not. Reported alongside the
        baselines for exactly that reason, not as the headline.
        """
        a = self.resolve(action)
        p = np.asarray(proba, dtype="float64")
        v = np.asarray(value_inr, dtype="float64")
        gain = p * a.effectiveness * self.fn_cost(v)
        cost = a.ops_cost_inr + (1.0 - p) * a.conversion_loss_prob * self.margin_inr(v)
        return gain > cost


def default_grid(proba: np.ndarray, n: int = 501) -> np.ndarray:
    """Threshold candidates: a uniform grid plus the score quantiles.

    The quantiles matter -- with a calibrated model on a 10% base rate most scores sit
    below 0.4, and a bare linspace would resolve the interesting region too coarsely.
    """
    p = np.asarray(proba, dtype="float64")
    uniform = np.linspace(0.0, 1.0, 101)
    quantiles = np.quantile(p, np.linspace(0.0, 1.0, n)) if len(p) else np.array([0.5])
    grid = np.unique(np.concatenate([uniform, quantiles, [0.0, 1.0 + 1e-9]]))
    return np.round(grid, 6)


def format_inr(x: float, symbol: str = "Rs.") -> str:
    """Compact rupee formatting for chart annotations (lakh/crore scale)."""
    ax = abs(x)
    if ax >= 1e7:
        return f"{symbol}{x / 1e7:,.2f} Cr"
    if ax >= 1e5:
        return f"{symbol}{x / 1e5:,.2f} L"
    if ax >= 1e3:
        return f"{symbol}{x / 1e3:,.1f} K"
    return f"{symbol}{x:,.0f}"


__all__ = [
    "ActionSpec",
    "CostBreakdown",
    "CostModel",
    "default_grid",
    "format_inr",
    "load_actions",
]
