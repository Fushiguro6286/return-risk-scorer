"""Selective labels: the feedback loop that quietly destroys models like this (Tier 2.2).

The problem
-----------
The moment this scorer is switched on, it starts editing its own training data. Block
COD on a high-risk order and one of two things happens: the customer prepays and the
intervention works, or the customer walks. Either way the counterfactual -- *would this
order have been returned?* -- is never observed. The label is censored, and censored
precisely on the rows the model was most confident about.

Retrain naively on that data next quarter and the failure is not a crash, it is a slow
lie:

1. Flagged orders come back overwhelmingly labelled 0 (their returns were prevented, or
   the order never happened).
2. The model learns that its own high-risk signature is *safe*.
3. Scores on genuinely risky orders fall, the policy flags fewer of them, returns climb.
4. The dashboard still looks fine, because the model is evaluated on the same censored
   data that broke it.

This is not hypothetical -- it is the standard selective-labels problem from consumer
lending, and it is the single most likely way this project fails in production.

The mitigation
--------------
Keep a small **always-approve holdout**: a random `holdout_frac` of flagged orders is
let through unactioned. Those orders are the only place unbiased outcomes still come
from, so they are (a) never censored and (b) up-weighted at retraining time by the
inverse of their sampling probability, which reconstructs the flagged population's true
outcome distribution from a 5% sample.

It costs real money -- you are knowingly eating returns you could have prevented -- and
`holdout_cost` prices that, because "we pay ~X rupees a quarter to keep the model
honest" is the form of the argument a merchant can actually approve.

What `run_experiment` does
--------------------------
Splits the test window in two. Period 1 is where the policy operates and censors labels.
Period 2 is the clean, never-touched evaluation set. Three models are then compared on
period 2: the original, one retrained on naively censored period-1 data, and one
retrained with the always-approve holdout and IPW. Only the third should hold up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from .metrics import evaluate
from .model import build_estimator
from .money import CostModel


@dataclass
class SelectiveLabelsResult:
    """Everything the README paragraph and the chart are built from."""

    action: str
    threshold: float
    holdout_frac: float
    period1_range: tuple[str, str]
    period2_range: tuple[str, str]
    n_period1: int
    n_period2: int
    n_flagged_p1: int
    n_censored_p1: int
    n_holdout_p1: int
    observed_base_rate_naive: float
    observed_base_rate_holdout: float
    true_base_rate_p1: float
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_cost_inr: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "threshold": self.threshold,
            "holdout_frac": self.holdout_frac,
            "period1": {"start": self.period1_range[0], "end": self.period1_range[1], "n": self.n_period1},
            "period2": {"start": self.period2_range[0], "end": self.period2_range[1], "n": self.n_period2},
            "n_flagged_period1": self.n_flagged_p1,
            "n_labels_censored": self.n_censored_p1,
            "n_always_approve_holdout": self.n_holdout_p1,
            "true_base_rate_period1": self.true_base_rate_p1,
            "observed_base_rate_naive_retrain": self.observed_base_rate_naive,
            "observed_base_rate_with_holdout": self.observed_base_rate_holdout,
            "base_rate_understatement_naive": self.true_base_rate_p1 - self.observed_base_rate_naive,
            "holdout_cost_inr": self.holdout_cost_inr,
            "comparison": self.comparison.to_dict(orient="records"),
            **self.notes,
        }


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_COLUMNS:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def _fit(train: pd.DataFrame, cfg: Config, weights: np.ndarray | None = None):
    est = build_estimator(cfg)
    est.fit(
        _matrix(train),
        train["returned"].to_numpy(),
        sample_weight=weights,
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return est


def simulate_censoring(
    features: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    holdout_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Apply the policy and mark which labels the merchant would still get to see.

    Adds three columns:
      * `flagged`      -- the policy acted (or would have)
      * `in_holdout`   -- flagged but deliberately let through, so the label survives
      * `label_observed` -- whether an outcome is available for retraining at all
    """
    rng = np.random.default_rng(seed)
    df = features.copy()
    df["risk_score"] = np.asarray(proba, dtype="float64")
    df["flagged"] = df["risk_score"].to_numpy() >= threshold
    df["in_holdout"] = df["flagged"] & (rng.random(len(df)) < holdout_frac)
    # An acted-on order yields no counterfactual: we never learn what would have happened.
    df["label_observed"] = (~df["flagged"]) | df["in_holdout"]
    return df


def naive_censored_training_set(sim: pd.DataFrame) -> pd.DataFrame:
    """What a team gets if they retrain on 'whatever the warehouse recorded'.

    The trap is that flagged orders do not vanish from the warehouse export -- they come
    back stamped 'not returned', because the intervention worked or the sale never
    happened. So the naive set keeps them, with the label silently overwritten to 0.
    """
    df = sim.copy()
    df.loc[df["flagged"] & ~df["in_holdout"], "returned"] = 0
    return df


def holdout_training_set(sim: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Unbiased set: drop censored rows, up-weight the always-approve holdout by 1/p."""
    df = sim[sim["label_observed"]].copy()
    frac = float(sim.loc[sim["flagged"], "in_holdout"].mean()) if sim["flagged"].any() else 0.0
    # Each held-out flagged order stands in for ~1/frac flagged orders we cannot observe.
    w = np.where(df["in_holdout"].to_numpy(), 1.0 / frac if frac > 0 else 1.0, 1.0)
    return df, w


def run_experiment(
    test_features: pd.DataFrame,
    proba: np.ndarray,
    cost: CostModel,
    cfg: Config,
    threshold: float,
    action: str | None = None,
) -> SelectiveLabelsResult:
    """Split the test window, censor period 1, retrain two ways, judge on clean period 2."""
    action = action or str(cfg["selective_labels"]["action"])
    holdout_frac = float(cfg["selective_labels"]["holdout_frac"])
    seed = int(cfg["seed"])

    df = test_features.sort_values("order_date", kind="stable").reset_index(drop=True)
    cut = len(df) // 2
    p1_raw, p2 = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    p1_proba = np.asarray(proba)[: len(p1_raw)]

    sim = simulate_censoring(p1_raw, p1_proba, threshold, holdout_frac, seed)
    true_base = float(sim["returned"].mean())

    naive = naive_censored_training_set(sim)
    hold, weights = holdout_training_set(sim)

    # The evaluation set is period 2, untouched by any policy.
    y2 = p2["returned"].to_numpy()
    X2 = _matrix(p2)
    value2 = cost.to_inr(p2["order_value_gbp"])

    rows = []

    def _score(name: str, p2_proba: np.ndarray, note: str) -> None:
        panel = evaluate(y2, p2_proba, threshold)
        bd = cost.evaluate_policy(y2, p2_proba >= threshold, value2, action, threshold=threshold)
        rows.append(
            {
                "model": name,
                "auc_pr_period2": panel.auc_pr,
                "roc_auc_period2": panel.roc_auc,
                "precision_period2": panel.precision,
                "recall_period2": panel.recall,
                "mean_score_period2": float(p2_proba.mean()),
                "orders_flagged_period2": bd.n_flagged,
                "flag_rate_period2": bd.flag_rate,
                "savings_inr_period2": bd.savings,
                "note": note,
            }
        )

    # 1. The incumbent, carried forward unchanged.
    _score(
        "original (no retrain)",
        np.asarray(proba)[len(p1_raw):],
        "Reference: the deployed model, never exposed to censored data.",
    )

    # 2. Retrained on censored data -- the failure mode.
    m_naive = _fit(naive, cfg)
    _score(
        "retrained on censored labels",
        m_naive.predict_proba(X2)[:, 1],
        "Flagged orders re-enter training stamped 'not returned'.",
    )

    # 3. Retrained with the always-approve holdout + IPW -- the fix.
    m_hold = _fit(hold, cfg, weights)
    _score(
        f"retrained w/ {holdout_frac:.0%} always-approve holdout",
        m_hold.predict_proba(X2)[:, 1],
        "Censored rows dropped; holdout up-weighted by 1/p to restore the flagged population.",
    )

    comparison = pd.DataFrame(rows)

    # Price the mitigation: returns we knowingly let through to keep learning.
    held = sim[sim["in_holdout"]]
    a = cost.resolve(action)
    holdout_cost = float(
        (
            cost.fn_cost(cost.to_inr(held.loc[held["returned"] == 1, "order_value_gbp"]))
            * a.effectiveness
        ).sum()
    )

    return SelectiveLabelsResult(
        action=action,
        threshold=float(threshold),
        holdout_frac=holdout_frac,
        period1_range=(str(p1_raw["order_date"].min()), str(p1_raw["order_date"].max())),
        period2_range=(str(p2["order_date"].min()), str(p2["order_date"].max())),
        n_period1=int(len(p1_raw)),
        n_period2=int(len(p2)),
        n_flagged_p1=int(sim["flagged"].sum()),
        n_censored_p1=int((sim["flagged"] & ~sim["in_holdout"]).sum()),
        n_holdout_p1=int(sim["in_holdout"].sum()),
        observed_base_rate_naive=float(naive["returned"].mean()),
        observed_base_rate_holdout=float(
            np.average(hold["returned"].to_numpy(), weights=weights)
        ),
        true_base_rate_p1=true_base,
        comparison=comparison,
        holdout_cost_inr=holdout_cost,
        notes={
            "holdout_cost_explanation": (
                "Expected rupee value of returns the always-approve holdout knowingly lets "
                "through: the price of keeping an unbiased label stream."
            )
        },
    )


def verdict(result: SelectiveLabelsResult) -> str:
    """The paragraph-sized finding, for the README."""
    c = result.comparison.set_index("model")
    naive_row = c.loc["retrained on censored labels"]
    fix_row = c.loc[[i for i in c.index if "holdout" in i][0]]
    orig = c.loc["original (no retrain)"]
    return (
        f"Acting on {result.n_flagged_p1:,} of {result.n_period1:,} period-1 orders censors "
        f"{result.n_censored_p1:,} labels. The observed base rate collapses from "
        f"{result.true_base_rate_p1:.2%} to {result.observed_base_rate_naive:.2%}, and a model "
        f"retrained on that data scores AUC-PR {naive_row['auc_pr_period2']:.3f} on clean "
        f"period-2 data versus {orig['auc_pr_period2']:.3f} for the incumbent. With a "
        f"{result.holdout_frac:.0%} always-approve holdout and inverse-propensity weighting the "
        f"reconstructed base rate is {result.observed_base_rate_holdout:.2%} and the retrained "
        f"model recovers to {fix_row['auc_pr_period2']:.3f}."
    )


__all__ = [
    "SelectiveLabelsResult",
    "holdout_training_set",
    "naive_censored_training_set",
    "run_experiment",
    "simulate_censoring",
    "verdict",
]
