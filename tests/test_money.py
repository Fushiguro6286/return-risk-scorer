"""Tests for the cost layer -- the part every rupee figure in the README rests on."""
from __future__ import annotations

import numpy as np
import pytest

from returnrisk.money import CostModel, default_grid, format_inr


@pytest.fixture
def scored():
    """A deterministic, *perfectly calibrated* score/label/value triple.

    Labels are drawn FROM the scores, so P(y=1 | p) == p by construction. That matters:
    the cost algebra assumes calibrated probabilities, so testing it against an
    arbitrary correlated score would be testing the wrong thing.
    """
    rng = np.random.default_rng(7)
    n = 20000
    p = np.clip(rng.beta(2.0, 9.0, n), 1e-4, 0.9999)  # mean ~0.18, like the real base rate
    y = rng.binomial(1, p)
    value_gbp = rng.gamma(3.0, 120.0, n) + 20.0
    return y, p, value_gbp


def test_cost_cells_are_arithmetically_consistent(cfg, scored):
    y, p, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    bd = cost.evaluate_policy(y, p >= 0.2, v, "remove_discount", threshold=0.2)

    assert bd.n_tp + bd.n_fp + bd.n_fn + bd.n_tn == bd.n
    assert bd.n_tp + bd.n_fp == bd.n_flagged
    assert bd.total_cost == pytest.approx(bd.cost_tp + bd.cost_fp + bd.cost_fn)
    assert bd.cost_tn == 0.0
    assert bd.savings == pytest.approx(bd.do_nothing_cost - bd.total_cost)
    # Correctly leaving a good order alone is the only free cell.
    assert bd.cost_fp > 0 and bd.cost_fn > 0


def test_do_nothing_is_the_flag_nothing_policy(cfg, scored):
    y, _, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    bd = cost.evaluate_policy(y, np.zeros(len(y), dtype=bool), v, "remove_cod")
    assert bd.total_cost == pytest.approx(cost.do_nothing_cost(y, v))
    assert bd.savings == pytest.approx(0.0)


def test_true_positive_counterfactual_exceeds_its_cost(cfg, scored):
    """Catching a return must cost less than letting it through, or the action is pointless."""
    y, p, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    bd = cost.evaluate_policy(y, p >= 0.2, v, "remove_cod", threshold=0.2)
    assert bd.cost_tp < bd.cost_tp_counterfactual


def test_thresholds_are_monotone_in_false_positive_cost(cfg, scored):
    """Tier 1.2's headline claim: a costlier mistake buys a higher bar.

    Asserted on the solved thresholds, not on the config, so it would catch a sign
    error in the sweep itself.
    """
    y, p, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    table = cost.optimal_threshold_per_action(y, p, v)

    assert table["fp_cost_mean_inr"].is_monotonic_increasing
    assert table["t_star"].is_monotonic_increasing, (
        f"cheaper actions must have lower thresholds; got {table[['action', 't_star']].to_dict()}"
    )
    assert table["orders_flagged"].is_monotonic_decreasing


def test_analytic_threshold_leaves_no_money_on_the_table(cfg, scored):
    """The closed form and the brute-force sweep must agree on a constant-value book.

    Compared on *cost*, not on the threshold itself: near the optimum the cost curve is
    flat, so two thresholds can differ visibly while being worth the same. Cost is what
    we actually care about, and it is the stricter check on the algebra.
    """
    y, p, _ = scored
    cost = CostModel(cfg)
    unit = 500.0 * cost.gbp_to_inr
    v = np.full(len(y), unit)
    for action in cost.actions:
        analytic = cost.analytic_threshold(unit, action)
        _swept, best, _ = cost.optimal_threshold(y, p, v, action)
        at_analytic = cost.evaluate_policy(y, p >= analytic, v, action, threshold=analytic)
        assert at_analytic.total_cost <= best.total_cost * 1.01, (
            f"{action}: closed-form threshold {analytic:.4f} costs "
            f"{at_analytic.total_cost:,.0f} vs swept optimum {best.total_cost:,.0f}"
        )


def test_analytic_thresholds_are_ordered_by_false_positive_cost(cfg):
    """The closed form itself must be monotone in the cost of a false positive."""
    cost = CostModel(cfg)
    unit = 500.0 * cost.gbp_to_inr
    thresholds = [cost.analytic_threshold(unit, a) for a in ("manual_pack_check", "remove_discount", "remove_cod")]
    assert thresholds == sorted(thresholds)


def test_optimal_threshold_beats_neighbouring_thresholds(cfg, scored):
    y, p, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    _t, bd, sweep = cost.optimal_threshold(y, p, v, "remove_discount")
    assert bd.total_cost == pytest.approx(sweep["total_cost"].min())
    assert bd.total_cost <= cost.do_nothing_cost(y, v)


def test_higher_effectiveness_never_costs_more(cfg, scored):
    """A strictly better intervention cannot produce a worse optimum."""
    y, p, v_gbp = scored
    cost = CostModel(cfg)
    v = cost.to_inr(v_gbp)
    a = cost.actions["remove_discount"]

    weak = type(a)(a.key, a.label, a.ops_cost_inr, a.conversion_loss_prob, 0.20)
    strong = type(a)(a.key, a.label, a.ops_cost_inr, a.conversion_loss_prob, 0.80)
    _, bd_weak, _ = cost.optimal_threshold(y, p, v, weak)
    _, bd_strong, _ = cost.optimal_threshold(y, p, v, strong)
    assert bd_strong.total_cost <= bd_weak.total_cost


def test_fn_cost_scales_with_order_value(cfg):
    cost = CostModel(cfg)
    small, large = cost.fn_cost(np.array([1000.0])), cost.fn_cost(np.array([100000.0]))
    assert large[0] > small[0]
    assert small[0] >= cost.reverse_logistics  # the fixed leg is always paid


def test_default_grid_spans_the_unit_interval():
    p = np.random.default_rng(0).beta(2, 8, 500)
    grid = default_grid(p)
    assert grid.min() <= 0.0 and grid.max() >= 1.0
    assert np.all(np.diff(grid) >= 0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(950.0, "Rs.950"), (12_500.0, "Rs.12.5 K"), (250_000.0, "Rs.2.50 L"), (34_000_000.0, "Rs.3.40 Cr")],
)
def test_rupee_formatting(value, expected):
    assert format_inr(value) == expected
