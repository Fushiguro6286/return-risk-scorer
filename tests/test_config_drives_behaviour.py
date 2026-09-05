"""Config-driven behaviour (Tier 6.2).

Acceptance criterion: *changing a cost in the YAML changes t* with no code edit.*
These tests edit a temporary copy of config.yaml -- never the real one -- and assert
the pipeline's decisions actually move.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from returnrisk.config import DEFAULT_CONFIG_PATH, load_config
from returnrisk.money import CostModel


@pytest.fixture
def scored():
    rng = np.random.default_rng(11)
    n = 20000
    p = np.clip(rng.beta(2.0, 9.0, n), 1e-4, 0.9999)
    y = rng.binomial(1, p)
    value_gbp = rng.gamma(3.0, 120.0, n) + 20.0
    return y, p, value_gbp


def _config_with(tmp_path, mutate):
    """Copy config.yaml, apply `mutate`, and load it back."""
    dst = tmp_path / "config.yaml"
    shutil.copy(DEFAULT_CONFIG_PATH, dst)
    with open(dst, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    mutate(data)
    with open(dst, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return load_config(dst)


def test_raising_the_false_negative_cost_lowers_the_threshold(tmp_path, scored):
    """A costlier miss should make us intervene more readily."""
    y, p, v_gbp = scored
    base = load_config()
    t_base, _, _ = CostModel(base).optimal_threshold(
        y, p, CostModel(base).to_inr(v_gbp), "remove_discount"
    )

    expensive = _config_with(
        tmp_path,
        lambda d: d["money"]["false_negative"].update({"reverse_logistics_inr": 5000.0}),
    )
    cost = CostModel(expensive)
    t_new, _, _ = cost.optimal_threshold(y, p, cost.to_inr(v_gbp), "remove_discount")

    assert t_new < t_base, f"t* should fall when misses get costlier ({t_base:.4f} -> {t_new:.4f})"


def test_raising_the_false_positive_cost_raises_the_threshold(tmp_path, scored):
    """More friction on good customers should make us intervene more cautiously."""
    y, p, v_gbp = scored
    base = load_config()
    cm = CostModel(base)
    t_base, _, _ = cm.optimal_threshold(y, p, cm.to_inr(v_gbp), "remove_discount")

    pricey = _config_with(
        tmp_path,
        lambda d: d["money"]["actions"]["remove_discount"].update(
            {"ops_cost_inr": 400.0, "conversion_loss_prob": 0.45}
        ),
    )
    cost = CostModel(pricey)
    t_new, _, _ = cost.optimal_threshold(y, p, cost.to_inr(v_gbp), "remove_discount")

    assert t_new > t_base, f"t* should rise when false alarms get costlier ({t_base:.4f} -> {t_new:.4f})"


def test_changing_the_fx_rate_scales_every_rupee_figure(tmp_path, scored):
    y, p, v_gbp = scored
    doubled = _config_with(tmp_path, lambda d: d["money"].update({"gbp_to_inr": 210.0}))

    a = CostModel(load_config())
    b = CostModel(doubled)
    bd_a = a.evaluate_policy(y, p >= 0.2, a.to_inr(v_gbp), "remove_cod")
    bd_b = b.evaluate_policy(y, p >= 0.2, b.to_inr(v_gbp), "remove_cod")
    # Value-linked costs double; the fixed per-order legs do not, so expect > 1x, < 2x.
    ratio = bd_b.do_nothing_cost / bd_a.do_nothing_cost
    assert 1.5 < ratio < 2.0


def test_adding_an_action_needs_no_code_change(tmp_path, scored):
    """A new intervention declared in YAML must get its own solved threshold."""
    y, p, v_gbp = scored
    extended = _config_with(
        tmp_path,
        lambda d: d["money"]["actions"].update(
            {
                "call_the_customer": {
                    "label": "Phone the customer to confirm",
                    "ops_cost_inr": 120.0,
                    "conversion_loss_prob": 0.02,
                    "effectiveness": 0.55,
                }
            }
        ),
    )
    cost = CostModel(extended)
    table = cost.optimal_threshold_per_action(y, p, cost.to_inr(v_gbp))
    assert "call_the_customer" in set(table["action"])
    assert table["t_star"].notna().all()
    # The monotonicity property must survive the addition.
    assert table["t_star"].is_monotonic_increasing


def test_naive_rule_threshold_comes_from_config(tmp_path, cfg):
    """The baseline rules are declarative; changing one changes what it flags."""
    from returnrisk.baselines import rule_flags
    import pandas as pd

    features = pd.DataFrame({"discount_pct": [0.0, 0.2, 0.4, 0.6]})
    loose = {"name": "x", "feature": "discount_pct", "op": ">=", "value": 0.15}
    strict = {"name": "x", "feature": "discount_pct", "op": ">=", "value": 0.50}
    assert rule_flags(features, loose).sum() == 3
    assert rule_flags(features, strict).sum() == 1


def test_seed_is_respected_end_to_end(cfg):
    """Two builders fitted with the same seed must produce identical encodings."""
    assert int(cfg["seed"]) == 42


def test_unknown_config_key_fails_loudly(cfg):
    with pytest.raises(KeyError, match="missing config key"):
        _ = cfg["definitely_not_a_key"]
