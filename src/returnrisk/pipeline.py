"""End-to-end run: raw invoices in, every report artifact and the served model out.

`python run_demo.py` calls `run()` and nothing else. One command, fixed seeds, all
outputs regenerated -- so a reviewer can reproduce every number in the README from a
clean clone rather than taking our word for it.

The order below is the order of the argument, not just of the code: label first, then
prove there is no leakage, then split honestly, then calibrate, then -- only once the
probability means something -- put money on it.
"""
from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import baselines, capacity, plots, segments, selective_labels, sensitivity, stability
from .config import Config, load_config
from .data.labeling import build_labelled_orders, summarise
from .data.loader import load_transactions
from .explain import Explainer, leakage_audit
from .features import FeatureBuilder, build_feature_frame, select_features
from .metrics import (
    calibration_frame,
    evaluate,
    expected_calibration_error,
    lift_table,
    pr_curve_frame,
    roc_curve_frame,
)
from .model import TrainedModel, feature_importance, train_and_calibrate
from .money import CostModel, format_inr
from .scoring import PROFILE_FILE, build_profile_store
from .split import temporal_split


@dataclass
class Artifacts:
    """Registry of what was written, so the smoke test can assert on it."""

    reports_dir: Path
    files: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, path: Path) -> Path:
        self.files.append(path)
        return path

    def table(self, df: pd.DataFrame, name: str) -> Path:
        p = self.reports_dir / name
        df.to_csv(p, index=False)
        return self.add(p)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _log(msg: str) -> None:
    print(msg, flush=True)


def run(cfg: Config | None = None, *, quick: bool = False) -> Artifacts:
    """Execute the whole pipeline and write every artifact.

    `quick` shrinks the expensive extras (SHAP sampling, sensitivity grid) for the
    smoke test; it never changes the model, the split or the headline metrics.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = cfg or load_config()
    _seed_everything(int(cfg["seed"]))
    plots.use_house_style()

    reports = cfg.dir("reports_dir")
    art = Artifacts(reports_dir=reports)
    summary: dict[str, Any] = {"config": cfg.as_dict()}
    t0 = time.time()

    # ---------------------------------------------------------------- 0.1 data + label
    _log("[1/12] Loading transactions and building the label...")
    txn = load_transactions(cfg)
    orders, label_report = build_labelled_orders(txn, cfg)
    _log("       " + summarise(label_report))
    summary["label"] = label_report.to_dict()
    summary["data"] = {
        "source": cfg["data"]["source"],
        "n_transaction_lines": int(len(txn)),
        "n_orders": int(len(orders)),
        "n_customers": int(orders["customer_id"].nunique()),
        "date_min": str(txn["invoice_date"].min()),
        "date_max": str(txn["invoice_date"].max()),
    }

    # ------------------------------------------------------------- 0.2 features + split
    _log("[2/12] Building leakage-safe features and the temporal split...")
    feat = build_feature_frame(orders)
    sp = temporal_split(feat, cfg)
    _log("\n".join("       " + ln for ln in sp.describe().splitlines()))
    summary["split"] = sp.meta

    lines = txn[~txn["is_cancellation"]]
    builder = FeatureBuilder(cfg).fit(sp.train, lines)
    train = builder.transform(sp.train, lines, is_train=True)
    calib = builder.transform(sp.calib, lines)
    test = builder.transform(sp.test, lines)
    select_features(train)  # runs the post-checkout allow-list assertion

    # -------------------------------------------------------------- 0.3 train + calibrate
    _log("[3/12] Training LightGBM and calibrating on the held-out slice...")
    model, brier = train_and_calibrate(train, calib, builder, cfg)
    summary["calibration"] = brier
    _log(
        f"       Brier {brier['brier_uncalibrated_calib_slice']:.4f} -> "
        f"{brier['brier_calibrated_calib_slice']:.4f} after isotonic calibration"
    )

    proba = model.predict_proba(test)
    proba_raw = model.predict_proba_raw(test)
    y = test["returned"].to_numpy()

    cost = CostModel(cfg)
    symbol = cost.symbol
    value_inr = cost.to_inr(test["order_value_gbp"])

    # -------------------------------------------------------------- 1.2 per-action t*
    _log("[4/12] Solving the cost-optimal threshold for each intervention...")
    action_table = cost.optimal_threshold_per_action(y, proba, value_inr)
    art.table(action_table, "per_action_thresholds.csv")
    default_action = cost.default_action
    t_star = float(action_table.loc[action_table["action"] == default_action, "t_star"].iloc[0])
    monotone = bool(
        action_table["t_star"].is_monotonic_increasing
        and action_table["orders_flagged"].is_monotonic_decreasing
    )
    summary["per_action_thresholds"] = {
        "rows": action_table.to_dict(orient="records"),
        "monotone_cheaper_action_lower_threshold": monotone,
        "default_action": default_action,
        "t_star": t_star,
    }
    for _, r in action_table.iterrows():
        _log(
            f"       {r['action']:<18} FP cost {format_inr(r['fp_cost_mean_inr'], symbol):>12}"
            f"   t*={r['t_star']:.3f}   flags {int(r['orders_flagged']):>6,}"
            f"   saves {format_inr(r['savings_inr'], symbol)}"
        )
    _log(f"       monotone (cheaper action -> lower threshold): {monotone}")

    # -------------------------------------------------------------- 0.4 metric panel
    _log("[5/12] Emitting the honest metric panel...")
    panel = evaluate(y, proba, t_star)
    ece = expected_calibration_error(y, proba)
    _log("       " + panel.summary())
    summary["metrics_test"] = {**panel.to_dict(), "ece": ece}
    summary["metrics_test_uncalibrated"] = evaluate(y, proba_raw, t_star).to_dict()

    pr = pr_curve_frame(y, proba)
    art.table(pr, "pr_curve.csv")
    art.table(roc_curve_frame(y, proba), "roc_curve.csv")
    cal_frame = calibration_frame(y, proba)
    art.table(cal_frame, "calibration_curve.csv")
    lift = lift_table(y, proba)
    art.table(lift, "lift_table.csv")

    art.add(plots.pr_curve(pr, panel.base_rate, panel.auc_pr, reports / "pr_curve.png"))
    art.add(plots.calibration_curve(cal_frame, reports / "calibration_curve.png", ece))
    art.add(
        plots.calibration_before_after(
            calibration_frame(y, proba_raw), cal_frame,
            reports / "calibration_before_after.png", brier,
        )
    )
    art.add(plots.lift_chart(lift, reports / "lift_by_decile.png"))

    # -------------------------------------------------------------- 1.1 the money layer
    _log("[6/12] Pricing every error in rupees...")
    sweeps, t_stars = {}, {}
    for key in cost.actions:
        _t, _bd, sweep = cost.optimal_threshold(y, proba, value_inr, key)
        sweeps[key] = sweep
        t_stars[key] = _t
        art.table(sweep, f"cost_sweep_{key}.csv")

    bd = cost.evaluate_policy(y, proba >= t_star, value_inr, default_action, threshold=t_star)
    summary["money"] = {
        "action": default_action,
        "threshold": t_star,
        "confusion_matrix_inr": {
            "true_negative": bd.cost_tn, "false_positive": bd.cost_fp,
            "false_negative": bd.cost_fn, "true_positive": bd.cost_tp,
        },
        "counts": {"tn": bd.n_tn, "fp": bd.n_fp, "fn": bd.n_fn, "tp": bd.n_tp},
        **bd.to_dict(),
    }
    art.add(
        plots.money_confusion_matrix(
            bd, reports / "money_confusion_matrix.png",
            action_label=cost.actions[default_action].label, symbol=symbol,
        )
    )
    art.add(plots.cost_curve(sweeps, t_stars, reports / "cost_vs_threshold.png", symbol))
    art.add(plots.per_action_thresholds(action_table, reports / "per_action_thresholds.png", symbol))
    _log(
        f"       Do nothing {format_inr(bd.do_nothing_cost, symbol)} -> "
        f"policy {format_inr(bd.total_cost, symbol)} "
        f"(saves {format_inr(bd.savings, symbol)}, "
        f"{bd.savings / bd.do_nothing_cost:.1%})"
    )

    # -------------------------------------------------------------- 1.3 baselines
    _log("[7/12] Comparing against do-nothing and a real merchant rule...")
    base_table = baselines.compare_baselines(test, proba, cost, cfg, t_star, default_action)
    art.table(base_table, "baseline_comparison.csv")
    art.add(plots.baseline_comparison(base_table, reports / "baseline_comparison.png", symbol))
    head = baselines.headline_comparison(base_table, cfg)
    summary["baselines"] = {
        "rows": base_table.to_dict(orient="records"),
        "rule_diagnostics": baselines.rule_diagnostics(test, cfg),
        "headline": head,
    }
    for _, r in base_table.iterrows():
        _log(
            f"       {r['policy'][:56]:<58} {format_inr(r['total_cost_inr'], symbol):>12}"
            f"   (P {r['precision']:.3f} R {r['recall']:.3f})"
        )
    _log(
        f"       Model beats every rule: {head['model_beats_all_rules']}; "
        f"best rule was '{head['best_rule'][:40]}' "
        f"(model saves a further {format_inr(head['savings_vs_best_rule_inr'], symbol)})"
    )

    # -------------------------------------------------------------- 1.4 sensitivity
    _log("[8/12] Sweeping plausible cost assumptions...")
    grid = sensitivity.cost_sensitivity(test, proba, cost, cfg, default_action)
    art.table(grid, "cost_sensitivity.csv")
    art.add(
        plots.sensitivity_heatmap(
            sensitivity.heatmap_pivot(grid, "t_star"),
            reports / "cost_sensitivity_heatmap.png",
            "Optimal threshold t* across plausible cost assumptions", "{:.3f}",
        )
    )
    art.add(
        plots.sensitivity_heatmap(
            sensitivity.heatmap_pivot(grid, "savings_pct"),
            reports / "cost_sensitivity_savings.png",
            "Share of return cost avoided, across the same grid", "{:.0%}",
        )
    )
    sens_summary = sensitivity.sensitivity_summary(grid)
    summary["sensitivity"] = sens_summary
    _log(
        f"       model saves money in {sens_summary['n_cells_profitable']}/"
        f"{sens_summary['n_cells']} cost scenarios; "
        f"t* ranges {sens_summary['t_star_min']:.3f}-{sens_summary['t_star_max']:.3f}"
    )

    # -------------------------------------------------------------- 2.1 stability
    _log("[9/12] Backtesting month by month across the test period...")
    bt = stability.monthly_backtest(test, proba, cost, cfg, t_star, default_action)
    art.table(bt, "stability_backtest.csv")
    art.add(plots.stability_plot(bt, reports / "stability.png", symbol))
    verdict = stability.stability_verdict(bt)
    summary["stability"] = {
        "monthly": bt.to_dict(orient="records"),
        "auc_pr_trend": stability.trend(bt, "auc_pr"),
        "savings_trend": stability.trend(bt, "savings_inr"),
        "verdict": verdict,
    }
    _log("       " + verdict)

    # -------------------------------------------------------- 2.3 / 2.4 segments + fairness
    _log("[10/12] Breaking performance down by segment...")
    seg = segments.segment_metrics(test, proba, cost, t_star, default_action)
    art.table(seg, "segment_metrics.csv")
    weak = segments.weakest_segment(seg)
    weak_name = f"{weak['dimension']} = {weak['segment']}" if not weak.empty else ""
    art.add(plots.segment_plot(seg, reports / "segment_metrics.png", weak_name))

    fair = segments.fairness_table(test, proba, t_star)
    art.table(fair, "fairness_by_segment.csv")
    art.add(plots.fairness_plot(fair, reports / "fairness_by_segment.png"))
    summary["segments"] = {
        "rows": seg.to_dict(orient="records"),
        "weakest": segments.weakness_sentence(seg),
    }
    summary["fairness"] = {
        "rows": fair.to_dict(orient="records"),
        "disparity": segments.disparity_report(fair),
        "verdict": segments.disparity_sentence(fair),
    }
    _log("       " + segments.weakness_sentence(seg))
    _log("       " + segments.disparity_sentence(fair))

    # -------------------------------------------------------------- 2.5 capacity
    _log("[11/12] Capacity-constrained (top-k) operating mode...")
    topk = capacity.precision_at_k(test, proba, cost, cfg)
    art.table(topk, "precision_at_k.csv")
    art.add(plots.precision_at_k_plot(topk, panel.base_rate, reports / "precision_at_k.png"))
    mode_note = capacity.recommend_mode(topk, bd.flag_rate)
    summary["capacity"] = {"rows": topk.to_dict(orient="records"), "recommendation": mode_note}
    _log("       " + mode_note)

    # -------------------------------------------------------------- 2.2 selective labels
    _log("[12/12] Selective-labels experiment (the feedback loop)...")
    sl = selective_labels.run_experiment(test, proba, cost, cfg, t_star)
    art.table(sl.comparison, "selective_labels_comparison.csv")
    art.add(plots.selective_labels_plot(sl, reports / "selective_labels.png"))
    sl_verdict = selective_labels.verdict(sl)
    summary["selective_labels"] = {**sl.to_dict(), "verdict": sl_verdict}
    _log("       " + sl_verdict)

    # -------------------------------------------------------------- 4.1 / 4.2 explainability
    _log("       SHAP explanations and the leakage audit...")
    explainer = Explainer(model)
    shap_rows = 800 if quick else 4000
    global_imp = explainer.global_importance(test, max_rows=shap_rows)
    art.table(global_imp, "shap_global_importance.csv")
    art.add(plots.shap_bar(global_imp, reports / "shap_global_importance.png"))
    try:
        art.add(
            plots.shap_beeswarm(
                explainer, test, reports / "shap_summary.png",
                max_rows=600 if quick else 2500,
            )
        )
    except Exception as exc:  # pragma: no cover - beeswarm is a nice-to-have
        _log(f"       (beeswarm skipped: {exc})")

    gain = feature_importance(model)
    art.table(gain, "gain_importance.csv")
    audit = leakage_audit(global_imp, gain, panel.auc_pr, panel.base_rate)
    summary["leakage_audit"] = audit
    _log("       " + audit["verdict"])

    # Sample reason codes, so the README can show what a merchant actually sees.
    flagged = test[proba >= t_star].head(200)
    if len(flagged):
        codes = explainer.reason_codes(flagged, max_reasons=3)
        sample = [
            {
                "order_id": str(flagged.iloc[i]["order_id"]),
                "risk_score": float(proba[proba >= t_star][i]),
                "order_value_gbp": float(flagged.iloc[i]["order_value_gbp"]),
                "reasons": [r["reason"] for r in codes[i]],
            }
            for i in range(min(10, len(flagged)))
        ]
        summary["reason_code_examples"] = sample
        pd.DataFrame(
            [{**s, "reasons": " | ".join(s["reasons"])} for s in sample]
        ).to_csv(reports / "reason_code_examples.csv", index=False)
        art.add(reports / "reason_code_examples.csv")

    # -------------------------------------------------------------- persist
    model.thresholds = {k: float(v) for k, v in t_stars.items()}
    model.metadata.update(
        {
            "test_auc_pr": panel.auc_pr,
            "test_base_rate": panel.base_rate,
            "default_action": default_action,
            "label_definition": summary["label"]["label_definition"],
        }
    )
    model_path = model.save(cfg.get_path("model_file"))
    art.add(model_path)
    _log(f"       Model saved to {model_path}")

    # Customer history the serving path cannot reconstruct from a request payload.
    profiles = build_profile_store(feat)
    profile_path = cfg.dir("models_dir") / PROFILE_FILE
    profiles.to_parquet(profile_path, index=False)
    art.add(profile_path)
    summary["serving"] = {
        "model_file": str(model_path),
        "profile_store": str(profile_path),
        "n_customer_profiles": int(len(profiles)),
        "thresholds": model.thresholds,
    }

    # A scored batch the dashboard can open without recomputing anything.
    scored = test.copy()
    scored["risk_score"] = proba
    scored["order_value_inr"] = value_inr
    scored["expected_loss_inr"] = proba * cost.fn_cost(value_inr)
    art.table(
        scored[
            [
                "order_id", "customer_id", "order_date", "order_value_gbp", "order_value_inr",
                "risk_score", "expected_loss_inr", "returned", "n_lines", "total_quantity",
                "discount_pct", "prior_orders", "prior_return_rate", "is_new_customer",
                "country", "region", "top_category",
            ]
        ],
        "scored_test_orders.csv",
    )

    # A small, committable sample so the repo is explorable without the 45MB download.
    sample_dir = Path(cfg["data"]["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample_orders.csv"
    test.head(500).to_csv(sample_path, index=False)
    art.add(sample_path)

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    summary["artifacts"] = [str(p.relative_to(Path.cwd())) if p.is_absolute() and str(p).startswith(str(Path.cwd())) else str(p) for p in art.files]
    art.summary = summary

    summary_path = reports / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, indent=2, default=str)
    art.add(summary_path)

    _log(f"\nDone in {summary['runtime_seconds']}s. {len(art.files)} artifacts in {reports}/")
    return art


def _jsonable(obj: Any) -> Any:
    """NumPy and pandas scalars are not JSON-serialisable; coerce them recursively."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = ["Artifacts", "run"]
