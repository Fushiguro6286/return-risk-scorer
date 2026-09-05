"""Merchant dashboard (Tier 3.2) -- the demo centrepiece.

    streamlit run app/dashboard.py

Six views, in the order a merchant would actually meet them:

1. **Score an order** -- type a basket in, get a score, an action and three reasons.
   Then, below it, what the system is actually *allowed* to do once the guardrails have
   had their say. The gap between those two panels is the argument for the policy layer.
2. **Portfolio** -- the scored test book, sorted by rupees at risk, with the action queue.
3. **Policy simulator** -- the what-if slider. Drag the threshold and watch orders
   flagged, rupees saved and the confusion matrix move together. This is the screen
   worth recording: it turns "the model has AUC-PR 0.34" into "here is what it is worth
   and here is the dial".
4. **Governance & audit trail** -- the guardrails in force, the append-only ledger, its
   hash-chain integrity check, the human review queue, and the dispute evidence pack.
   This is the screen that answers "prove it".
5. **Run on your data** -- upload any transaction export, confirm the column mapping,
   and train the whole thing on it. Also where datasets are removed again.
6. **Reports** -- every chart and table the pipeline wrote, plus the printable report.

Datasets
--------
Every view above reads from whichever dataset is chosen in the picker at the top of the
page. The committed UCI run is one entry in that list; each upload is another, with its
own `reports_user/<key>/`, `models_user/<key>/` and `audit_user/<key>/`. Nothing is
shared between them, which is what makes "remove this dataset" a complete operation and
makes it impossible for an upload to overwrite the baseline the README is written
against. The baseline itself is registered read-only and cannot be deleted here.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from returnrisk.audit import DecisionLedger  # noqa: E402
from returnrisk.config import load_config  # noqa: E402
from returnrisk.data.ingest import (  # noqa: E402
    CANONICAL_FIELDS,
    READABLE_SUFFIXES,
    ColumnMapping,
    list_tables,
    profile,
    read_any,
    suggest_mapping,
    validate,
    with_file_currency,
)
from returnrisk.datasets import (  # noqa: E402
    DatasetRegistry,
    config_for,
    training_config_for,
)
from returnrisk.decision import DecisionService  # noqa: E402
from returnrisk.money import CostModel, format_inr  # noqa: E402
from returnrisk.pipeline import run as run_pipeline  # noqa: E402
from returnrisk.report_print import SECTIONS, build_report_html  # noqa: E402
from returnrisk.responder import Responder  # noqa: E402
from returnrisk.scoring import OrderInput, ScoringService  # noqa: E402

import theme  # noqa: E402


class _StreamlitLog:
    """Adapts the pipeline's `print` logging into a live box on the page."""

    def __init__(self, sink) -> None:
        self._sink = sink

    def write(self, text: str) -> int:
        for line in str(text).splitlines():
            if line.strip():
                self._sink(line)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - required by the file protocol
        return None


def _pretty(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


#: One line per chart, so a reader knows what they are looking at without the README.
CHART_NOTES: dict[str, str] = {
    "money_confusion_matrix": "Every cell priced in rupees. The whole argument in one picture.",
    "calibration_curve": "Predicted probability against observed frequency.",
    "calibration_before_after": "What isotonic calibration fixed.",
    "pr_curve": "Precision against recall. The base rate is the floor.",
    "roc_curve": "The rank-ordering view.",
    "lift_by_decile": "How much risk concentrates in the top deciles.",
    "cost_vs_threshold": "Total cost against threshold. The minimum is t*.",
    "cost_sensitivity_heatmap": "Does it still save money if the costs are wrong?",
    "cost_sensitivity_savings": "Savings across the whole cost grid.",
    "per_action_thresholds": "Each intervention gets its own optimal threshold.",
    "baseline_comparison": "Against the rules a merchant would actually write.",
    "precision_at_k": "If you can only review k% of orders.",
    "segment_metrics": "Where the model is strong and where it is not.",
    "fairness_by_segment": "Who absorbs the friction of a false alarm.",
    "stability": "Month by month across the test period.",
    "selective_labels": "What happens to the labels once you start acting on them.",
    "shap_summary": "Per-order feature contributions.",
    "shap_global_importance": "Which features carry the model.",
}

st.set_page_config(
    page_title="Return-Risk Scorer",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.css(), unsafe_allow_html=True)

INK = "#0b0b0b"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


# --------------------------------------------------------------------------- loading
# Every loader is keyed by dataset. Streamlit's cache is process-wide, so without the
# key in the signature switching datasets would hand back the previous one's model.
@st.cache_resource(show_spinner="Loading the trained model...")
def get_service(dataset_key: str) -> ScoringService:
    dataset = get_registry().get(dataset_key)
    return ScoringService.load(config_for(dataset, load_config()))


@st.cache_data(show_spinner="Loading the scored test book...")
def get_scored_orders(dataset_key: str) -> pd.DataFrame:
    dataset = get_registry().get(dataset_key)
    path = dataset.path("reports") / "scored_test_orders.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["order_date"])


@st.cache_resource(show_spinner="Opening the decision ledger...")
def get_decision_service(dataset_key: str) -> DecisionService:
    """Scoring + guardrails + audit ledger, sharing the already-loaded model.

    The ledger is per-dataset too: decisions recorded against a model trained on one
    merchant's data have no business appearing in another's audit trail.
    """
    dataset = get_registry().get(dataset_key)
    return DecisionService(get_service(dataset_key), config_for(dataset, load_config()))


@st.cache_resource
def get_responder(dataset_key: str) -> Responder:
    dataset = get_registry().get(dataset_key)
    return Responder(config_for(dataset, load_config()))


def get_registry() -> DatasetRegistry:
    """Re-read on every call: uploads and deletions must be visible immediately."""
    return DatasetRegistry()


def _clear_caches() -> None:
    """Drop every cached model/ledger. Called after a train or a delete."""
    get_service.clear()
    get_scored_orders.clear()
    get_decision_service.clear()
    get_responder.clear()


# --------------------------------------------------------------------------- shell
# The sidebar carries identity, context and navigation; the main column shows one screen
# at a time. Nav is a radio rather than tabs because the six views are a sequence -- the
# order is the argument -- and a rail keeps the current position visible while scrolled.
registry = get_registry()
datasets = registry.all()
trained = [d for d in datasets if d.is_trained]

PAGES = [
    ("01", "Score an order"),
    ("02", "Portfolio"),
    ("03", "Policy simulator"),
    ("04", "Governance & audit"),
    ("05", "Run on your data"),
    ("06", "Reports"),
]

with st.sidebar:
    st.markdown(theme.brand(), unsafe_allow_html=True)

    if not trained:
        st.error("No trained model found.")
        st.caption("Run `python run_demo.py`, or upload data on **Run on your data**.")
        st.stop()

    st.markdown(theme.eyebrow("Context"), unsafe_allow_html=True)
    active_key = st.selectbox(
        "Dataset",
        [d.key for d in trained],
        format_func=lambda k: next(d.label for d in trained if d.key == k),
        key="active_dataset",
        label_visibility="collapsed",
        help="Every screen reads from the dataset selected here.",
    )

    dataset = registry.get(active_key)
    try:
        svc = get_service(active_key)
    except FileNotFoundError:
        st.error(f"`{dataset.label}` has no model file. Retrain or remove it.")
        st.stop()

    cfg = svc.cfg
    cost: CostModel = svc.cost
    symbol = cost.symbol
    decider = get_decision_service(active_key)
    responder = get_responder(active_key)

    meta = svc.model.metadata
    chain = decider.ledger.verify()
    n_charts, n_tables = dataset.artifact_counts()

    # Replaces the reference mock's texture block with the same shape made load-bearing:
    # the facts you need before trusting any number on the screen to the right.
    st.markdown(
        theme.panel(
            "Model",
            theme.statrow(
                [
                    theme.stat("AUC-PR", f"{meta.get('test_auc_pr', float('nan')):.3f}",
                               tone="accent"),
                    theme.stat("Base rate", f"{meta.get('test_base_rate', float('nan')):.1%}"),
                ]
            )
            + '<div style="height:11px"></div>'
            + theme.statrow(
                [
                    theme.stat("Ledger", f"{chain.get('n_entries', 0):,}", "entries"),
                    theme.stat("Artifacts", f"{n_charts + n_tables}",
                               f"{n_charts} charts"),
                ]
            )
            + '<div style="height:12px"></div>'
            + (
                theme.chip("Chain intact", "ok", dot=True)
                if chain.get("ok")
                else theme.chip(f"Chain broken @ {chain.get('broken_at')}", "bad", dot=True)
            )
            + ("" if dataset.is_baseline else " " + theme.chip("User data", "info")),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(theme.eyebrow("Views"), unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [f"{n}  {label}" for n, label in PAGES],
        label_visibility="collapsed",
        key="nav",
    )
    page_no = page.split()[0]

st.markdown(
    theme.topbar(
        next(label for n, label in PAGES if n == page_no),
        theme.chip(dataset.label, "info") + " "
        + theme.chip("System active", "ok", dot=True),
    ),
    unsafe_allow_html=True,
)


# ============================================================ 1. score a single order
if page_no == "01":
    left, right = st.columns([1, 1.55], gap="medium")

    # ------------------------------------------------------------------ the order
    with left:
        with st.container(border=True):
            head, clear = st.columns([3, 1])
            head.markdown(
                '<div class="rr-stat-k" style="padding-top:6px">Order parameters</div>',
                unsafe_allow_html=True,
            )
            if clear.button("Clear", key="clear_order"):
                for k in list(st.session_state):
                    if k.startswith("ord_"):
                        del st.session_state[k]
                st.rerun()

            c1, c2 = st.columns(2)
            order_value = c1.number_input(
                "Order value (GBP)", 1.0, 50000.0, 485.0, step=10.0, key="ord_value"
            )
            total_qty = c2.number_input("Total units", 1, 20000, 4, key="ord_qty")
            n_lines = c1.number_input("Distinct SKUs", 1, 400, 3, key="ord_lines")
            discount = c2.slider("Discount %", -0.5, 1.0, 0.38, 0.01, key="ord_disc")

            avg_price = c1.number_input(
                "Avg unit price", 0.0, 5000.0, 121.0, step=0.5, key="ord_avg"
            )
            max_price = c2.number_input(
                "Highest unit price", 0.0, 5000.0, 180.0, step=0.5, key="ord_max"
            )
            min_price = c1.number_input(
                "Lowest unit price", 0.0, 5000.0, 60.0, step=0.5, key="ord_min"
            )
            customer_id = c2.text_input("Customer ID", "17850", key="ord_cust")

            cats = [*svc.model.feature_builder.categories_[:40], "OTHER"]
            countries = [*svc.model.feature_builder.countries_, "OTHER"]
            category = st.selectbox("Product family", cats, index=0, key="ord_cat")
            country = st.selectbox("Destination country", countries, index=0, key="ord_ctry")

            action = st.selectbox(
                "Target intervention (prices risk against)",
                list(cost.actions),
                index=list(cost.actions).index(cost.default_action),
                format_func=lambda k: f"{cost.actions[k].label}  (t* = {svc.threshold_for(k):.3f})",
                key="ord_action",
            )

    # Built once and reused by both panels, so the score and the governed decision can
    # never be describing different orders.
    order_input = OrderInput(
        order_value_gbp=float(order_value),
        n_lines=int(n_lines),
        total_quantity=int(total_qty),
        avg_unit_price=float(avg_price),
        max_unit_price=float(max_price),
        min_unit_price=float(min_price),
        country=country,
        top_category=category,
        discount_pct=float(discount),
        customer_id=customer_id or None,
    )
    result = svc.score(order_input, action=action)
    governed = decider.decide(order_input, action=action, record=False)
    verdict = governed.policy

    # ---------------------------------------------------------------- the decision
    with right:
        contribs = [abs(d.get("contribution", 0.0)) for d in (result.reason_detail or [])]
        ref = f"ORD-{governed.order_fingerprint[:8].upper()}"

        st.markdown(
            '<div class="rr-panel">'
            + theme.verdict(result.flagged, result.recommended_action_label, f"ID: {ref}")
            + '<div class="rr-panel-body">'
            + '<div style="display:flex;gap:34px;flex-wrap:wrap">'
            + (
                '<div style="flex:1 1 260px;min-width:240px">'
                '<div class="rr-stat-k">Predicted return risk</div>'
                f'<div class="rr-score">{result.risk_score * 100:.1f}<small>%</small></div>'
                + theme.gauge(result.risk_score, result.threshold)
                + "</div>"
            )
            + (
                '<div style="flex:0 1 230px">'
                + theme.stat(
                    "Expected loss if unchecked",
                    format_inr(result.expected_loss_inr, symbol),
                    tone="flag" if result.flagged else "",
                )
                + '<div style="height:14px"></div>'
                + theme.statrow(
                    [
                        theme.stat("Threshold (t*)", f"{result.threshold:.1%}"),
                        theme.stat(
                            "Value of acting",
                            format_inr(result.expected_saving_inr, symbol),
                            tone="good" if result.expected_saving_inr > 0 else "",
                        ),
                    ]
                )
                + "</div>"
            )
            + "</div></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            theme.panel("Principal drivers", theme.drivers(result.top_reasons, contribs)),
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------- the governed answer
        # The panel above is what the *model* thinks. This is what the *system* is
        # allowed to do, and the gap between them is the point of the whole layer.
        rules = [h.rule for h in verdict.rules_fired]
        note = verdict.rules_fired[0].explanation if verdict.rules_fired else ""
        if verdict.requires_human_review:
            final_label = f"{verdict.final_action_label} · awaiting human review"
        elif verdict.in_holdout:
            final_label = "Let through (always-approve holdout)"
        else:
            final_label = verdict.final_action_label

        st.markdown(
            theme.panel(
                "Policy enforcement",
                theme.flow(
                    result.recommended_action_label, final_label, rules, note
                )
                + (
                    ""
                    if verdict.model_action == verdict.final_action
                    else '<div class="rr-flow-note" style="margin-top:11px">'
                    f"The model asked for <b>{verdict.model_action}</b>; policy settled on "
                    f"<b>{verdict.final_action}</b>.</div>"
                ),
                aside=(
                    theme.chip("Human review", "flag")
                    if verdict.requires_human_review
                    else theme.chip("Holdout", "info")
                    if verdict.in_holdout
                    else ""
                ),
            ),
            unsafe_allow_html=True,
        )

        # Behind buttons on purpose: Streamlit reruns this script on every widget
        # change, and a live API call per keystroke would be slow and wasteful.
        b1, b2, b3 = st.columns([1, 1, 1.35])
        draft = b1.button("✎  Draft merchant note", key="draft_note")
        commit = b2.button("⛓  Commit to ledger", type="primary", key="commit_dec")
        b3.markdown(
            f'<div style="text-align:right;padding-top:7px" class="rr-flow-note">'
            f"Effectiveness assumed {cost.actions[action].effectiveness:.0%}</div>",
            unsafe_allow_html=True,
        )

        if draft:
            written = responder.merchant_note(governed.to_dict())
            st.markdown(
                theme.panel(
                    "Merchant note",
                    f'<div style="font-size:13px;line-height:1.6;color:var(--text-dim)">'
                    f"{written.text}</div>",
                    aside=theme.chip(written.generated_by, "info"),
                ),
                unsafe_allow_html=True,
            )
            if written.fallback_reason:
                st.caption(
                    f"Deterministic template used ({written.fallback_reason}). The decision "
                    "above is unaffected — the responder only ever writes prose."
                )

        if commit:
            committed = decider.decide(order_input, action=action, record=True)
            st.success(
                f"Recorded as `{committed.decision_id}` at ledger sequence "
                f"{committed.seq}. Hash `{committed.entry_hash[:16]}…`"
            )


# ==================================================================== 2. portfolio view
if page_no == "02":
    scored = get_scored_orders(active_key)
    if scored.empty:
        st.warning(
            f"No scored test book for **{dataset.label}**. Its run predates this view, or "
            "training did not finish - retrain it on the *Run on your data* tab."
        )
    else:
        st.subheader("Held-out test book, ranked by rupees at risk")
        action2 = st.selectbox(
            "Action", list(cost.actions),
            index=list(cost.actions).index(cost.default_action),
            format_func=lambda k: cost.actions[k].label, key="pf_action",
        )
        t = svc.threshold_for(action2)
        df = scored.copy()
        df["flagged"] = df["risk_score"] >= t

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Orders", f"{len(df):,}")
        k2.metric("Flagged", f"{int(df['flagged'].sum()):,}", f"{df['flagged'].mean():.1%} of book")
        k3.metric("Rupees at risk", format_inr(float(df["expected_loss_inr"].sum()), symbol))
        k4.metric(
            "In the queue",
            format_inr(float(df.loc[df["flagged"], "expected_loss_inr"].sum()), symbol),
            help="Expected return cost sitting inside the flagged queue",
        )

        only_flagged = st.checkbox("Show only the action queue", value=True)
        view = df[df["flagged"]] if only_flagged else df
        view = view.sort_values("expected_loss_inr", ascending=False).head(400)

        st.dataframe(
            view[
                ["order_id", "order_date", "country", "top_category", "order_value_gbp",
                 "risk_score", "expected_loss_inr", "prior_orders", "prior_return_rate",
                 "discount_pct", "flagged", "returned"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=float(df["risk_score"].max()), format="%.3f"
                ),
                "expected_loss_inr": st.column_config.NumberColumn(
                    f"Expected loss ({symbol})", format="%.0f"
                ),
                "order_value_gbp": st.column_config.NumberColumn("Value (GBP)", format="%.0f"),
                "prior_return_rate": st.column_config.NumberColumn("Past return rate", format="%.2f"),
                "discount_pct": st.column_config.NumberColumn("Discount", format="%.2f"),
                "returned": st.column_config.CheckboxColumn("Actually returned", disabled=True),
            },
        )
        st.caption(
            "`Actually returned` is the held-out outcome, shown here only because this is a "
            "backtest. It is not available at scoring time."
        )


# =============================================================== 3. what-if simulator
if page_no == "03":
    scored = get_scored_orders(active_key)
    if scored.empty:
        st.warning(
            f"No scored test book for **{dataset.label}**. Its run predates this view, or "
            "training did not finish - retrain it on the *Run on your data* tab."
        )
    else:
        st.subheader("Move the threshold and watch the money move")

        c1, c2 = st.columns([1, 2])
        action3 = c1.selectbox(
            "Intervention", list(cost.actions),
            index=list(cost.actions).index(cost.default_action),
            format_func=lambda k: cost.actions[k].label, key="wi_action",
        )
        t_star = svc.threshold_for(action3)
        threshold = c2.slider(
            "Risk-score threshold", 0.0, 1.0, float(round(t_star, 3)), 0.005,
            help=f"The cost-optimal t* for this action is {t_star:.3f}",
        )

        y = scored["returned"].to_numpy()
        p = scored["risk_score"].to_numpy()
        value_inr = scored["order_value_inr"].to_numpy()
        bd = cost.evaluate_policy(y, p >= threshold, value_inr, action3, threshold=threshold)
        best = cost.evaluate_policy(y, p >= t_star, value_inr, action3, threshold=t_star)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orders flagged", f"{bd.n_flagged:,}", f"{bd.flag_rate:.1%} of book")
        m2.metric(
            f"Saved vs doing nothing ({symbol})",
            format_inr(bd.savings, symbol),
            delta=format_inr(bd.savings - best.savings, symbol) + " vs t*",
        )
        m3.metric("Precision", f"{bd.precision:.1%}")
        m4.metric("Recall", f"{bd.recall:.1%}")

        if abs(threshold - t_star) < 1e-9:
            st.success(f"You are at the cost-optimal threshold t* = {t_star:.3f}.", icon="🎯")
        elif bd.savings < best.savings:
            st.info(
                f"Cost-optimal is t* = {t_star:.3f}, worth "
                f"{format_inr(best.savings - bd.savings, symbol)} more than this setting.",
                icon="ℹ️",
            )

        st.markdown("#### Rupee confusion matrix at this threshold")
        matrix = pd.DataFrame(
            [
                [
                    f"{format_inr(bd.cost_tn, symbol)}\n{bd.n_tn:,} orders",
                    f"{format_inr(bd.cost_fp, symbol)}\n{bd.n_fp:,} orders",
                ],
                [
                    f"{format_inr(bd.cost_fn, symbol)}\n{bd.n_fn:,} orders",
                    f"{format_inr(bd.cost_tp, symbol)}\n{bd.n_tp:,} orders",
                ],
            ],
            index=["Did NOT return", "Actually returned"],
            columns=["Not flagged", "Flagged"],
        )
        st.table(matrix)
        st.caption(
            f"Do nothing: {format_inr(bd.do_nothing_cost, symbol)}  ·  "
            f"This policy: {format_inr(bd.total_cost, symbol)}  ·  "
            f"Caught orders would have cost {format_inr(bd.cost_tp_counterfactual, symbol)} unflagged."
        )

        st.markdown("#### Cost against threshold")
        sweep = cost.sweep(y, p, value_inr, action3)
        chart = sweep[["threshold", "total_cost"]].copy()
        chart["Total cost (Rs. lakh)"] = chart["total_cost"] / 1e5
        chart = chart.set_index("threshold")[["Total cost (Rs. lakh)"]]
        st.line_chart(chart, height=280, color=S1)
        st.caption(
            f"The minimum sits at t* = {t_star:.3f}. Left of it you buy recall you cannot "
            f"afford; right of it you leave returns on the table."
        )


# ======================================================== 4. governance & audit trail
# The screen that answers "prove it". Everything here reads the append-only ledger --
# nothing on this tab re-scores anything, because the whole point is to show what was
# decided at the time, not what the current model would decide now.
if page_no == "04":
    ledger: DecisionLedger = decider.ledger
    stats = ledger.stats()

    st.subheader("The guardrails in force")
    st.caption(
        "The model recommends; this layer decides. Every rule is declared in "
        "`config.yaml`, and every rule that fires is written into the ledger with a "
        "sentence a merchant can read."
    )

    pol = dict(cfg.get("policy") or {})
    shield = dict(pol.get("loyalty_shield") or {})
    rules_table = pd.DataFrame(
        [
            {
                "Guardrail": "min_order_value",
                "Effect": "suppress",
                "Bound": f"{symbol}{float(pol.get('min_order_value_inr', 0)):,.0f}",
                "Fired": stats["guardrails_fired"].get("min_order_value", 0),
            },
            {
                "Guardrail": "loyalty_shield",
                "Effect": "suppress",
                "Bound": (
                    f">={shield.get('min_prior_orders')} orders @ "
                    f"<={float(shield.get('max_prior_return_rate', 1)):.0%}"
                ),
                "Fired": stats["guardrails_fired"].get("loyalty_shield", 0),
            },
            {
                "Guardrail": "new_customer_cap",
                "Effect": "downgrade",
                "Bound": str(pol.get("new_customer_max_action")),
                "Fired": stats["guardrails_fired"].get("new_customer_cap", 0),
            },
            {
                "Guardrail": "human_review",
                "Effect": "escalate",
                "Bound": f">={symbol}{float(pol.get('human_review_above_inr', 0)):,.0f}",
                "Fired": stats["guardrails_fired"].get("human_review", 0),
            },
            {
                "Guardrail": "daily_action_cap",
                "Effect": "escalate",
                "Bound": f"{float(pol.get('max_daily_action_rate', 1)):.0%} of the book",
                "Fired": stats["guardrails_fired"].get("daily_action_cap", 0),
            },
            {
                "Guardrail": "selective_labels_holdout",
                "Effect": "holdout",
                "Bound": f"{float(pol.get('holdout_frac', 0)):.0%} of flagged",
                "Fired": stats["guardrails_fired"].get("selective_labels_holdout", 0),
            },
        ]
    )
    st.dataframe(rules_table, hide_index=True, width="stretch")

    st.divider()
    st.subheader("The ledger")

    chain = stats["chain"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Decisions recorded", f"{stats['n_decisions']:,}")
    a2.metric("Outcomes known", f"{stats['n_outcomes_recorded']:,}")
    a3.metric("Actioned today", f"{stats['action_rate_today']:.1%}")
    a4.metric("Hash chain", "intact" if chain["ok"] else f"BROKEN @ {chain['broken_at']}")

    if chain["ok"]:
        st.success(
            f"Chain verified across {chain['n_entries']} entries. Every record carries "
            f"the SHA-256 of the one before it, so editing any past decision breaks "
            f"every hash after it and this check names the row.",
            icon="🔒",
        )
    else:
        st.error(f"Ledger integrity failure at entry {chain['broken_at']}: {chain['reason']}")

    if stats["n_decisions"] == 0:
        st.info(
            "No decisions recorded yet. Score an order on tab 1 and press "
            "**Commit this decision to the audit ledger**, or POST to `/decide`.",
            icon="ℹ️",
        )
    else:
        rows = ledger.tail(200)
        outcomes = ledger.outcomes()
        frame = pd.DataFrame(
            [
                {
                    "decision_id": r["decision_id"][:12],
                    "recorded_at": str(r.get("recorded_at", ""))[:19].replace("T", " "),
                    "value": format_inr(float(r.get("order_value_inr", 0)), symbol),
                    "score": f"{float(r.get('risk_score', 0)):.1%}",
                    "model said": r.get("model_action", "-"),
                    "system did": r.get("final_action", "-"),
                    "outcome": r.get("outcome", "-"),
                    "guardrails": ", ".join(
                        h.get("rule", "") for h in (r.get("rules_fired") or [])
                    )
                    or "-",
                    "observed": (
                        "-"
                        if r["decision_id"] not in outcomes
                        else ("returned" if outcomes[r["decision_id"]]["returned"] else "kept")
                    ),
                }
                for r in rows
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch", height=300)

        overridden = sum(
            1 for r in rows if r.get("model_action") not in (r.get("final_action"), "none")
        )
        st.caption(
            f"{overridden} of the last {len(rows)} decisions were changed by a guardrail. "
            f"That number is the honest cost of governance: some of those suppressions "
            f"were returns we chose not to prevent."
        )

        st.divider()
        st.subheader("Review queue and dispute pack")
        queue = [r for r in rows if r.get("requires_human_review")]
        st.caption(
            f"{len(queue)} decision(s) the system refused to make on its own - above the "
            f"{symbol}{float(pol.get('human_review_above_inr', 0)):,.0f} limit, or at the "
            f"daily action cap."
        )

        chosen = st.selectbox(
            "Inspect a decision",
            [r["decision_id"] for r in rows],
            format_func=lambda d: (
                f"{d[:12]}  -  "
                f"{next(r.get('outcome', '') for r in rows if r['decision_id'] == d)}"
            ),
        )
        row = ledger.get(chosen)
        if row is not None:
            left_c, right_c = st.columns([1, 1])
            with left_c:
                st.markdown("**Recorded decision**")
                st.json(
                    {
                        k: row[k]
                        for k in (
                            "decision_id",
                            "seq",
                            "recorded_at",
                            "risk_score",
                            "threshold",
                            "model_action",
                            "final_action",
                            "outcome",
                            "model_version",
                            "config_fingerprint",
                            "entry_hash",
                        )
                        if k in row
                    },
                    expanded=False,
                )
                obs = outcomes.get(chosen)
                c_yes, c_no = st.columns(2)
                if c_yes.button("Mark returned", key="ret_yes"):
                    ledger.record_outcome(chosen, True, "marked in dashboard")
                    st.rerun()
                if c_no.button("Mark kept", key="ret_no"):
                    ledger.record_outcome(chosen, False, "marked in dashboard")
                    st.rerun()
                if obs:
                    st.caption(
                        f"Observed outcome: **{'returned' if obs['returned'] else 'kept'}** "
                        f"(recorded {str(obs['recorded_at'])[:19]})"
                    )

            with right_c:
                st.markdown("**Dispute evidence pack**")
                st.caption(
                    "The evidence table is assembled from the ledger and is identical "
                    "with or without an LLM. Only the narrative paragraph is generated."
                )
                if st.button("Build evidence pack", key="pack"):
                    pack = responder.dispute_pack(row, outcomes.get(chosen))
                    st.write(pack["narrative"]["text"])
                    st.caption(
                        f"Narrative generated by: {pack['narrative']['generated_by']}"
                    )
                    st.json(pack["evidence"], expanded=False)


# ========================================================== 5. run on your own data
# Upload -> name -> map columns -> validate -> train. The mapping step is deliberately
# not skippable: a column silently mapped to the wrong field produces a model that
# trains happily, scores confidently, and means nothing.
#
# Each upload becomes its own registered dataset with its own reports/models/audit
# directories, so runs never overwrite one another and deleting one cannot touch
# another. The committed UCI baseline is registered read-only and cannot be deleted.
if page_no == "05":
    add_tab, manage_tab = st.tabs(["Upload and train", f"Manage datasets ({len(datasets)})"])

    # ------------------------------------------------------------------ upload
    with add_tab:
        st.subheader("Run the whole pipeline on your own data")
        st.caption(
            "Upload a **line-level** transaction export — one row per product line, several "
            "rows per order. CSV, TSV, Excel, Parquet, JSON/JSONL, SQLite (.db/.sqlite) or a "
            "zip of any of those. Columns are matched automatically; you confirm the match "
            "before anything is trained."
        )

        with st.expander("What the pipeline needs, and what it does with it"):
            st.markdown(
                "\n".join(
                    f"- **{f.name}**{'' if f.required else '  _(optional)_'} — {f.description}"
                    for f in CANONICAL_FIELDS
                )
            )
            st.caption(
                "Each dataset gets its own `reports_user/<name>/`, `models_user/<name>/` and "
                "`audit_user/<name>/`, so the committed UCI baseline in `reports/` is never "
                "touched and two of your own runs never collide."
            )

        uploaded = st.file_uploader(
            "Transaction export",
            type=[s.lstrip(".") for s in sorted(READABLE_SUFFIXES)],
            key="byod_upload",
        )

        if uploaded is not None:
            # Staged outside the registry: nothing is registered until the user commits
            # by pressing Train, so an abandoned upload leaves no entry behind.
            staging = REPO_ROOT / "data" / "uploads" / "_staging"
            staging.mkdir(parents=True, exist_ok=True)
            saved = staging / uploaded.name
            payload = uploaded.getbuffer()
            saved.write_bytes(payload)

            tables = list_tables(saved)
            chosen_table = None
            if tables:
                chosen_table = st.selectbox(
                    "This file holds several tables/sheets — which one has the transactions?",
                    tables,
                )

            try:
                raw = read_any(saved, table=chosen_table)
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                st.stop()

            st.success(
                f"Read **{len(raw):,} rows** x {raw.shape[1]} columns from `{uploaded.name}`"
            )
            st.dataframe(raw.head(6), width="stretch", height=230)

            # ---- name it ---------------------------------------------------------
            st.markdown("#### Name this dataset")
            default_name = Path(uploaded.name).stem.replace("_", " ").strip()[:48] or "My data"
            dataset_label = st.text_input(
                "Shown in the dataset picker at the top of every tab",
                value=default_name,
                key="byod_name",
            )

            # ---- mapping ---------------------------------------------------------
            suggested, sniff_issues = suggest_mapping(raw)
            st.markdown("#### Column mapping")
            st.caption("Auto-detected. Change anything that is wrong before running.")

            options = ["(none)"] + [str(c) for c in raw.columns]
            mapping = ColumnMapping(return_mode=suggested.return_mode)
            cols = st.columns(2)
            for i, fspec in enumerate(CANONICAL_FIELDS):
                guess = suggested.columns.get(fspec.name)
                idx = options.index(str(guess)) if guess in list(raw.columns) else 0
                label = f"{fspec.name}{'' if fspec.required else '  (optional)'}"
                picked = cols[i % 2].selectbox(
                    label, options, index=idx, key=f"map_{fspec.name}", help=fspec.description
                )
                mapping.columns[fspec.name] = None if picked == "(none)" else picked

            st.markdown("#### How are returns recorded?")
            rc1, rc2, rc3 = st.columns(3)
            mode_labels = {
                "flag": "A returned / status column",
                "cancellation": "Reversing rows (negative quantity)",
            }
            mode = rc1.radio(
                "Return convention",
                list(mode_labels),
                index=list(mode_labels).index(suggested.return_mode),
                format_func=lambda k: mode_labels[k],
            )
            mapping.return_mode = mode
            mapping.assumed_return_lag_days = rc2.number_input(
                "Assumed days until a return is known",
                1.0, 365.0, float(suggested.assumed_return_lag_days), step=1.0,
                help="Used only in flag mode when there is no return-date column. This decides "
                     "when a past return enters a customer's history, so it is a real "
                     "modelling assumption.",
                disabled=(mode != "flag" or bool(mapping.columns.get("return_date"))),
            )
            mapping.currency_to_inr = rc3.number_input(
                "Multiply amounts by this to get rupees",
                0.0001, 1000.0, 1.0, step=0.5,
                help="Leave at 1.0 if your prices are already in rupees.",
            )

            # ---- validation ------------------------------------------------------
            issues = [i for i in sniff_issues if i.level != "error"] + validate(raw, mapping)
            errors = [i for i in issues if i.level == "error"]
            warns = [i for i in issues if i.level == "warning"]
            infos = [i for i in issues if i.level == "info"]

            st.markdown("#### Checks")
            for issue in errors:
                st.error(f"**{issue.message}**" + (f"\n\n{issue.fix}" if issue.fix else ""))
            for issue in warns:
                st.warning(issue.message + (f"  \n_{issue.fix}_" if issue.fix else ""))
            for issue in infos:
                st.info(issue.message)
            if not issues:
                st.success("No problems found.")

            info = profile(raw, mapping)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Rows", f"{info['n_rows']:,}")
            p2.metric("Orders", f"{info.get('n_orders', 0):,}")
            p3.metric("Customers", f"{info.get('n_customers', 0):,}")
            p4.metric("Days covered", f"{info.get('span_days', 0):,}")

            # ---- run -------------------------------------------------------------
            st.markdown("#### Train")
            if errors:
                st.caption("Fix the errors above to enable training.")
            st.button(
                "Train a new dataset on this file",
                type="primary",
                disabled=bool(errors or not dataset_label.strip()),
                key="byod_run",
            )

            if st.session_state.get("byod_run") and not errors and dataset_label.strip():
                new_ds = registry.create(
                    dataset_label,
                    upload_bytes=bytes(payload),
                    source_name=uploaded.name,
                    table=chosen_table,
                    mapping=mapping.to_dict(),
                    profile_info=info,
                )
                user_cfg = with_file_currency(training_config_for(new_ds, load_config()))

                log_box = st.empty()
                lines: list[str] = []

                def _capture(msg: str) -> None:
                    lines.append(str(msg))
                    log_box.code("\n".join(lines[-18:]), language="text")

                ok = False
                with st.spinner(f"Training **{new_ds.label}** — this takes about a minute..."):
                    try:
                        with contextlib.redirect_stdout(_StreamlitLog(_capture)):
                            run_pipeline(user_cfg, quick=True)
                        ok = True
                    except Exception as exc:
                        st.error(f"The pipeline stopped: {type(exc).__name__}: {exc}")
                        st.caption(
                            "The most common cause is a mis-mapped column — check the date "
                            "and quantity mappings above."
                        )
                        # A dataset with no model is unusable and would clutter the picker.
                        registry.delete(new_ds.key)
                        st.caption("The half-finished dataset was removed.")

                if ok:
                    registry.mark_trained(new_ds.key)
                    _clear_caches()
                    st.success(
                        f"**{new_ds.label}** is trained. Pick it in the **Dataset** selector "
                        "at the top of the page and every tab — scoring, portfolio, policy "
                        "simulator, governance and reports — will run on it."
                    )
                    summ = new_ds.summary()
                    if summ:
                        m = summ.get("metrics_test", {})
                        money_block = summ.get("money", {})
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("AUC-PR", f"{m.get('auc_pr', float('nan')):.3f}")
                        k2.metric("Base rate", f"{m.get('base_rate', float('nan')):.1%}")
                        k3.metric(
                            "Lift",
                            f"{(m.get('auc_pr', 0) / max(m.get('base_rate', 1e-9), 1e-9)):.2f}x",
                        )
                        saved_inr = money_block.get("savings")
                        k4.metric(
                            "Saved vs doing nothing",
                            format_inr(float(saved_inr), symbol) if saved_inr else "-",
                        )
                    if st.button("Reload the page", key="byod_reload"):
                        st.rerun()

    # ------------------------------------------------------------------ manage
    with manage_tab:
        st.subheader("Datasets on this machine")
        st.caption(
            "Removing a dataset deletes its uploaded file, its trained model, every "
            "report it produced and its decision ledger. The committed UCI baseline "
            "cannot be removed — it is the reference every other run is read against."
        )

        for entry in datasets:
            n_charts, n_tables = entry.artifact_counts()
            summ = entry.summary()
            metrics = summ.get("metrics_test", {}) if summ else {}

            with st.container(border=True):
                head, stat, act = st.columns([2.2, 2.4, 1.0], gap="medium")

                with head:
                    st.markdown(f"**{entry.label}**")
                    if entry.is_baseline:
                        st.caption("Committed baseline · `reports/` · read-only")
                    else:
                        src = entry.source_name or "—"
                        st.caption(
                            f"`{entry.reports_dir}/`  \nFrom **{src}** · "
                            f"added {entry.created_at[:10]}"
                        )

                with stat:
                    if entry.is_trained and metrics:
                        s1, s2, s3 = st.columns(3)
                        s1.metric("AUC-PR", f"{metrics.get('auc_pr', float('nan')):.3f}")
                        s2.metric("Base rate", f"{metrics.get('base_rate', float('nan')):.1%}")
                        s3.metric("Artifacts", f"{n_charts + n_tables}")
                    elif entry.is_trained:
                        st.caption(f"Trained · {n_charts} charts, {n_tables} tables")
                    else:
                        st.warning("Registered but never trained.", icon="⚠️")
                    if entry.n_rows:
                        st.caption(
                            f"{entry.n_rows:,} rows · {entry.n_orders:,} orders · "
                            f"{entry.n_customers:,} customers"
                        )

                with act:
                    if entry.is_baseline:
                        st.button(
                            "Protected", disabled=True, key=f"del_{entry.key}",
                            help="The committed baseline cannot be deleted from the UI.",
                            width="stretch",
                        )
                    else:
                        confirm_key = f"confirm_del_{entry.key}"
                        if st.session_state.get(confirm_key):
                            st.error("Delete permanently?")
                            yes, no = st.columns(2)
                            if yes.button("Yes", key=f"yes_{entry.key}", width="stretch"):
                                try:
                                    removed = registry.delete(entry.key)
                                except (KeyError, ValueError) as exc:
                                    st.error(str(exc))
                                else:
                                    _clear_caches()
                                    st.session_state.pop(confirm_key, None)
                                    # The picker may be pointing at what just went.
                                    if st.session_state.get("active_dataset") == entry.key:
                                        st.session_state.pop("active_dataset", None)
                                    st.session_state["last_deleted"] = (entry.label, removed)
                                    st.rerun()
                            if no.button("Cancel", key=f"no_{entry.key}", width="stretch"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                        else:
                            if st.button(
                                "Remove", key=f"del_{entry.key}", type="secondary",
                                width="stretch",
                            ):
                                st.session_state[confirm_key] = True
                                st.rerun()

        gone = st.session_state.pop("last_deleted", None)
        if gone:
            label, removed = gone
            st.success(f"Removed **{label}**.")
            if removed:
                st.code("\n".join(removed), language="text")


# ================================================================= 6. all the reports
# Everything the pipeline wrote for the dataset selected at the top of the page, plus
# the printable document.
if page_no == "06":
    reports_dir = dataset.path("reports")
    st.subheader(f"Every artifact the pipeline produced — {dataset.label}")
    st.caption(f"Reading `{dataset.reports_dir}/`. Switch dataset at the top of the page.")

    if not reports_dir.exists():
        st.warning(f"`{dataset.reports_dir}/` does not exist yet.")
    else:
        charts = sorted(reports_dir.glob("*.png"))
        tables = sorted(reports_dir.glob("*.csv"))
        summary_file = reports_dir / "summary.json"

        c1, c2, c3 = st.columns(3)
        c1.metric("Charts", len(charts))
        c2.metric("Tables", len(tables))
        c3.metric(
            "Total artifacts", len(charts) + len(tables) + (1 if summary_file.exists() else 0)
        )

        if summary_file.exists():
            summ = json.loads(summary_file.read_text(encoding="utf-8"))
            m = summ.get("metrics_test", {})
            money_block = summ.get("money", {})
            label_block = summ.get("label", {})
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("AUC-PR", f"{m.get('auc_pr', float('nan')):.3f}")
            h2.metric("Base rate", f"{m.get('base_rate', float('nan')):.1%}")
            h3.metric("Orders analysed", f"{label_block.get('n_orders_analysed', 0):,}")
            if money_block.get("savings"):
                h4.metric("Saved", format_inr(float(money_block["savings"]), symbol))

        # ------------------------------------------------------------------ print
        # Streamlit renders inside nested scroll containers and lazily mounts anything
        # below the fold, so Ctrl+P on this page yields a clipped screenshot with the
        # charts missing. The printable document is generated instead: one standalone
        # HTML file with every image inlined, a print stylesheet, and a Print button
        # that opens the browser's own Save-as-PDF dialog.
        with st.expander("🖨️  Print / Save as PDF", expanded=False):
            st.caption(
                "Builds a standalone report — every chart embedded, sized for A4. Open it "
                "and press **Print / Save as PDF**, or hand the file to someone who does "
                "not have this dashboard."
            )
            picked_sections = [
                s.key
                for s in SECTIONS
                if st.checkbox(s.label, value=s.default, key=f"print_{s.key}")
            ]

            if st.button("Build the printable report", key="build_print", type="primary"):
                if not summary_file.exists() and "summary" in picked_sections:
                    st.warning(
                        "No `summary.json` for this dataset, so the headline block will be "
                        "empty. The charts and tables are unaffected."
                    )
                ledger_stats = ledger_rows = chain_ok = None
                if "governance" in picked_sections:
                    try:
                        led = decider.ledger
                        ledger_stats = led.stats()
                        chain_ok = bool(led.verify().get("ok"))
                        recent = led.tail(40)
                        ledger_rows = pd.DataFrame(recent) if recent else None
                    except Exception as exc:  # a missing ledger must not kill the report
                        st.info(f"Audit trail unavailable for this dataset ({exc}).")

                with st.spinner("Rendering..."):
                    doc = build_report_html(
                        dataset_label=dataset.label,
                        reports_dir=reports_dir,
                        sections=picked_sections,
                        chart_notes=CHART_NOTES,
                        ledger_stats=ledger_stats,
                        ledger_rows=ledger_rows,
                        chain_ok=chain_ok,
                        policy=dict(cfg.get("policy") or {}),
                        money_symbol=symbol,
                    )

                stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
                fname = f"return-risk-report_{dataset.key.strip('_')}_{stamp}.html"
                st.success(
                    f"Built — {len(doc) / 1_000_000:.1f} MB, fully self-contained "
                    f"({doc.count('data:image/png')} charts embedded)."
                )
                st.download_button(
                    "⬇️  Download the report",
                    doc.encode("utf-8"),
                    file_name=fname,
                    mime="text/html",
                    type="primary",
                    key="dl_print",
                )
                st.caption(
                    "Open the downloaded file in any browser and press **Print / Save as "
                    "PDF** in the black bar at the top. That bar is hidden in the printed "
                    "output."
                )

        view = st.radio(
            "Show", ["Charts", "Tables", "Run summary (JSON)"], horizontal=True, key="reports_view"
        )

        if view == "Charts":
            if not charts:
                st.info("No charts in this folder yet.")
            for i in range(0, len(charts), 2):
                row = st.columns(2)
                for slot, path in zip(row, charts[i : i + 2]):
                    with slot:
                        st.markdown(f"**{_pretty(path.stem)}**")
                        st.image(str(path), width="stretch")
                        st.caption(CHART_NOTES.get(path.stem, ""))

        elif view == "Tables":
            if not tables:
                st.info("No tables in this folder yet.")
            for path in tables:
                with st.expander(f"{_pretty(path.stem)}   ·   {path.name}"):
                    try:
                        frame = pd.read_csv(path)
                    except Exception as exc:
                        st.error(f"Could not read {path.name}: {exc}")
                        continue
                    st.caption(f"{len(frame):,} rows x {frame.shape[1]} columns")
                    st.dataframe(frame.head(500), width="stretch", height=280)
                    st.download_button(
                        "Download CSV",
                        path.read_bytes(),
                        file_name=path.name,
                        mime="text/csv",
                        key=f"dl_{path.stem}",
                    )

        else:
            if summary_file.exists():
                st.json(json.loads(summary_file.read_text(encoding="utf-8")), expanded=False)
                st.download_button(
                    "Download summary.json",
                    summary_file.read_bytes(),
                    file_name="summary.json",
                    mime="application/json",
                )
            else:
                st.info("No summary.json in this folder.")
