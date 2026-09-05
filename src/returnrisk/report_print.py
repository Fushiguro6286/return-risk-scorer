"""Builds a self-contained, printable HTML report for one dataset's run.

Why a standalone file rather than print CSS on the dashboard: Streamlit renders inside
nested scroll containers and lazily mounts anything below the fold, so `Ctrl+P` on the
live page reliably produces a clipped first screen with the charts missing. A generated
document sidesteps all of that -- every image is inlined as a data URI, so the file
prints identically whether it is opened from disk, emailed, or archived, and it still
renders years from now when `reports/` has moved on.

The document is the run's evidence, in the order an outsider needs it: what was run and
on what data, what it scored, what it is worth in money, then the charts, the tables,
and the audit trail underneath. Nothing here recomputes anything -- it reads the
artifacts the pipeline already wrote, so the printed numbers cannot drift from the
dashboard's.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

#: Charts in the order they tell the story, not the order the filesystem lists them.
CHART_ORDER: tuple[str, ...] = (
    "money_confusion_matrix",
    "cost_vs_threshold",
    "per_action_thresholds",
    "baseline_comparison",
    "pr_curve",
    "roc_curve",
    "calibration_curve",
    "calibration_before_after",
    "lift_by_decile",
    "precision_at_k",
    "cost_sensitivity_heatmap",
    "cost_sensitivity_savings",
    "segment_metrics",
    "fairness_by_segment",
    "stability",
    "selective_labels",
    "shap_global_importance",
    "shap_summary",
)

#: Tables worth putting on paper. The rest stay as CSVs -- a 7,500-row scored book is
#: not a printed artifact, and padding the document with it buries the ones that matter.
TABLE_ALLOW: tuple[str, ...] = (
    "baseline_comparison",
    "per_action_thresholds",
    "lift_table",
    "precision_at_k",
    "fairness_by_segment",
    "segment_metrics",
    "cost_sensitivity",
    "calibration_curve",
    "reason_code_examples",
)

MAX_TABLE_ROWS = 40


@dataclass(frozen=True)
class PrintSection:
    """One toggleable block of the document."""

    key: str
    label: str
    default: bool = True


SECTIONS: tuple[PrintSection, ...] = (
    PrintSection("summary", "Headline metrics and money"),
    PrintSection("charts", "Charts"),
    PrintSection("tables", "Tables"),
    PrintSection("governance", "Governance and audit trail"),
    PrintSection("methodology", "Method, label definition and limits"),
)


# --------------------------------------------------------------------------- helpers
def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _img_data_uri(path: Path) -> str | None:
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def _pretty(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    try:
        if spec:
            return format(float(value), spec)
    except (TypeError, ValueError):
        return _esc(value)
    return _esc(value)


def _ordered_charts(reports_dir: Path) -> list[Path]:
    """Known charts in narrative order, then anything else alphabetically."""
    found = {p.stem: p for p in sorted(reports_dir.glob("*.png"))}
    ordered = [found.pop(stem) for stem in CHART_ORDER if stem in found]
    return ordered + sorted(found.values())


def _table_to_html(path: Path, max_rows: int = MAX_TABLE_ROWS) -> str:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return f'<p class="muted">Could not read {_esc(path.name)}: {_esc(exc)}</p>'

    total = len(frame)
    shown = frame.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False, name=None)
    )
    note = (
        f'<p class="muted">Showing {max_rows} of {total:,} rows. '
        f"Full table in <code>{_esc(path.name)}</code>.</p>"
        if total > max_rows
        else ""
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"


def _kv_grid(pairs: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="stat-k">{_esc(k)}</div>'
        f'<div class="stat-v">{_esc(v)}</div></div>'
        for k, v in pairs
        if v not in ("", None)
    )
    return f'<div class="stats">{cells}</div>' if cells else ""


# ----------------------------------------------------------------------------- CSS
_CSS = """
:root{--ink:#0b0b0b;--mute:#5b6470;--line:#d9dee5;--bg:#fff;--accent:#2a78d6;
--good:#0ca30c;--bad:#d03b3b;--panel:#f6f8fa;}
*{box-sizing:border-box;}
body{margin:0;background:#eef1f5;color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.sheet{max-width:960px;margin:24px auto;background:var(--bg);padding:44px 52px;
box-shadow:0 1px 4px rgba(16,24,40,.12);border-radius:6px;}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em;}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--ink);}
h3{font-size:14px;margin:20px 0 6px;}
p{margin:8px 0;}
.sub{color:var(--mute);margin:0 0 18px;font-size:13px;}
.muted{color:var(--mute);font-size:12px;}
code{background:var(--panel);padding:1px 5px;border-radius:3px;font-size:12px;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 18px;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:10px 12px;}
.stat-k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);}
.stat-v{font-size:19px;font-weight:600;margin-top:3px;}
figure{margin:0 0 22px;break-inside:avoid;page-break-inside:avoid;}
figure img{width:100%;border:1px solid var(--line);border-radius:4px;}
figcaption{font-size:12px;color:var(--mute);margin-top:5px;}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin:6px 0 4px;}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;}
th{background:var(--panel);font-weight:600;}
tbody tr:nth-child(even){background:#fbfcfd;}
.wrap{overflow-x:auto;}
.block{break-inside:avoid;page-break-inside:avoid;margin-bottom:20px;}
.bar{position:sticky;top:0;background:var(--ink);color:#fff;padding:10px 18px;
display:flex;gap:12px;align-items:center;justify-content:space-between;z-index:9;}
.bar button{background:#fff;color:var(--ink);border:0;border-radius:4px;padding:7px 16px;
font-size:13px;font-weight:600;cursor:pointer;}
.bar button:hover{background:#e8edf3;}
.chain-ok{color:var(--good);font-weight:600;}
.chain-bad{color:var(--bad);font-weight:600;}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:9px 13px;
margin:12px 0;font-size:12.5px;}
@page{size:A4;margin:14mm;}
@media print{
  body{background:#fff;}
  .sheet{max-width:none;margin:0;padding:0;box-shadow:none;border-radius:0;}
  .bar{display:none !important;}
  h2{page-break-after:avoid;}
  figure img{border-color:#bbb;}
  a{text-decoration:none;color:inherit;}
}
"""

_TOOLBAR = """
<div class="bar">
  <span><strong>Return-Risk report</strong> &mdash; use Print to save as PDF</span>
  <button onclick="window.print()">Print / Save as PDF</button>
</div>
"""


# ---------------------------------------------------------------------- the builder
def build_report_html(
    *,
    dataset_label: str,
    reports_dir: Path,
    sections: Iterable[str] = ("summary", "charts", "tables", "governance", "methodology"),
    chart_notes: dict[str, str] | None = None,
    ledger_stats: dict[str, Any] | None = None,
    ledger_rows: pd.DataFrame | None = None,
    chain_ok: bool | None = None,
    policy: dict[str, Any] | None = None,
    money_symbol: str = "Rs.",
    generated_at: datetime | None = None,
) -> str:
    """Render the whole document as one HTML string with every image inlined."""
    wanted = set(sections)
    notes = chart_notes or {}
    stamp = (generated_at or datetime.now()).strftime("%d %b %Y, %H:%M")
    reports_dir = Path(reports_dir)

    summary: dict[str, Any] = {}
    summary_file = reports_dir / "summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}

    metrics = summary.get("metrics_test", {}) or {}
    money = summary.get("money", {}) or {}
    label = summary.get("label", {}) or {}
    split = summary.get("split", {}) or {}

    parts: list[str] = [
        "<h1>Return-Risk Scorer &mdash; run report</h1>",
        f'<p class="sub">Dataset <strong>{_esc(dataset_label)}</strong> &nbsp;·&nbsp; '
        f"artifacts from <code>{_esc(reports_dir.name)}/</code> &nbsp;·&nbsp; "
        f"generated {_esc(stamp)}</p>",
    ]

    if not summary:
        parts.append(
            '<div class="note">No <code>summary.json</code> in this folder, so the '
            "headline metrics are unavailable. Train this dataset first.</div>"
        )

    # -------------------------------------------------------------- 1. the numbers
    if "summary" in wanted and summary:
        base_rate = metrics.get("base_rate")
        auc_pr = metrics.get("auc_pr")
        lift = (auc_pr / base_rate) if (auc_pr and base_rate) else None
        parts.append("<h2>Headline</h2>")
        parts.append(
            _kv_grid(
                [
                    ("AUC-PR", _fmt(auc_pr, ".3f")),
                    ("Base rate", _fmt(base_rate, ".2%")),
                    ("Lift over base", f"{lift:.2f}x" if lift else "-"),
                    ("ROC-AUC", _fmt(metrics.get("roc_auc"), ".3f")),
                    ("Brier", _fmt(metrics.get("brier"), ".4f")),
                    ("Precision", _fmt(metrics.get("precision"), ".3f")),
                    ("Recall", _fmt(metrics.get("recall"), ".3f")),
                    ("Orders analysed", f"{int(label.get('n_orders_analysed', 0)):,}"),
                ]
            )
        )
        if money:
            do_nothing = float(money.get("do_nothing_cost", 0) or 0)
            savings = float(money.get("savings", 0) or 0)
            parts.append("<h3>What it is worth</h3>")
            parts.append(
                _kv_grid(
                    [
                        ("Action priced", str(money.get("action", "-"))),
                        ("Threshold t*", _fmt(money.get("threshold"), ".3f")),
                        ("Do nothing", f"{money_symbol}{do_nothing:,.0f}"),
                        ("Under policy", f"{money_symbol}{float(money.get('total_cost', 0) or 0):,.0f}"),
                        ("Saved", f"{money_symbol}{savings:,.0f}"),
                        ("Saved (%)", f"{savings / do_nothing:.1%}" if do_nothing else "-"),
                        ("Flagged", f"{int(money.get('n_flagged', 0)):,}"),
                        ("Flag rate", _fmt(money.get("flag_rate"), ".1%")),
                    ]
                )
            )
        if split:
            def _slice(name: str) -> str:
                block = split.get(name) or {}
                return f"{int(block.get('n', 0)):,}" if block else "?"

            parts.append(
                f'<p class="muted">Temporal split &mdash; train {_slice("train")}, '
                f'calibrate {_slice("calib")}, test {_slice("test")}; '
                f'embargo {split.get("embargo_days", "?")} days '
                f'({int(split.get("n_embargoed", 0)):,} orders dropped whose return '
                "window was still open).</p>"
            )

    # ---------------------------------------------------------------- 2. the charts
    if "charts" in wanted:
        charts = _ordered_charts(reports_dir)
        if charts:
            parts.append("<h2>Charts</h2>")
            for path in charts:
                uri = _img_data_uri(path)
                if uri is None:
                    continue
                caption = notes.get(path.stem, "")
                parts.append(
                    "<figure>"
                    f"<h3>{_esc(_pretty(path.stem))}</h3>"
                    f'<img src="{uri}" alt="{_esc(_pretty(path.stem))}">'
                    + (f"<figcaption>{_esc(caption)}</figcaption>" if caption else "")
                    + "</figure>"
                )

    # ---------------------------------------------------------------- 3. the tables
    if "tables" in wanted:
        tables = [p for p in sorted(reports_dir.glob("*.csv")) if p.stem in TABLE_ALLOW]
        if tables:
            parts.append("<h2>Tables</h2>")
            for path in tables:
                parts.append(
                    f'<div class="block"><h3>{_esc(_pretty(path.stem))}</h3>'
                    f'<div class="wrap">{_table_to_html(path)}</div></div>'
                )

    # ------------------------------------------------------------ 4. the governance
    if "governance" in wanted and (ledger_stats or policy is not None):
        parts.append("<h2>Governance and audit trail</h2>")
        if policy:
            shield = dict(policy.get("loyalty_shield") or {})
            rows = [
                ("min_order_value", "suppress",
                 f"{money_symbol}{float(policy.get('min_order_value_inr', 0)):,.0f}"),
                ("loyalty_shield", "suppress",
                 f">={shield.get('min_prior_orders', '?')} orders at "
                 f"<={float(shield.get('max_prior_return_rate', 1)):.0%} return rate"),
                ("new_customer_cap", "downgrade",
                 str(policy.get("new_customer_max_action", "-"))),
            ]
            fired = (ledger_stats or {}).get("guardrails_fired", {}) or {}
            body = "".join(
                f"<tr><td>{_esc(n)}</td><td>{_esc(e)}</td><td>{_esc(b)}</td>"
                f"<td>{int(fired.get(n, 0)):,}</td></tr>"
                for n, e, b in rows
            )
            parts.append(
                "<h3>Guardrails in force</h3><table><thead><tr><th>Guardrail</th>"
                "<th>Effect</th><th>Bound</th><th>Times fired</th></tr></thead>"
                f"<tbody>{body}</tbody></table>"
            )

        if ledger_stats:
            chain = (
                '<span class="chain-ok">intact</span>' if chain_ok
                else '<span class="chain-bad">BROKEN</span>' if chain_ok is False
                else "not checked"
            )
            by_action = ledger_stats.get("by_final_action", {}) or {}
            n_decisions = int(ledger_stats.get("n_decisions", 0))
            n_none = int(by_action.get("none", 0))
            parts.append(
                _kv_grid(
                    [
                        ("Decisions recorded", f"{n_decisions:,}"),
                        ("Actioned", f"{n_decisions - n_none:,}"),
                        ("Outcomes known",
                         f"{int(ledger_stats.get('n_outcomes_recorded', 0)):,}"),
                        ("Expected saving",
                         f"{money_symbol}"
                         f"{float(ledger_stats.get('expected_saving_inr_total', 0) or 0):,.0f}"),
                    ]
                )
            )
            parts.append(f"<p>Hash chain: {chain}.</p>")
            if by_action:
                rows = "".join(
                    f"<tr><td>{_esc(k)}</td><td>{int(v):,}</td></tr>"
                    for k, v in sorted(by_action.items(), key=lambda kv: -int(kv[1]))
                )
                parts.append(
                    "<h3>Decisions by final action</h3><table><thead><tr>"
                    f"<th>Action</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>"
                )

        if ledger_rows is not None and not ledger_rows.empty:
            keep = [
                c for c in ("seq", "recorded_at", "risk_score", "final_action",
                            "outcome", "entry_hash")
                if c in ledger_rows.columns
            ]
            view = ledger_rows[keep].tail(MAX_TABLE_ROWS) if keep else ledger_rows.tail(MAX_TABLE_ROWS)
            head = "".join(f"<th>{_esc(c)}</th>" for c in view.columns)
            body = "".join(
                "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
                for row in view.itertuples(index=False, name=None)
            )
            parts.append(
                f"<h3>Ledger &mdash; most recent {len(view)} entries</h3>"
                f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>"
                '<p class="muted">Append-only and hash-chained: each row commits to its '
                "predecessor, so an edited or removed row breaks verification at a "
                "known sequence number. This is tamper evidence, not tamper proofing.</p>"
            )

    # ------------------------------------------------------------- 5. the method
    if "methodology" in wanted:
        parts.append("<h2>Method and limits</h2>")
        definition = label.get("label_definition") or label.get("definition")
        if definition:
            parts.append(f"<p><strong>Label.</strong> {_esc(definition)}</p>")
        audit = summary.get("leakage_audit") or {}
        if audit.get("verdict"):
            parts.append(f"<p><strong>Leakage audit.</strong> {_esc(audit['verdict'])}</p>")
        parts.append(
            '<div class="note">Every figure here is measured on a held-out slice that is '
            "strictly later than the training data, with an embargo covering the return "
            "window, so no order contributes to both training and evaluation. Costs are "
            "assumptions declared in <code>config.yaml</code>, not measurements &mdash; the "
            "sensitivity sweep exists because those assumptions are the weakest link.</div>"
        )
        parts.append(
            '<p class="muted">Generated by the Return-Risk Scorer dashboard. Figures are '
            "read from the artifacts the pipeline wrote; this document performs no "
            "computation of its own.</p>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Return-Risk report &mdash; {_esc(dataset_label)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{_TOOLBAR}<div class='sheet'>{''.join(parts)}</div></body></html>"
    )


__all__ = ["build_report_html", "SECTIONS", "PrintSection", "CHART_ORDER", "TABLE_ALLOW"]
