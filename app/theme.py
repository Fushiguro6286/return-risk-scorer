"""The dark instrument-panel skin: design tokens, global CSS, and the composed blocks.

Streamlit gives you widgets, not a design system. Everything here exists to close that
gap in one place: the tokens below are the single source of colour and spacing, the CSS
restyles Streamlit's own DOM to match, and the builders emit the few compositions that
Streamlit has no widget for -- the verdict banner, the risk gauge, the driver list, the
guardrail flow.

Why hand-built HTML rather than more widgets: these blocks are read, not operated. A
gauge showing a score against its threshold is a picture of one number; expressing it
with columns and progress bars would cost more code and read worse. Anything the user
actually *interacts* with stays a real Streamlit widget so state and reruns keep working.

Every builder returns a string and is rendered with `st.markdown(..., unsafe_allow_html
=True)`. That flag is why `_esc` is not optional: dataset labels, reason codes and
guardrail names all reach these functions from user data.
"""
from __future__ import annotations

import html
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- tokens
# Sampled from the reference design, then regularised into a scale.
BG = "#06060a"          # page
PANEL = "#101019"       # cards, sidebar
PANEL_2 = "#1a1a28"     # inset rows
PANEL_3 = "#222234"     # inputs, chips
BORDER = "#27273c"      # hairlines
BORDER_SOFT = "#1b1b2a"

TEXT = "#e8eaf2"
TEXT_DIM = "#9ba0b5"
MUTED = "#6f7285"

ACCENT = "#4d7cfe"      # primary: scores, active nav, links
ACCENT_DIM = "#346bd3"
FLAG = "#fa7f2a"        # flagged / tripped
GOOD = "#33d17a"        # safe, chain intact
CRITICAL = "#ef4444"    # blocked, chain broken

#: Safe -> critical. Paired with a label everywhere; never the only signal.
RISK_RAMP = (
    f"linear-gradient(90deg,{ACCENT} 0%,#2bb3a3 22%,{GOOD} 38%,"
    f"#e0c341 60%,{FLAG} 80%,{CRITICAL} 100%)"
)


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


# ------------------------------------------------------------------------------ CSS
def css() -> str:
    """The whole skin. Injected once per run, before anything else renders."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg:{BG}; --panel:{PANEL}; --panel-2:{PANEL_2}; --panel-3:{PANEL_3};
  --border:{BORDER}; --border-soft:{BORDER_SOFT};
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --muted:{MUTED};
  --accent:{ACCENT}; --accent-dim:{ACCENT_DIM};
  --flag:{FLAG}; --good:{GOOD}; --critical:{CRITICAL};
  --r:8px;
}}

/* ---------------------------------------------------------------- foundations */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background:var(--bg) !important;
  color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-feature-settings:'tnum' 1,'cv05' 1;   /* tabular figures everywhere */
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background:transparent !important; }}
[data-testid="stAppViewContainer"] > .main .block-container {{
  padding:1.1rem 1.6rem 3rem; max-width:1560px;
}}
#MainMenu, footer {{ visibility:hidden; }}

h1,h2,h3,h4,h5 {{ color:var(--text); font-weight:600; letter-spacing:-.01em; }}
p, span, label, li {{ color:var(--text-dim); }}
code {{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  background:var(--panel-3); color:{ACCENT}; padding:1px 6px;
  border-radius:4px; font-size:.82em;
}}
a {{ color:var(--accent); }}
hr {{ border-color:var(--border); }}

/* ------------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {{
  background:var(--panel) !important;
  border-right:1px solid var(--border);
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top:.8rem; }}

.rr-brand {{
  display:flex; align-items:center; gap:9px;
  padding:2px 4px 16px; margin-bottom:4px;
  border-bottom:1px solid var(--border);
}}
.rr-brand-mark {{
  width:26px; height:26px; border-radius:7px; flex:0 0 26px;
  background:linear-gradient(145deg,var(--accent),#7a4dfe);
  display:grid; place-items:center; font-size:13px;
  box-shadow:0 0 0 1px rgba(77,124,254,.35), 0 2px 10px rgba(77,124,254,.28);
}}
.rr-brand-name {{ font-size:14.5px; font-weight:650; color:var(--text); letter-spacing:-.01em; }}

.rr-eyebrow {{
  font-size:9.5px; font-weight:700; letter-spacing:.16em; text-transform:uppercase;
  color:var(--muted); margin:16px 4px 7px;
}}

/* Nav is a radio; the label carries the "01 Score an order" text. */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:1px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  padding:8px 10px; border-radius:7px; margin:0; cursor:pointer;
  border:1px solid transparent; transition:background .12s,border-color .12s;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ background:var(--panel-2); }}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] > label p {{
  font-size:12.5px; color:var(--muted); font-weight:500; margin:0;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  background:linear-gradient(90deg,rgba(77,124,254,.16),rgba(77,124,254,.03));
  border-color:rgba(77,124,254,.34);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
  color:var(--text); font-weight:600;
}}

/* --------------------------------------------------------------- form controls */
[data-testid="stSidebar"] .stSelectbox label, .stSelectbox label,
.stNumberInput label, .stTextInput label, .stSlider label, .stRadio > label,
.stCheckbox label, .stFileUploader label, .stSelectSlider label {{
  font-size:9.5px !important; font-weight:700 !important; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted) !important;
}}
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {{
  background:var(--panel-3) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; border-radius:6px !important; font-size:13px !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
  border-color:var(--accent) !important; box-shadow:0 0 0 2px rgba(77,124,254,.2) !important;
}}
[data-baseweb="popover"] li {{ background:var(--panel-2) !important; color:var(--text) !important; }}
[data-baseweb="popover"] li:hover {{ background:var(--panel-3) !important; }}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
  background:var(--panel-3) !important; border-color:var(--border) !important;
}}

.stSlider [data-baseweb="slider"] div[role="slider"] {{
  background:var(--accent) !important;
  box-shadow:0 0 0 3px rgba(77,124,254,.25) !important;
}}

.stButton > button {{
  background:var(--panel-3); color:var(--text-dim);
  border:1px solid var(--border); border-radius:7px;
  font-size:12.5px; font-weight:550; padding:.44rem 1rem; width:100%;
  transition:all .13s;
}}
.stButton > button:hover {{
  border-color:var(--accent); color:var(--text); background:var(--panel-2);
}}
.stButton > button[kind="primary"] {{
  background:linear-gradient(180deg,var(--accent),var(--accent-dim));
  color:#fff; border-color:transparent; font-weight:600;
}}
.stButton > button[kind="primary"]:hover {{ filter:brightness(1.12); }}
.stDownloadButton > button {{
  background:var(--panel-3); color:var(--text); border:1px solid var(--border);
  border-radius:7px; font-size:12.5px;
}}

/* -------------------------------------------------------------------- panels */
.rr-panel {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:var(--r); margin-bottom:14px; overflow:hidden;
}}
.rr-panel-head {{
  padding:9px 15px; border-bottom:1px solid var(--border);
  font-size:9.5px; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); display:flex; justify-content:space-between; align-items:center;
}}
.rr-panel-body {{ padding:15px; }}

/* Streamlit's own bordered container, dressed to match .rr-panel. */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background:var(--panel); border-color:var(--border) !important; border-radius:var(--r);
}}

.rr-topbar {{
  display:flex; justify-content:space-between; align-items:center;
  padding:0 2px 12px; margin-bottom:14px; border-bottom:1px solid var(--border);
}}
.rr-topbar h2 {{ font-size:15px; margin:0; font-weight:600; }}
.rr-topbar-right {{ display:flex; align-items:center; gap:14px; }}

/* --------------------------------------------------------------------- chips */
.rr-chip {{
  display:inline-flex; align-items:center; gap:6px;
  font-size:10px; font-weight:650; letter-spacing:.07em; text-transform:uppercase;
  padding:3px 9px; border-radius:5px; border:1px solid;
}}
.rr-chip.ok {{ color:var(--good); border-color:rgba(51,209,122,.32); background:rgba(51,209,122,.09); }}
.rr-chip.flag {{ color:var(--flag); border-color:rgba(250,127,42,.34); background:rgba(250,127,42,.10); }}
.rr-chip.bad {{ color:var(--critical); border-color:rgba(239,68,68,.34); background:rgba(239,68,68,.10); }}
.rr-chip.info {{ color:var(--accent); border-color:rgba(77,124,254,.34); background:rgba(77,124,254,.10); }}
.rr-chip.mute {{ color:var(--muted); border-color:var(--border); background:var(--panel-2); }}
.rr-dot {{ width:6px; height:6px; border-radius:50%; background:currentColor;
  box-shadow:0 0 7px currentColor; }}

/* ------------------------------------------------------------------- verdict */
.rr-verdict {{
  display:flex; justify-content:space-between; align-items:center;
  padding:11px 15px; border-bottom:1px solid var(--border);
  font-size:13.5px; font-weight:650;
}}
.rr-verdict.is-flag {{ color:var(--flag); background:rgba(250,127,42,.055); }}
.rr-verdict.is-ok {{ color:var(--good); background:rgba(51,209,122,.05); }}
.rr-verdict-id {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--muted);
  letter-spacing:.03em; font-weight:500; }}

.rr-score {{ font-size:52px; font-weight:700; line-height:1; color:var(--accent);
  letter-spacing:-.035em; }}
.rr-score small {{ font-size:22px; font-weight:600; margin-left:1px; }}

/* ---------------------------------------------------------------- risk gauge */
.rr-gauge {{ margin-top:14px; }}
.rr-gauge-track {{
  position:relative; height:7px; border-radius:4px; background:{RISK_RAMP};
}}
.rr-gauge-mark {{ position:absolute; top:-4px; width:2px; height:15px; border-radius:1px; }}
.rr-gauge-mark.t {{ background:var(--text); box-shadow:0 0 0 1px var(--panel); }}
.rr-gauge-mark.s {{ background:#fff; box-shadow:0 0 6px rgba(255,255,255,.9),0 0 0 1px var(--panel); }}
.rr-gauge-scale {{
  position:relative; height:15px; margin-top:5px;
  font-size:9px; color:var(--muted); font-weight:600; letter-spacing:.04em;
}}
.rr-gauge-scale span {{ position:absolute; transform:translateX(-50%); white-space:nowrap; }}
.rr-gauge-scale .edge {{ transform:none; }}
.rr-gauge-scale .edge.r {{ right:0; }}

/* --------------------------------------------------------------------- stats */
.rr-statrow {{ display:flex; gap:26px; flex-wrap:wrap; }}
.rr-stat-k {{
  font-size:9px; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); margin-bottom:4px;
}}
.rr-stat-v {{ font-size:19px; font-weight:650; color:var(--text); letter-spacing:-.02em; }}
.rr-stat-v.accent {{ color:var(--accent); }}
.rr-stat-v.good {{ color:var(--good); }}
.rr-stat-v.flag {{ color:var(--flag); }}
.rr-stat-sub {{ font-size:10.5px; color:var(--muted); margin-top:2px; }}

/* ------------------------------------------------------------------- drivers */
.rr-driver {{
  display:flex; gap:10px; align-items:flex-start; padding:8px 0;
  border-bottom:1px solid var(--border-soft); font-size:12.5px; color:var(--text-dim);
}}
.rr-driver:last-child {{ border-bottom:0; }}
.rr-driver-ico {{ flex:0 0 15px; color:var(--flag); font-size:12px; line-height:1.35; }}
.rr-driver-bar {{ flex:0 0 34px; height:3px; border-radius:2px; margin-top:7px;
  background:var(--panel-3); overflow:hidden; }}
.rr-driver-bar i {{ display:block; height:100%; background:var(--flag); }}

/* ------------------------------------------------------------- guardrail flow */
.rr-flow {{ display:flex; align-items:stretch; gap:0; flex-wrap:wrap; }}
.rr-flow-node {{
  flex:1 1 170px; background:var(--panel-2); border:1px solid var(--border);
  border-radius:7px; padding:10px 13px; position:relative;
}}
.rr-flow-node.trip {{ border-color:rgba(250,127,42,.42); background:rgba(250,127,42,.06); }}
.rr-flow-node.final {{ border-color:rgba(51,209,122,.34); background:rgba(51,209,122,.05); }}
.rr-flow-k {{
  font-size:8.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin-bottom:5px;
}}
.rr-flow-v {{ font-size:12.5px; font-weight:600; color:var(--text); }}
.rr-flow-v.accent {{ color:var(--accent); }}
.rr-flow-v.good {{ color:var(--good); }}
.rr-flow-note {{ font-size:10.5px; color:var(--muted); margin-top:4px; line-height:1.45; }}
.rr-flow-arrow {{ flex:0 0 26px; display:grid; place-items:center; color:var(--muted);
  font-size:13px; }}
.rr-flow-badge {{
  position:absolute; top:-8px; right:9px; font-size:8px; font-weight:750;
  letter-spacing:.1em; padding:2px 6px; border-radius:3px;
  background:var(--flag); color:#160c02;
}}
.rr-mono {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--accent); }}

/* -------------------------------------------------------------- data display */
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:var(--r); }}
[data-testid="stMetricValue"] {{ font-size:21px; font-weight:650; color:var(--text); }}
[data-testid="stMetricLabel"] p {{
  font-size:9px !important; font-weight:700; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted) !important;
}}
[data-testid="stMetricDelta"] {{ font-size:11px; }}

.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
  background:transparent; color:var(--muted); font-size:12.5px; font-weight:550;
  padding:7px 14px; border-radius:6px 6px 0 0;
}}
.stTabs [aria-selected="true"] {{ color:var(--accent) !important; background:var(--panel-2); }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent); }}

.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
  background:var(--panel-2) !important; border-radius:6px; font-size:12.5px;
  color:var(--text-dim) !important;
}}
[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:var(--r); }}

[data-testid="stAlert"] {{ border-radius:7px; border:1px solid var(--border); font-size:12.5px; }}
[data-testid="stFileUploaderDropzone"] {{
  background:var(--panel-2); border:1.5px dashed var(--border); border-radius:var(--r);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--accent); }}
.stProgress > div > div > div {{ background:var(--accent); }}
[data-testid="stImage"] img {{ border:1px solid var(--border); border-radius:6px; }}

::-webkit-scrollbar {{ width:9px; height:9px; }}
::-webkit-scrollbar-track {{ background:var(--bg); }}
::-webkit-scrollbar-thumb {{ background:var(--panel-3); border-radius:5px; }}
::-webkit-scrollbar-thumb:hover {{ background:#333349; }}
</style>
"""


# ------------------------------------------------------------------------ builders
def topbar(title: str, right: str = "") -> str:
    return (
        f'<div class="rr-topbar"><h2>{_esc(title)}</h2>'
        f'<div class="rr-topbar-right">{right}</div></div>'
    )


def chip(text: str, kind: str = "mute", dot: bool = False) -> str:
    inner = '<span class="rr-dot"></span>' if dot else ""
    return f'<span class="rr-chip {kind}">{inner}{_esc(text)}</span>'


def panel(title: str, body: str, aside: str = "") -> str:
    return (
        f'<div class="rr-panel"><div class="rr-panel-head"><span>{_esc(title)}</span>'
        f"<span>{aside}</span></div>"
        f'<div class="rr-panel-body">{body}</div></div>'
    )


def stat(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="rr-stat-sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div><div class="rr-stat-k">{_esc(label)}</div>'
        f'<div class="rr-stat-v {tone}">{_esc(value)}</div>{sub_html}</div>'
    )


def statrow(cells: Sequence[str]) -> str:
    return f'<div class="rr-statrow">{"".join(cells)}</div>'


def verdict(flagged: bool, action_label: str, ref: str = "") -> str:
    """The banner across the top of the decision panel."""
    if flagged:
        text, kind, icon = f"FLAG → {action_label}", "is-flag", "▲"
    else:
        text, kind, icon = f"NO ACTION — {action_label}", "is-ok", "●"
    ref_html = f'<span class="rr-verdict-id">{_esc(ref)}</span>' if ref else ""
    return (
        f'<div class="rr-verdict {kind}"><span>{icon}&nbsp; {_esc(text)}</span>'
        f"{ref_html}</div>"
    )


def gauge(score: float, threshold: float) -> str:
    """Score against its threshold on a safe->critical ramp.

    Both markers are clamped into the track and the numbers are printed underneath,
    because position alone is not readable at this size -- and the distance between the
    two marks is the thing being communicated, not either value on its own.
    """
    s = max(0.0, min(1.0, float(score)))
    t = max(0.0, min(1.0, float(threshold)))
    sp, tp = s * 100, t * 100
    # Nudge labels off the edges so they cannot clip out of the container.
    s_lab, t_lab = min(max(sp, 7), 93), min(max(tp, 7), 93)
    return (
        '<div class="rr-gauge"><div class="rr-gauge-track">'
        f'<div class="rr-gauge-mark t" style="left:{tp:.2f}%"></div>'
        f'<div class="rr-gauge-mark s" style="left:{sp:.2f}%"></div></div>'
        '<div class="rr-gauge-scale">'
        '<span class="edge">0%</span>'
        f'<span style="left:{t_lab:.2f}%">t* {t:.1%}</span>'
        f'<span style="left:{s_lab:.2f}%;color:#fff">Score {s:.1%}</span>'
        '<span class="edge r">100%</span>'
        "</div></div>"
    )


def drivers(reasons: Iterable[str], weights: Sequence[float] | None = None) -> str:
    """Top reason codes, strongest first, with a relative-contribution bar."""
    items = [r for r in reasons if r]
    if not items:
        return '<div class="rr-flow-note">No dominant driver for this order.</div>'
    ws = list(weights or [])
    top = max([abs(w) for w in ws], default=0) or 1.0
    out = []
    for i, reason in enumerate(items):
        frac = (abs(ws[i]) / top * 100) if i < len(ws) else 100 - i * 22
        out.append(
            '<div class="rr-driver">'
            '<span class="rr-driver-ico">◆</span>'
            f"<span style='flex:1'>{_esc(reason)}</span>"
            f'<span class="rr-driver-bar"><i style="width:{max(8, min(100, frac)):.0f}%"></i></span>'
            "</div>"
        )
    return "".join(out)


def flow(model_action: str, final_action: str, rules: Sequence[str] = (), note: str = "") -> str:
    """Model recommendation -> guardrails -> what the system is allowed to do.

    The gap between the first and last node is the argument for having a policy layer,
    so the middle node is rendered even when nothing fired -- an empty guardrail step
    reads as "checked and cleared", where omitting it reads as "not checked".
    """
    nodes = [
        '<div class="rr-flow-node">'
        '<div class="rr-flow-k">Model recommendation</div>'
        f'<div class="rr-flow-v accent">{_esc(model_action)}</div></div>'
    ]
    if rules:
        rule_html = " · ".join(f'<span class="rr-mono">{_esc(r)}</span>' for r in rules)
        nodes.append(
            '<div class="rr-flow-arrow">→</div>'
            '<div class="rr-flow-node trip"><span class="rr-flow-badge">TRIPPED</span>'
            f'<div class="rr-flow-k">Guardrail</div><div class="rr-flow-v">{rule_html}</div>'
            + (f'<div class="rr-flow-note">{_esc(note)}</div>' if note else "")
            + "</div>"
        )
    else:
        nodes.append(
            '<div class="rr-flow-arrow">→</div>'
            '<div class="rr-flow-node"><div class="rr-flow-k">Guardrails</div>'
            '<div class="rr-flow-v" style="color:var(--muted)">None tripped</div>'
            '<div class="rr-flow-note">All policy bounds cleared.</div></div>'
        )
    nodes.append(
        '<div class="rr-flow-arrow">→</div>'
        '<div class="rr-flow-node final"><div class="rr-flow-k">Final allowed action</div>'
        f'<div class="rr-flow-v good">{_esc(final_action)}</div></div>'
    )
    return f'<div class="rr-flow">{"".join(nodes)}</div>'


def brand() -> str:
    return (
        '<div class="rr-brand"><div class="rr-brand-mark">🛡</div>'
        '<div class="rr-brand-name">Return-Risk Scorer</div></div>'
    )


def eyebrow(text: str) -> str:
    return f'<div class="rr-eyebrow">{_esc(text)}</div>'


__all__ = [
    "css", "topbar", "chip", "panel", "stat", "statrow", "verdict", "gauge",
    "drivers", "flow", "brand", "eyebrow",
    "ACCENT", "FLAG", "GOOD", "CRITICAL", "TEXT", "MUTED", "PANEL", "BG", "BORDER",
]
