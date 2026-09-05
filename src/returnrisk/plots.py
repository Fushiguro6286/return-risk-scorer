"""Every chart the report emits, in one house style.

Design rules followed here (they are not decoration -- they are what makes the charts
readable to someone who is not us):

* **Sequential = one hue, light to dark**, used only for magnitude (the money matrix,
  the sensitivity heatmap). No rainbows.
* **Categorical hues in fixed slot order**, never cycled: blue, orange, aqua. The
  three-slot set is validated colour-blind-safe against this surface, and aqua sits
  below 3:1 contrast so every series that uses it also carries a direct label.
* **Never two y-axes.** Where two measures of different scale belong together (AUC-PR
  and rupees saved over time) they become stacked small multiples sharing an x-axis.
* **Recessive chrome.** Hairline grid, no top/right spines, muted tick labels; the data
  is the only thing with contrast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .money import CostBreakdown, format_inr

# --- design tokens ----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue, orange, aqua (validated triple)
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ = LinearSegmentedColormap.from_list("rr_blue", BLUE_RAMP)

FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def use_house_style() -> None:
    """Apply the shared rcParams. Called once by the pipeline."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def _clean(ax: plt.Axes, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(xgrid)
    ax.yaxis.grid(not xgrid)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --- Tier 1.1: the headline artifact ------------------------------------------------
def money_confusion_matrix(
    bd: CostBreakdown, path: Path, *, action_label: str, symbol: str = "Rs."
) -> Path:
    """The money shot: a confusion matrix whose cells are rupees, not counts.

    A count-based confusion matrix invites the wrong conversation ("look how many we
    caught"). Pricing the cells forces the right one: the bottom-left cell is what
    inaction costs, and the top-right cell is what over-flagging costs.
    """
    tp_saved = bd.cost_tp_counterfactual - bd.cost_tp
    cells = [
        [("Correctly left alone", bd.cost_tn, bd.n_tn, "no action needed, none taken"),
         ("False alarm", bd.cost_fp, bd.n_fp, "friction imposed on a good order")],
        [("Missed return", bd.cost_fn, bd.n_fn, "the return we failed to prevent"),
         ("Caught", bd.cost_tp, bd.n_tp,
          f"action + leakage, down from {format_inr(bd.cost_tp_counterfactual, symbol)} "
          f"(saves {format_inr(tp_saved, symbol)})")],
    ]
    vmax = max(bd.cost_fn, bd.cost_tp, bd.cost_fp, 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for r in range(2):
        for c in range(2):
            name, cost, n, sub = cells[r][c]
            frac = np.clip(cost / vmax, 0.0, 1.0)
            face = SEQ(0.08 + 0.82 * frac)
            txt = "#ffffff" if frac > 0.55 else INK
            ax.add_patch(
                plt.Rectangle((c, 1 - r), 1, 1, facecolor=face, edgecolor=SURFACE, linewidth=3)
            )
            ax.text(c + 0.5, 1 - r + 0.74, name, ha="center", va="center",
                    fontsize=11, color=txt, fontweight="600")
            ax.text(c + 0.5, 1 - r + 0.50, format_inr(cost, symbol), ha="center", va="center",
                    fontsize=21, color=txt, fontweight="700")
            ax.text(c + 0.5, 1 - r + 0.30, f"{n:,} orders", ha="center", va="center",
                    fontsize=10, color=txt)
            ax.text(c + 0.5, 1 - r + 0.15, sub, ha="center", va="center",
                    fontsize=8.0, color=txt, alpha=0.88, style="italic", wrap=True)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Not flagged", "Flagged"], fontsize=10.5, color=INK_2)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Did NOT return", "Actually returned"], fontsize=10.5, color=INK_2)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Model decision", fontsize=10, color=MUTED, labelpad=8)
    ax.set_ylabel("What actually happened", fontsize=10, color=MUTED, labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)

    saved_pct = bd.savings / bd.do_nothing_cost if bd.do_nothing_cost else 0.0
    fig.suptitle(
        f"Cost of returns on the held-out test set  |  {action_label}  |  t* = {bd.threshold:.3f}",
        fontsize=13, fontweight="600", color=INK, y=1.02,
    )
    fig.text(
        0.5, -0.055,
        f"Do nothing: {format_inr(bd.do_nothing_cost, symbol)}      "
        f"With this policy: {format_inr(bd.total_cost, symbol)}      "
        f"Saved: {format_inr(bd.savings, symbol)} ({saved_pct:.1%})",
        ha="center", fontsize=12, color=INK, fontweight="600",
    )
    fig.text(
        0.5, -0.10,
        f"{bd.n:,} test orders  ·  precision {bd.precision:.1%}  ·  recall {bd.recall:.1%}  ·  "
        f"flagged {bd.flag_rate:.1%}",
        ha="center", fontsize=9, color=MUTED,
    )
    return _save(fig, path)


def cost_curve(
    sweeps: dict[str, pd.DataFrame], t_stars: dict[str, float], path: Path, symbol: str = "Rs."
) -> Path:
    """Expected rupee cost against threshold, one line per action, t* marked.

    Three series, direct-labelled at their minima -- which is also the relief the
    aqua slot needs to clear the contrast rule.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = [S1, S2, S3]
    for (name, sweep), color in zip(sweeps.items(), colors, strict=False):
        ax.plot(sweep["threshold"], sweep["total_cost"] / 1e5, color=color, label=name, zorder=3)
        t = t_stars[name]
        row = sweep.iloc[(sweep["threshold"] - t).abs().idxmin()]
        y = row["total_cost"] / 1e5
        ax.scatter([t], [y], s=64, color=color, zorder=5,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            f"{name}\nt* = {t:.3f}",
            xy=(t, y), xytext=(10, 14), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="600",
        )

    if sweeps:
        first = next(iter(sweeps.values()))
        do_nothing = float(first["do_nothing_cost"].iloc[0]) / 1e5
        ax.axhline(do_nothing, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
        ax.annotate("do nothing", xy=(0.985, do_nothing), xytext=(0, 5),
                    textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)

    ax.set_xlabel("Risk-score threshold")
    ax.set_ylabel(f"Total expected cost ({symbol} lakh)")
    ax.set_title("Each intervention has its own cost-minimising threshold")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    _clean(ax)
    return _save(fig, path)


def per_action_thresholds(table: pd.DataFrame, path: Path, symbol: str = "Rs.") -> Path:
    """The monotonicity result: as the false-positive cost rises, so does t*."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = [a.replace("_", " ") for a in table["action"]]
    x = np.arange(len(table))

    ax1.plot(table["fp_cost_mean_inr"], table["t_star"], color=S1, marker="o",
             markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax1.set_xscale("log")
    # Labels sit above their point, centred, with room reserved on both axes so the
    # first and last annotations are not clipped by the log-scaled frame.
    for _, r in table.iterrows():
        ax1.annotate(
            f"{r['action'].replace('_', ' ')}\nt*={r['t_star']:.3f}",
            xy=(r["fp_cost_mean_inr"], r["t_star"]), xytext=(0, 14),
            textcoords="offset points", fontsize=8.5, color=INK_2,
            ha="center", va="bottom",
        )
    xs = table["fp_cost_mean_inr"].to_numpy(dtype="float64")
    ys = table["t_star"].to_numpy(dtype="float64")
    ax1.set_xlim(xs.min() / 3.0, xs.max() * 3.0)
    span = max(ys.max() - ys.min(), 1e-6)
    ax1.set_ylim(ys.min() - 0.18 * span, ys.max() + 0.42 * span)
    ax1.set_xlabel(f"Mean false-positive cost per order ({symbol}, log scale)")
    ax1.set_ylabel("Optimal threshold t*")
    ax1.set_title("Costlier mistakes buy a higher bar")
    _clean(ax1)

    ax2.bar(x, table["orders_flagged"], color=S1, width=0.6, zorder=3)
    for i, (n, rate) in enumerate(zip(table["orders_flagged"], table["flag_rate"], strict=True)):
        ax2.text(i, n, f"{n:,}\n({rate:.0%})", ha="center", va="bottom",
                 fontsize=9, color=INK_2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Orders flagged")
    ax2.set_title("...and so flags fewer orders")
    ax2.margins(y=0.18)
    _clean(ax2)
    return _save(fig, path)


def baseline_comparison(table: pd.DataFrame, path: Path, symbol: str = "Rs.") -> Path:
    """Total rupee cost of every policy, model highlighted."""
    df = table.sort_values("total_cost_inr", ascending=True)
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    colors = [S1 if p.startswith("Model") else MUTED for p in df["policy"]]
    y = np.arange(len(df))
    ax.barh(y, df["total_cost_inr"] / 1e5, color=colors, height=0.62, zorder=3)
    for i, (v, s) in enumerate(zip(df["total_cost_inr"], df["savings_vs_do_nothing_inr"], strict=True)):
        ax.text(v / 1e5, i, f"  {format_inr(v, symbol)}   (saves {format_inr(s, symbol)})",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([p if len(p) < 52 else p[:49] + "..." for p in df["policy"]], fontsize=9)
    ax.set_xlabel(f"Total cost on the test set ({symbol} lakh) - lower is better")
    ax.set_title("The model against the alternatives, priced in rupees")
    ax.margins(x=0.30)
    _clean(ax, xgrid=True)
    return _save(fig, path)


# --- Tier 0.4: the honest metric panel ---------------------------------------------
def pr_curve(pr: pd.DataFrame, base_rate: float, auc_pr: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(pr["recall"], pr["precision"], color=S1, zorder=3)
    ax.axhline(base_rate, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
    ax.annotate(f"base rate {base_rate:.1%}\n(a coin-flip model)", xy=(0.55, base_rate),
                xytext=(0, 8), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall  ·  AUC-PR = {auc_pr:.3f}  ({auc_pr / base_rate:.2f}x base)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1.0, float(pr["precision"].max()) * 1.05))
    _clean(ax)
    return _save(fig, path)


def calibration_curve(frame: pd.DataFrame, path: Path, ece: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    lim = float(max(frame["mean_predicted"].max(), frame["observed_rate"].max())) * 1.12
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4,
            label="perfect calibration")
    ax.plot(frame["mean_predicted"], frame["observed_rate"], color=S1, marker="o",
            markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3,
            label="calibrated model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed return rate")
    ax.set_title(f"Reliability (quantile bins)  ·  ECE = {ece:.4f}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    _clean(ax)
    return _save(fig, path)


def calibration_before_after(
    uncal: pd.DataFrame, cal: pd.DataFrame, path: Path, brier: dict[str, float]
) -> Path:
    """Two panels so the effect of isotonic regression is visible, not asserted."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for ax, frame, name, key in (
        (axes[0], uncal, "Raw LightGBM", "brier_uncalibrated_calib_slice"),
        (axes[1], cal, "After isotonic calibration", "brier_calibrated_calib_slice"),
    ):
        lim = 0.75
        ax.plot([0, lim], [0, lim], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
        ax.plot(frame["mean_predicted"], frame["observed_rate"], color=S1, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        ax.set_title(f"{name}\nBrier {brier[key]:.4f}", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        _clean(ax)
    axes[0].set_ylabel("Observed return rate")
    return _save(fig, path)


def lift_chart(lift: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(lift))
    ax.bar(x, lift["return_rate"], color=S1, width=0.66, zorder=3)
    base = float((lift["returns"].sum() / lift["n"].sum()))
    ax.axhline(base, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
    ax.annotate(f"base rate {base:.1%}", xy=(len(lift) - 1, base), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i + 1}" for i in x])
    ax.set_xlabel("Risk decile (D1 = riskiest)")
    ax.set_ylabel("Observed return rate")
    ax.set_title("Does the ranking work? Return rate by risk decile")
    _clean(ax)
    return _save(fig, path)


# --- Tier 1.4 -----------------------------------------------------------------------
def sensitivity_heatmap(pivot: pd.DataFrame, path: Path, title: str, fmt: str = "{:.2f}") -> Path:
    """t* (or savings) across the FN x FP cost grid. Sequential ramp: magnitude only."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    data = pivot.to_numpy(dtype="float64")
    im = ax.imshow(data, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}x" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{r:g}x" for r in pivot.index])
    ax.set_xlabel("False-positive cost multiplier")
    ax.set_ylabel("False-negative cost multiplier")
    ax.set_title(title)
    lo, hi = np.nanmin(data), np.nanmax(data)
    rng = (hi - lo) or 1.0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = data[i, j]
            frac = (v - lo) / rng
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if frac > 0.55 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _save(fig, path)


# --- Tier 2.1 -----------------------------------------------------------------------
def stability_plot(backtest: pd.DataFrame, path: Path, symbol: str = "Rs.") -> Path:
    """Stacked small multiples -- never a second y-axis."""
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.2), sharex=True)
    x = np.arange(len(backtest))
    labels = [p[:7] for p in backtest["period"]]

    axes[0].plot(x, backtest["auc_pr"], color=S1, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3, label="AUC-PR")
    axes[0].plot(x, backtest["base_rate"], color=MUTED, linestyle=(0, (4, 3)),
                 linewidth=1.4, label="base rate (AUC-PR floor)")
    axes[0].set_ylabel("AUC-PR")
    axes[0].set_title("Does performance hold up month by month on the held-out period?")
    axes[0].legend(loc="best")
    axes[0].set_ylim(0, max(0.6, float(backtest["auc_pr"].max()) * 1.25))

    axes[1].plot(x, backtest["precision"], color=S2, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3, label="precision")
    axes[1].plot(x, backtest["recall"], color=S3, marker="s", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3, label="recall")
    axes[1].set_ylabel("at fixed t*")
    axes[1].legend(loc="best")

    axes[2].bar(x, backtest["savings_inr"] / 1e5, color=S1, width=0.62, zorder=3)
    axes[2].set_ylabel(f"Saved ({symbol} lakh)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0)
    axes[2].set_xlabel("Test-period month")

    for ax in axes:
        _clean(ax)

    # The first and last slices are partial months; without this note the short bars at
    # either end read as decay rather than as fewer days.
    n = backtest["n_orders"]
    if len(n) > 2 and (n.iloc[0] < 0.6 * n.iloc[1:-1].mean() or n.iloc[-1] < 0.6 * n.iloc[1:-1].mean()):
        fig.text(
            0.5, 0.055,
            f"First and last slices are partial months ({int(n.iloc[0]):,} and "
            f"{int(n.iloc[-1]):,} orders vs ~{int(n.iloc[1:-1].mean()):,} mid-window) - "
            "their lower rupee totals reflect fewer days, not decay.",
            ha="center", fontsize=8.5, color=MUTED, style="italic",
        )
    return _save(fig, path)


# --- Tier 2.3 / 2.4 ------------------------------------------------------------------
def segment_plot(table: pd.DataFrame, path: Path, weakest: str) -> Path:
    """AUC-PR lift over each segment's own base rate; 1.0x is worthless."""
    df = table[table["reliable"]].copy()
    df["name"] = df["dimension"] + " = " + df["segment"]
    df = df.sort_values("auc_pr_lift")
    fig, ax = plt.subplots(figsize=(9.0, max(4.2, 0.34 * len(df))))
    y = np.arange(len(df))
    colors = [CRITICAL if n == weakest else S1 for n in df["name"]]
    ax.barh(y, df["auc_pr_lift"], color=colors, height=0.66, zorder=3)
    ax.axvline(1.0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
    ax.annotate("1.0x = no better than\nthe base rate", xy=(1.0, len(df) - 0.5),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=MUTED,
                va="top")
    for i, (v, n) in enumerate(zip(df["auc_pr_lift"], df["n_orders"], strict=True)):
        ax.text(v, i, f"  {v:.2f}x  (n={n:,})", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=9)
    ax.set_xlabel("AUC-PR as a multiple of that segment's own base rate")
    ax.set_title("Where the model works, and where it does not (worst segment in red)")
    ax.margins(x=0.22)
    _clean(ax, xgrid=True)
    return _save(fig, path)


def fairness_plot(fair: pd.DataFrame, path: Path) -> Path:
    """Flag rate against the segment's own return rate: is friction proportionate?"""
    df = fair[fair["reliable"]].copy()
    df["name"] = df["dimension"].str.replace("_", " ") + " = " + df["segment"]
    # One shared x-axis: the categories are identical, so labelling both panels would
    # duplicate nine rotated labels and collide with the lower title.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 8.6), sharex=True, gridspec_kw={"hspace": 0.28}
    )

    x = np.arange(len(df))
    w = 0.38
    ax1.bar(x - w / 2, df["base_rate"], width=w - 0.02, color=MUTED, label="actual return rate",
            zorder=3)
    ax1.bar(x + w / 2, df["flag_rate"], width=w - 0.02, color=S1, label="flag rate", zorder=3)
    ax1.set_ylabel("Share of orders")
    ax1.set_title("Is the friction we impose proportionate to the risk we observe?", pad=26)
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    ax1.margins(y=0.16)
    ax1.tick_params(labelbottom=False)
    _clean(ax1)

    colors = [CRITICAL if v < df["precision"].median() else S1 for v in df["precision"]]
    ax2.bar(x, df["precision"], width=0.6, color=colors, zorder=3)
    for i, v in enumerate(df["precision"]):
        ax2.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["name"], rotation=30, ha="right", fontsize=8.5)
    ax2.set_ylabel("Precision when flagged")
    ax2.set_title("Precision by segment - a low bar here means friction paid for nothing")
    ax2.margins(y=0.20)
    _clean(ax2)
    return _save(fig, path)


# --- Tier 2.5 -------------------------------------------------------------------------
def precision_at_k_plot(topk: pd.DataFrame, base_rate: float, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(topk))
    labels = [f"top {k:.0%}" for k in topk["k_pct"]]

    ax1.bar(x, topk["precision_at_k"], color=S1, width=0.6, zorder=3)
    ax1.axhline(base_rate, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
    ax1.annotate(f"base rate {base_rate:.1%}", xy=(len(topk) - 0.4, base_rate),
                 xytext=(0, 6), textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    for i, (v, l) in enumerate(zip(topk["precision_at_k"], topk["lift_vs_base"], strict=True)):
        ax1.text(i, v, f"{v:.0%}\n({l:.2f}x)", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Precision@k")
    ax1.set_title("If you can only review k% of orders")
    ax1.margins(y=0.22)
    _clean(ax1)

    ax2.bar(x, topk["recall_at_k"], color=S2, width=0.6, zorder=3)
    for i, v in enumerate(topk["recall_at_k"]):
        ax2.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Recall@k")
    ax2.set_title("...this is the share of all returns you touch")
    ax2.margins(y=0.20)
    _clean(ax2)
    return _save(fig, path)


# --- Tier 2.2 -------------------------------------------------------------------------
def selective_labels_plot(result: Any, path: Path) -> Path:
    """Base-rate collapse and the AUC-PR consequence, side by side."""
    comp = result.comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    names = ["true\n(uncensored)", "naive retrain\n(censored)", f"with {result.holdout_frac:.0%}\nholdout + IPW"]
    vals = [result.true_base_rate_p1, result.observed_base_rate_naive, result.observed_base_rate_holdout]
    colors = [MUTED, CRITICAL, GOOD]
    ax1.bar(np.arange(3), vals, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.2%}", ha="center", va="bottom", fontsize=9.5, color=INK_2)
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Return rate the next model would see")
    ax1.set_title("Acting on scores hides the outcomes")
    ax1.margins(y=0.22)
    _clean(ax1)

    short = ["original", "retrained on\ncensored labels", f"retrained w/\n{result.holdout_frac:.0%} holdout"]
    bar_colors = [MUTED, CRITICAL, GOOD]
    ax2.bar(np.arange(len(comp)), comp["auc_pr_period2"], color=bar_colors[: len(comp)],
            width=0.6, zorder=3)
    for i, v in enumerate(comp["auc_pr_period2"]):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK_2)
    ax2.set_xticks(np.arange(len(comp)))
    ax2.set_xticklabels(short[: len(comp)], fontsize=9)
    ax2.set_ylabel("AUC-PR on clean period-2 data")
    ax2.set_title("...and the next model is worse for it")
    ax2.margins(y=0.22)
    _clean(ax2)
    return _save(fig, path)


# --- Tier 4 ---------------------------------------------------------------------------
def shap_bar(global_imp: pd.DataFrame, path: Path, top_n: int = 18) -> Path:
    df = global_imp.head(top_n).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.4, max(4.4, 0.32 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["mean_abs_shap"], color=S1, height=0.66, zorder=3)
    for i, (v, s) in enumerate(zip(df["mean_abs_shap"], df["share"], strict=True)):
        ax.text(v, i, f"  {s:.1%}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Mean |SHAP| (log-odds contribution)")
    ax.set_title("What actually drives the score - all of it known at checkout")
    ax.margins(x=0.16)
    _clean(ax, xgrid=True)
    return _save(fig, path)


def shap_beeswarm(explainer: Any, df: pd.DataFrame, path: Path, max_rows: int = 2500) -> Path:
    """SHAP's own summary plot -- the standard global explainability artifact."""
    import shap

    sample = df.sample(min(len(df), max_rows), random_state=0) if len(df) > max_rows else df
    X = explainer._matrix(sample)
    vals = explainer.shap_values(sample)
    fig = plt.figure(figsize=(8.6, 6.4))
    shap.summary_plot(vals, X, show=False, plot_size=None, max_display=18)
    plt.title("SHAP summary - per-order signed contributions", fontsize=12, color=INK)
    plt.tight_layout()
    return _save(fig, path)


__all__ = [
    "baseline_comparison",
    "calibration_before_after",
    "calibration_curve",
    "cost_curve",
    "fairness_plot",
    "lift_chart",
    "money_confusion_matrix",
    "per_action_thresholds",
    "pr_curve",
    "precision_at_k_plot",
    "segment_plot",
    "selective_labels_plot",
    "sensitivity_heatmap",
    "shap_bar",
    "shap_beeswarm",
    "stability_plot",
    "use_house_style",
]
