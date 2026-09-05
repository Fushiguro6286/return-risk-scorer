"""The honest metric panel.

Headline is **AUC-PR**, always printed next to the base rate, because on an 18% positive
rate a model can post 0.82 accuracy by predicting "no return" for everything and a
respectable ROC-AUC while being useless at the top of the ranking. AUC-PR against its
own base-rate floor is the number that says whether the ranking is worth acting on.

ROC-AUC is reported too, but as a secondary diagnostic, never as the headline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class MetricPanel:
    """Every headline number for one (slice, threshold) pair."""

    n: int
    n_positive: int
    base_rate: float
    auc_pr: float
    auc_pr_lift_vs_base: float
    roc_auc: float
    brier: float
    log_loss: float
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_flagged: int
    flag_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"AUC-PR {self.auc_pr:.4f} (base rate {self.base_rate:.2%}, "
            f"lift {self.auc_pr_lift_vs_base:.2f}x) | ROC-AUC {self.roc_auc:.4f} | "
            f"Brier {self.brier:.4f} | @t={self.threshold:.3f}: "
            f"P {self.precision:.3f} R {self.recall:.3f} F1 {self.f1:.3f} "
            f"(flagged {self.n_flagged:,} / {self.n:,})"
        )


def evaluate(y: np.ndarray | pd.Series, proba: np.ndarray | pd.Series, threshold: float) -> MetricPanel:
    """Compute the full panel. `threshold` should be the cost-optimal t*, not 0.5."""
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype="float64")
    pred = (p >= threshold).astype(int)
    base = float(y.mean()) if len(y) else 0.0
    ap = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")

    return MetricPanel(
        n=int(len(y)),
        n_positive=int(y.sum()),
        base_rate=base,
        auc_pr=ap,
        auc_pr_lift_vs_base=float(ap / base) if base > 0 else float("nan"),
        roc_auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        threshold=float(threshold),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        accuracy=float((pred == y).mean()),
        n_flagged=int(pred.sum()),
        flag_rate=float(pred.mean()),
    )


def pr_curve_frame(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
        }
    )


def roc_curve_frame(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y, proba)
    return pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})


def calibration_frame(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table.

    Quantile bins rather than uniform: with a low base rate the top uniform bins are
    empty, which makes a badly calibrated model look fine.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype="float64")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(p.min(), p.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
                "lower_edge": float(edges[b]),
                "upper_edge": float(edges[b + 1]),
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out["observed_rate"] - out["mean_predicted"]
    return out


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |observed - predicted| across bins. Lower is better; 0 is perfect."""
    frame = calibration_frame(y, proba, n_bins)
    if frame.empty:
        return float("nan")
    w = frame["n"] / frame["n"].sum()
    return float((w * frame["gap"].abs()).sum())


def lift_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift -- the plainest way to show a merchant the ranking works."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False, ignore_index=True)
    df["decile"] = np.minimum((np.arange(len(df)) * n_bins) // max(len(df), 1), n_bins - 1)
    base = df["y"].mean()
    out = df.groupby("decile").agg(n=("y", "size"), returns=("y", "sum"), mean_score=("p", "mean"))
    out["return_rate"] = out["returns"] / out["n"]
    out["lift"] = out["return_rate"] / base if base else np.nan
    return out.reset_index()


__all__ = [
    "MetricPanel",
    "calibration_frame",
    "evaluate",
    "expected_calibration_error",
    "lift_table",
    "pr_curve_frame",
    "roc_curve_frame",
]
