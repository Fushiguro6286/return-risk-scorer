"""LightGBM + isotonic calibration, and the artefact that gets served.

Calibration is not decoration here. The whole money layer multiplies a probability by a
rupee cost, so a score of 0.30 has to mean "30 of every 100 such orders come back". A
raw boosted-tree score does not; isotonic regression fitted on a held-out slice makes it
so, and the Brier score before/after is reported as evidence.

Deliberately absent: SMOTE, class weights, resampling of any kind. They shift the
predicted base rate away from the true one, which is exactly the property the cost
curve depends on. An 18% base rate is not extreme enough to need them anyway.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from .config import Config
from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, FeatureBuilder, select_features


def _freeze(estimator: Any) -> Any:
    """Wrap a fitted estimator so CalibratedClassifierCV will not refit it.

    scikit-learn replaced ``cv="prefit"`` with ``FrozenEstimator`` (1.6+, removed 1.8).
    Support both so the repo is not pinned to one minor version.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return FrozenEstimator(estimator)
    except ImportError:  # pragma: no cover - older scikit-learn
        return estimator


def build_estimator(cfg: Config):
    """Construct the (unfitted) LightGBM classifier from config params."""
    from lightgbm import LGBMClassifier

    params = dict(cfg["model"]["params"])
    params.setdefault("random_state", int(cfg["seed"]))
    params.setdefault("n_jobs", -1)
    return LGBMClassifier(**params)


@dataclass
class TrainedModel:
    """Everything the API and dashboard need, and nothing they do not."""

    calibrated: Any
    raw: Any
    feature_builder: FeatureBuilder
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(return) for a feature frame."""
        return self.calibrated.predict_proba(X[self.feature_columns])[:, 1]

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.raw.predict_proba(X[self.feature_columns])[:, 1]

    def save(self, path: str | Path) -> Path:
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=3)
        return p

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        import joblib

        return joblib.load(Path(path))


def _as_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the allow-listed columns and keep categoricals as pandas dtype.

    LightGBM consumes `category` dtype natively -- no one-hot, and crucially no encoder
    fitted on data outside the training slice.
    """
    X = select_features(df).copy()
    for c in CATEGORICAL_COLUMNS:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def train_and_calibrate(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    builder: FeatureBuilder,
    cfg: Config,
) -> tuple[TrainedModel, dict[str, float]]:
    """Fit on train, calibrate on the held-out calibration slice.

    Returns the artefact and the before/after Brier scores that justify calibrating.
    """
    X_tr, y_tr = _as_model_matrix(train), train["returned"].to_numpy()
    X_ca, y_ca = _as_model_matrix(calib), calib["returned"].to_numpy()

    raw = build_estimator(cfg)
    raw.fit(X_tr, y_tr, categorical_feature=CATEGORICAL_COLUMNS)

    method = str(cfg["model"]["calibration"])
    # Modern scikit-learn: FrozenEstimator + default CV over the calibration slice.
    # Legacy scikit-learn: cv="prefit". Either way `raw` is never refitted.
    kwargs: dict[str, Any] = {"method": method}
    if _legacy():
        kwargs["cv"] = "prefit"
    calibrated = CalibratedClassifierCV(_freeze(raw), **kwargs)
    calibrated.fit(X_ca, y_ca)

    brier = {
        "brier_uncalibrated_calib_slice": float(
            brier_score_loss(y_ca, raw.predict_proba(X_ca)[:, 1])
        ),
        "brier_calibrated_calib_slice": float(
            brier_score_loss(y_ca, calibrated.predict_proba(X_ca)[:, 1])
        ),
    }

    model = TrainedModel(
        calibrated=calibrated,
        raw=raw,
        feature_builder=builder,
        feature_columns=list(FEATURE_COLUMNS),
        metadata={
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "calibration": method,
            "n_train": int(len(train)),
            "n_calib": int(len(calib)),
            "train_base_rate": float(train["returned"].mean()),
            "train_start": str(train["order_date"].min()),
            "train_end": str(train["order_date"].max()),
            "model_params": dict(cfg["model"]["params"]),
            "seed": int(cfg["seed"]),
        },
    )
    return model, brier


def _legacy() -> bool:
    """True when this scikit-learn still wants ``cv='prefit'`` instead of FrozenEstimator."""
    try:
        import sklearn.frozen  # noqa: F401

        return False
    except ImportError:  # pragma: no cover
        return True


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    """Gain-based importance of the underlying booster, for the leakage audit."""
    booster = model.raw
    gains = booster.booster_.feature_importance(importance_type="gain")
    names = booster.booster_.feature_name()
    df = pd.DataFrame({"feature": names, "gain": gains})
    df["gain_pct"] = df["gain"] / df["gain"].sum() if df["gain"].sum() else 0.0
    return df.sort_values("gain", ascending=False, ignore_index=True)


__all__ = [
    "TrainedModel",
    "build_estimator",
    "feature_importance",
    "train_and_calibrate",
]
