"""Smoke test (Tier 6.3): the whole pipeline runs and writes every promised artifact.

Runs on the synthetic dataset so CI needs no 45MB download, and asserts on the shape of
the output rather than on particular numbers -- the numbers belong to the real data.
"""
from __future__ import annotations

import json

import pytest

from returnrisk.pipeline import run

#: Every file the README, the model card and the dashboard rely on existing.
REQUIRED_ARTIFACTS = [
    "summary.json",
    "money_confusion_matrix.png",
    "cost_vs_threshold.png",
    "per_action_thresholds.png",
    "per_action_thresholds.csv",
    "baseline_comparison.png",
    "baseline_comparison.csv",
    "cost_sensitivity.csv",
    "cost_sensitivity_heatmap.png",
    "pr_curve.png",
    "calibration_curve.png",
    "calibration_before_after.png",
    "lift_by_decile.png",
    "stability.png",
    "stability_backtest.csv",
    "segment_metrics.png",
    "segment_metrics.csv",
    "fairness_by_segment.png",
    "fairness_by_segment.csv",
    "precision_at_k.png",
    "precision_at_k.csv",
    "selective_labels.png",
    "selective_labels_comparison.csv",
    "shap_global_importance.png",
    "shap_global_importance.csv",
    "gain_importance.csv",
]


@pytest.fixture(scope="module")
def smoke_run(cfg, tmp_path_factory):
    """One synthetic end-to-end run, shared by every assertion in this module."""
    out = tmp_path_factory.mktemp("reports")
    models = tmp_path_factory.mktemp("models")
    patched = cfg.as_dict()
    patched["data"] = {**patched["data"], "source": "synthetic"}
    patched["paths"] = {
        "reports_dir": str(out),
        "models_dir": str(models),
        "model_file": str(models / "model.joblib"),
    }
    return run(type(cfg)(_data=patched, path=cfg.path), quick=True), out


def test_pipeline_completes(smoke_run):
    art, _ = smoke_run
    assert art.summary["runtime_seconds"] > 0
    assert len(art.files) >= len(REQUIRED_ARTIFACTS)


@pytest.mark.parametrize("name", REQUIRED_ARTIFACTS)
def test_required_artifact_written(smoke_run, name):
    _, out = smoke_run
    path = out / name
    assert path.exists(), f"{name} was not written"
    assert path.stat().st_size > 0, f"{name} is empty"


def test_summary_json_is_valid_and_complete(smoke_run):
    _, out = smoke_run
    with open(out / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)

    for section in (
        "label", "data", "split", "calibration", "metrics_test", "money",
        "per_action_thresholds", "baselines", "sensitivity", "stability",
        "segments", "fairness", "capacity", "selective_labels", "leakage_audit",
    ):
        assert section in summary, f"summary.json is missing '{section}'"


def test_headline_metrics_are_present_and_sane(smoke_run):
    art, _ = smoke_run
    m = art.summary["metrics_test"]
    assert 0.0 <= m["auc_pr"] <= 1.0
    assert 0.0 <= m["base_rate"] <= 1.0
    # AUC-PR must at least be reported against its floor.
    assert m["auc_pr"] >= m["base_rate"] * 0.5
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] > 0


def test_calibration_improves_the_brier_score(smoke_run):
    """Acceptance criterion 0.3 -- calibration must actually help on the calib slice."""
    art, _ = smoke_run
    c = art.summary["calibration"]
    assert c["brier_calibrated_calib_slice"] <= c["brier_uncalibrated_calib_slice"] + 1e-9


def test_per_action_thresholds_are_monotone(smoke_run):
    """Acceptance criterion 1.2 -- the headline risk-maturity result."""
    art, _ = smoke_run
    assert art.summary["per_action_thresholds"]["monotone_cheaper_action_lower_threshold"]


def test_model_beats_do_nothing(smoke_run):
    art, _ = smoke_run
    assert art.summary["baselines"]["headline"]["model_beats_do_nothing"]


def test_leakage_audit_ran(smoke_run):
    art, _ = smoke_run
    audit = art.summary["leakage_audit"]
    assert "verdict" in audit and audit["verdict"]
    assert isinstance(audit["flags"], list)


def test_selective_labels_experiment_shows_censoring(smoke_run):
    """The censored base rate must fall below the true one, or the sim is broken."""
    art, _ = smoke_run
    sl = art.summary["selective_labels"]
    assert sl["n_labels_censored"] > 0
    assert sl["observed_base_rate_naive_retrain"] < sl["true_base_rate_period1"]


def test_saved_model_round_trips(smoke_run):
    """The served artifact must load and score without the training code path."""
    from returnrisk.model import TrainedModel

    art, _ = smoke_run
    path = next(p for p in art.files if str(p).endswith(".joblib"))
    model = TrainedModel.load(path)
    assert model.thresholds
    assert set(model.feature_columns)
    assert "label_definition" in model.metadata
