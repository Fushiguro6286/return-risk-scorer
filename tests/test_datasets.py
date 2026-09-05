"""The dataset registry: isolation between runs, and that deleting one is total.

The property that matters here is not that the happy path works -- it is that a user
dataset cannot reach the committed baseline. Every test in the isolation group exists
because the alternative is a merchant's upload silently overwriting the reference run
the whole README is written against.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from returnrisk.config import Config, load_config
from returnrisk.datasets import (
    BASELINE_KEY,
    Dataset,
    DatasetRegistry,
    baseline_dataset,
    config_for,
    training_config_for,
)


@pytest.fixture(autouse=True)
def sandbox_repo_root(tmp_path, monkeypatch):
    """Point the module's REPO_ROOT at a temp tree.

    `Dataset.path()` resolves every directory against the repo root, so without this a
    test that creates or deletes a dataset would write into the real `reports_user/`.
    """
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("returnrisk.datasets.REPO_ROOT", root)
    return root


@pytest.fixture
def registry(tmp_path: Path) -> DatasetRegistry:
    """A registry backed by a temp file, so tests never touch the real one."""
    return DatasetRegistry(tmp_path / "registry.json")


# --------------------------------------------------------------------------- paths
def test_baseline_owns_the_committed_directories():
    base = baseline_dataset()
    assert base.reports_dir == "reports"
    assert base.models_dir == "models"
    assert base.audit_dir == "audit"
    assert base.is_baseline


def test_user_dataset_never_shares_a_directory_with_the_baseline(registry):
    ds = registry.create("My shop")
    base = baseline_dataset()
    for which in ("reports_dir", "models_dir", "audit_dir"):
        assert getattr(ds, which) != getattr(base, which)
        assert getattr(ds, which).endswith(ds.key)


def test_two_uploads_get_separate_directories(registry):
    a = registry.create("Same name")
    b = registry.create("Same name")
    assert a.key != b.key
    assert a.reports_dir != b.reports_dir
    assert a.audit_dir != b.audit_dir


def test_labels_are_slugified_into_safe_keys(registry):
    ds = registry.create("Q3 export // 2024 (final!)")
    assert "/" not in ds.key and "\\" not in ds.key
    assert ".." not in ds.key


# ------------------------------------------------------------------------- config
def test_config_for_redirects_every_path(registry):
    base = load_config()
    ds = registry.create("Redirect me")
    cfg = config_for(ds, base)
    assert cfg["paths"]["reports_dir"] == ds.reports_dir
    assert cfg["paths"]["models_dir"] == ds.models_dir
    assert cfg["paths"]["audit_dir"] == ds.audit_dir
    assert cfg["paths"]["model_file"] == ds.model_file
    # The original is untouched -- Config is meant to be immutable.
    assert base["paths"]["reports_dir"] == "reports"


def test_config_for_leaves_the_baseline_alone():
    base = load_config()
    cfg = config_for(baseline_dataset(), base)
    assert cfg["paths"]["reports_dir"] == "reports"
    assert cfg["money"]["gbp_to_inr"] == base["money"]["gbp_to_inr"]


def test_user_dataset_uses_its_own_fx_rate(registry):
    """A rupee-priced file must not be inflated by the UCI pound rate.

    At 105x the cost of a missed return dwarfs a false alarm, the optimal threshold
    collapses to zero, and the policy 'recommends' actioning every order.
    """
    base = load_config()
    assert base["money"]["gbp_to_inr"] > 1  # the baseline really is in pounds

    ds = registry.create("Rupee shop", mapping={"currency_to_inr": 1.0})
    assert config_for(ds, base)["money"]["gbp_to_inr"] == 1.0

    other = registry.create("Dollar shop", mapping={"currency_to_inr": 83.0})
    assert config_for(other, base)["money"]["gbp_to_inr"] == 83.0


def test_training_config_points_the_pipeline_at_the_upload(registry, tmp_path):
    base = load_config()
    ds = registry.create(
        "With a file",
        upload_bytes=b"a,b\n1,2\n",
        source_name="orders.csv",
        mapping={"columns": {"order_id": "a"}, "currency_to_inr": 1.0},
    )
    cfg = training_config_for(ds, base)
    assert cfg["data"]["source"] == "file"
    assert Path(cfg["data"]["file"]["path"]).exists()
    assert cfg["data"]["file"]["mapping"]["columns"]["order_id"] == "a"


def test_training_config_refuses_the_baseline():
    with pytest.raises(ValueError, match="run_demo"):
        training_config_for(baseline_dataset(), load_config())


def test_training_config_refuses_a_dataset_with_no_upload(registry):
    ds = registry.create("No file attached")
    with pytest.raises(ValueError, match="no stored upload"):
        training_config_for(ds, load_config())


# ------------------------------------------------------------------------ registry
def test_create_persists_across_instances(registry, tmp_path):
    registry.create("Durable")
    reloaded = DatasetRegistry(tmp_path / "registry.json")
    assert [d.label for d in reloaded.user_datasets()] == ["Durable"]


def test_upload_is_stored_under_the_dataset_key(registry):
    ds = registry.create("Has upload", upload_bytes=b"hello", source_name="x.csv")
    assert ds.upload_path is not None
    assert ds.upload_path.read_bytes() == b"hello"
    assert ds.key in ds.upload_path.parts


def test_unknown_key_falls_back_to_baseline(registry):
    """A stale browser selection must not raise mid-render."""
    assert registry.get("does-not-exist").is_baseline
    assert registry.get(None).is_baseline
    assert registry.get("").is_baseline


def test_baseline_is_always_listed_first(registry):
    registry.create("One")
    registry.create("Two")
    assert registry.all()[0].is_baseline
    assert len(registry.all()) == 3


def test_is_trained_reads_disk_not_a_flag(registry):
    ds = registry.create("Untrained")
    assert not ds.is_trained  # registered, but no model file was ever written


def test_rename_keeps_the_key_and_therefore_the_directories(registry):
    ds = registry.create("Before")
    registry.rename(ds.key, "After")
    again = registry.get(ds.key)
    assert again.label == "After"
    assert again.reports_dir == ds.reports_dir


# ------------------------------------------------------------------------- delete
def test_delete_removes_every_directory_the_dataset_owned(registry):
    ds = registry.create("Doomed", upload_bytes=b"x", source_name="d.csv")
    for which in ("reports", "models", "audit"):
        ds.path(which).mkdir(parents=True, exist_ok=True)
        (ds.path(which) / "artifact.txt").write_text("payload", encoding="utf-8")

    removed = registry.delete(ds.key)

    assert len(removed) == 4  # reports, models, audit, upload
    for which in ("reports", "models", "audit"):
        assert not ds.path(which).exists()
    assert ds.upload_path is not None and not ds.upload_path.exists()
    assert not registry.exists(ds.key)


def test_delete_refuses_the_baseline(registry):
    with pytest.raises(ValueError, match="cannot be deleted"):
        registry.delete(BASELINE_KEY)


def test_delete_of_unknown_key_raises(registry):
    with pytest.raises(KeyError):
        registry.delete("never-existed")


def test_deleting_one_dataset_leaves_the_others_standing(registry):
    keep = registry.create("Keep me")
    drop = registry.create("Drop me")
    keep.path("reports").mkdir(parents=True, exist_ok=True)
    (keep.path("reports") / "keep.txt").write_text("still here", encoding="utf-8")

    registry.delete(drop.key)

    assert registry.exists(keep.key)
    assert (keep.path("reports") / "keep.txt").exists()


def test_delete_never_walks_outside_the_repo(registry, tmp_path, sandbox_repo_root):
    """A registry naming an absolute path elsewhere must not have it removed."""
    outsider = tmp_path / "not-in-the-repo"
    outsider.mkdir()
    (outsider / "precious.txt").write_text("do not delete", encoding="utf-8")

    ds = registry.create("Escapee")
    # Forge a source_path that resolves outside the repo root.
    registry._entries[ds.key].source_path = str(outsider / "precious.txt")
    registry.delete(ds.key)

    assert (outsider / "precious.txt").exists()


def test_registry_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{ not json at all", encoding="utf-8")
    assert DatasetRegistry(path).user_datasets() == []


def test_registry_ignores_a_persisted_baseline_row(tmp_path):
    """The baseline is synthesised, never read from disk, so it cannot be shadowed."""
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {"key": BASELINE_KEY, "label": "Impostor", "kind": "baseline"},
                    {"key": "real", "label": "Real one", "kind": "user"},
                ],
            }
        ),
        encoding="utf-8",
    )
    reg = DatasetRegistry(path)
    assert [d.key for d in reg.user_datasets()] == ["real"]
    assert reg.all()[0].label == baseline_dataset().label


def test_round_trips_through_dict():
    ds = Dataset(key="k", label="L", n_rows=5, mapping={"currency_to_inr": 2.0})
    assert Dataset.from_dict(ds.to_dict()) == ds


def test_from_dict_tolerates_unknown_keys():
    """A registry written by a newer build must not crash an older one."""
    ds = Dataset.from_dict({"key": "k", "label": "L", "some_future_field": 1})
    assert ds.key == "k"
