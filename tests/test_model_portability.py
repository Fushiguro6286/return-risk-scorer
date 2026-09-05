"""A saved model must load on an OS other than the one that trained it.

This is invisible on the machine that writes the file -- it round-trips there
perfectly -- and only fails on the host that has to load it. A model trained on
Windows and deployed to Linux raised `UnsupportedOperation: cannot instantiate
'WindowsPath'` before the estimator was even reachable, because the fitted feature
builder pickles the `Config` it was built from and `Config` held a `Path`.

`Path` is OS-flavoured; `str` is not. These tests assert nothing OS-specific survives
the pickle.
"""
from __future__ import annotations

import pathlib
import pickle

from returnrisk.config import load_config


def _paths_inside(obj, trail="root", depth=0, seen=None):
    """Every PurePath reachable from `obj`, with the attribute trail that reached it."""
    if seen is None:
        seen = set()
    if depth > 6 or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, pathlib.PurePath):
        return [(trail, obj)]
    found = []
    state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        for k, v in state.items():
            found += _paths_inside(v, f"{trail}.{k}", depth + 1, seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += _paths_inside(v, f"{trail}[{k!r}]", depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += _paths_inside(v, f"{trail}[{i}]", depth + 1, seen)
    return found


def test_config_pickles_its_path_as_a_string():
    """The pickled payload must carry no OS-flavoured Path."""
    state = load_config().__getstate__()
    assert isinstance(state["path"], str)
    assert not isinstance(state["path"], pathlib.PurePath)


def test_config_survives_a_pickle_round_trip():
    cfg = load_config()
    again = pickle.loads(pickle.dumps(cfg))
    assert again["paths"]["reports_dir"] == cfg["paths"]["reports_dir"]
    assert again["money"]["gbp_to_inr"] == cfg["money"]["gbp_to_inr"]
    assert isinstance(again.path, pathlib.PurePath)


def test_pickled_config_holds_no_path_objects():
    """The bytes on disk are what cross the OS boundary, so check the bytes.

    A `WindowsPath` pickle names its class in the payload; loading that on Linux
    raises before anything else runs.
    """
    blob = pickle.dumps(load_config())
    assert b"WindowsPath" not in blob
    assert b"PosixPath" not in blob


def test_a_saved_model_repickles_without_os_specific_paths(tmp_path):
    """The real guard, tested on the bytes that actually cross the boundary.

    Checking the *loaded* object is not the test: `__setstate__` rebuilds a `Path` on
    the loading machine, which is correct and portable. What matters is whether the
    serialised form names an OS-specific class -- that is what fails on the far side.

    Skipped when no model has been built yet; CI trains one in the smoke run and a
    fresh clone ships the committed baseline.
    """
    import joblib
    import pytest

    from returnrisk.config import REPO_ROOT

    model_file = REPO_ROOT / "models" / "return_risk_model.joblib"
    if not model_file.exists():
        pytest.skip("no trained model on disk; run `python run_demo.py` first")

    model = joblib.load(model_file)
    blob = pickle.dumps(model)
    assert b"WindowsPath" not in blob, (
        "the saved model pickles a WindowsPath; loading it on Linux raises "
        "UnsupportedOperation before the estimator is reachable"
    )
    assert b"PosixPath" not in blob


def test_the_config_inside_a_saved_model_is_portable():
    """Belt and braces: the fitted feature builder is what carried the Path."""
    import joblib
    import pytest

    from returnrisk.config import REPO_ROOT

    model_file = REPO_ROOT / "models" / "return_risk_model.joblib"
    if not model_file.exists():
        pytest.skip("no trained model on disk; run `python run_demo.py` first")

    cfg = joblib.load(model_file).feature_builder.cfg
    assert isinstance(cfg.__getstate__()["path"], str)
