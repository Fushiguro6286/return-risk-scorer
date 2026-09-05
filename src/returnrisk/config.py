"""Configuration loading.

Why a dict-like wrapper instead of plain dicts: every module reaches for nested keys
(`cfg["money"]["actions"]["remove_cod"]["effectiveness"]`) and a typo there should fail
loudly at the point of use rather than silently return None and poison a cost curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config(Mapping[str, Any]):
    """Immutable, Mapping-compatible view over config.yaml."""

    _data: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"missing config key {key!r} in {self.path}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_path(self, key: str) -> Path:
        """Resolve a config path value against the repo root, creating parents."""
        p = REPO_ROOT / str(self["paths"][key])
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, key: str) -> Path:
        """Resolve and create a directory named in `paths`."""
        p = REPO_ROOT / str(self["paths"][key])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


def load_config(path: str | Path | None = None) -> Config:
    """Read config.yaml. Kept side-effect free so tests can point at a temp copy."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return Config(_data=data, path=p)
