"""The registry of trainable datasets, and the paths each one owns.

The committed UCI run lives in `reports/`, `models/` and `audit/`. Anything a user
uploads gets its own slug and its own three directories beside those, so two runs never
share a directory and deleting one cannot touch another. The baseline is registered
here too -- as a read-only entry -- because every screen in the dashboard picks a
dataset the same way, and the baseline being "just another dataset that happens to be
undeletable" is what keeps that code path single.

The registry file itself is the source of truth for *what exists*; the directories are
the source of truth for *what has been trained*. A dataset can be registered with no
artifacts yet (uploaded, not yet run), which is why `Dataset.is_trained` looks at disk
rather than at a flag.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import REPO_ROOT, Config

#: Where the registry itself is kept. Beside the uploads it describes.
REGISTRY_FILE = REPO_ROOT / "data" / "datasets" / "registry.json"

#: Parents of the per-dataset directories. `<parent>/<slug>/` is one dataset's world.
USER_REPORTS_ROOT = "reports_user"
USER_MODELS_ROOT = "models_user"
USER_AUDIT_ROOT = "audit_user"
UPLOAD_ROOT = "data/uploads"

#: The immutable, committed run. Not deletable, not retrainable from the dashboard.
BASELINE_KEY = "__baseline__"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", str(name).strip().lower()).strip("-")
    return slug[:48] or "dataset"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Dataset:
    """One trainable dataset and the four directories it owns."""

    key: str
    label: str
    kind: str = "user"                       # "user" | "baseline"
    created_at: str = field(default_factory=_utcnow)
    trained_at: str | None = None
    source_name: str | None = None           # original upload filename
    source_path: str | None = None           # repo-relative path to the saved upload
    table: str | None = None                 # sheet / table inside the upload
    mapping: dict[str, Any] = field(default_factory=dict)
    n_rows: int = 0
    n_orders: int = 0
    n_customers: int = 0

    # ------------------------------------------------------------------ paths
    @property
    def is_baseline(self) -> bool:
        return self.kind == "baseline"

    @property
    def reports_dir(self) -> str:
        return "reports" if self.is_baseline else f"{USER_REPORTS_ROOT}/{self.key}"

    @property
    def models_dir(self) -> str:
        return "models" if self.is_baseline else f"{USER_MODELS_ROOT}/{self.key}"

    @property
    def audit_dir(self) -> str:
        return "audit" if self.is_baseline else f"{USER_AUDIT_ROOT}/{self.key}"

    @property
    def model_file(self) -> str:
        return f"{self.models_dir}/return_risk_model.joblib"

    def path(self, which: str) -> Path:
        """Absolute path to one of this dataset's directories."""
        return REPO_ROOT / getattr(self, f"{which}_dir")

    @property
    def upload_path(self) -> Path | None:
        return REPO_ROOT / self.source_path if self.source_path else None

    # ------------------------------------------------------------------ state
    @property
    def is_trained(self) -> bool:
        """Disk, not a flag: a registered-but-never-run dataset has no model."""
        return (REPO_ROOT / self.model_file).exists()

    def summary(self) -> dict[str, Any]:
        """The run's `summary.json`, or an empty dict if it has not been trained."""
        path = self.path("reports") / "summary.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def artifact_counts(self) -> tuple[int, int]:
        d = self.path("reports")
        if not d.exists():
            return (0, 0)
        return (len(list(d.glob("*.png"))), len(list(d.glob("*.csv"))))

    # ------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "created_at": self.created_at,
            "trained_at": self.trained_at,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "table": self.table,
            "mapping": self.mapping,
            "n_rows": self.n_rows,
            "n_orders": self.n_orders,
            "n_customers": self.n_customers,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Dataset":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def baseline_dataset() -> Dataset:
    return Dataset(key=BASELINE_KEY, label="UCI baseline (committed)", kind="baseline")


class DatasetRegistry:
    """Reads and writes `data/datasets/registry.json`.

    Every mutating method rewrites the whole file. The file is small (one entry per
    upload) and the alternative -- incremental edits -- buys nothing but the chance of a
    half-written registry.
    """

    def __init__(self, path: Path | str = REGISTRY_FILE) -> None:
        self.path = Path(path)
        self._entries: dict[str, Dataset] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            self._adopt_legacy_run()
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._entries = {}
            return
        self._entries = {
            str(row["key"]): Dataset.from_dict(row)
            for row in payload.get("datasets", [])
            if row.get("key") and row.get("kind") != "baseline"
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": 1,
            "updated_at": _utcnow(),
            "datasets": [d.to_dict() for d in self._entries.values()],
        }
        self.path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    def _adopt_legacy_run(self) -> None:
        """Take ownership of a flat `reports_user/` left by an older build.

        Before per-dataset directories existed, every upload overwrote one shared
        folder. Rather than strand those artifacts where no screen can reach them, move
        them under a slug and register it.
        """
        flat_reports = REPO_ROOT / USER_REPORTS_ROOT
        flat_models = REPO_ROOT / USER_MODELS_ROOT
        if not (flat_reports / "summary.json").exists():
            return

        key = "previous-run"
        target_reports = flat_reports / key
        target_models = flat_models / key
        target_reports.mkdir(parents=True, exist_ok=True)
        target_models.mkdir(parents=True, exist_ok=True)
        for src in list(flat_reports.iterdir()):
            if src.is_file():
                src.rename(target_reports / src.name)
        if flat_models.exists():
            for src in list(flat_models.iterdir()):
                if src.is_file():
                    src.rename(target_models / src.name)

        self._entries[key] = Dataset(
            key=key,
            label="Previous run (recovered)",
            trained_at=_utcnow(),
            source_name="(from an earlier build)",
        )
        self._save()

    # ------------------------------------------------------------------ reads
    def __iter__(self) -> Iterator[Dataset]:
        return iter(self.all())

    def all(self) -> list[Dataset]:
        """Baseline first, then user datasets newest-first."""
        users = sorted(self._entries.values(), key=lambda d: d.created_at, reverse=True)
        return [baseline_dataset(), *users]

    def user_datasets(self) -> list[Dataset]:
        return [d for d in self.all() if not d.is_baseline]

    def get(self, key: str | None) -> Dataset:
        """Resolve a key, falling back to the baseline for anything unknown.

        Unknown keys are common and benign -- a stale selection in a browser session
        that outlived the dataset it pointed at -- so this returns the baseline rather
        than raising into the middle of a page render.
        """
        if not key or key == BASELINE_KEY:
            return baseline_dataset()
        entry = self._entries.get(key)
        return entry if entry is not None else baseline_dataset()

    def exists(self, key: str) -> bool:
        return key in self._entries

    # ------------------------------------------------------------------ writes
    def _unique_key(self, base: str) -> str:
        key, n = base, 2
        while key in self._entries:
            key, n = f"{base}-{n}", n + 1
        return key

    def create(
        self,
        label: str,
        *,
        upload_bytes: bytes | None = None,
        source_name: str | None = None,
        table: str | None = None,
        mapping: dict[str, Any] | None = None,
        profile_info: dict[str, Any] | None = None,
    ) -> Dataset:
        """Register a dataset and persist its upload under its own slug."""
        key = self._unique_key(_slugify(label))
        info = profile_info or {}
        dataset = Dataset(
            key=key,
            label=str(label).strip() or key,
            table=table,
            mapping=mapping or {},
            source_name=source_name,
            n_rows=int(info.get("n_rows", 0)),
            n_orders=int(info.get("n_orders", 0)),
            n_customers=int(info.get("n_customers", 0)),
        )

        if upload_bytes is not None and source_name:
            dest_dir = REPO_ROOT / UPLOAD_ROOT / key
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / source_name
            dest.write_bytes(upload_bytes)
            dataset.source_path = str(Path(UPLOAD_ROOT) / key / source_name).replace("\\", "/")

        self._entries[key] = dataset
        self._save()
        return dataset

    def mark_trained(self, key: str) -> None:
        if key in self._entries:
            self._entries[key].trained_at = _utcnow()
            self._save()

    def rename(self, key: str, label: str) -> None:
        if key in self._entries:
            self._entries[key].label = str(label).strip() or key
            self._save()

    def delete(self, key: str) -> list[str]:
        """Remove a dataset's upload, model, reports and ledger.

        Returns the repo-relative paths actually removed, so the caller can show the
        user what went rather than a bare "deleted". The committed baseline is refused
        outright -- it is the reference every user run is read against.
        """
        if key == BASELINE_KEY:
            raise ValueError("the committed UCI baseline cannot be deleted")
        dataset = self._entries.get(key)
        if dataset is None:
            raise KeyError(f"no dataset named {key!r}")

        removed: list[str] = []
        targets = [
            dataset.path("reports"),
            dataset.path("models"),
            dataset.path("audit"),
        ]
        upload = dataset.upload_path
        if upload is not None:
            targets.append(upload.parent)

        for target in targets:
            # Refuse to walk outside the repo, whatever the registry claims.
            try:
                rel = target.resolve().relative_to(REPO_ROOT.resolve())
            except ValueError:
                continue
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(str(rel).replace("\\", "/"))

        del self._entries[key]
        self._save()
        return removed


def config_for(dataset: Dataset, base: Config) -> Config:
    """A `Config` whose paths point at this dataset's directories.

    Everything downstream -- the model loader, the ledger, the pipeline's artifact
    writer -- already resolves its location through `cfg["paths"]`, so redirecting a
    whole run is a matter of swapping four strings rather than threading a directory
    argument through every module.

    The FX rate travels with the paths. `money.gbp_to_inr` is 105 because the committed
    dataset is a British wholesaler priced in pounds; a user file is denominated in
    whatever `mapping.currency_to_inr` says (1.0 by default -- already rupees). Scoring a
    rupee-priced order at 105x inflates the cost of a missed return until every order is
    worth actioning, so this has to be swapped here and not only at training time.
    """
    payload = base.as_dict()
    payload["paths"] = {
        **payload.get("paths", {}),
        "reports_dir": dataset.reports_dir,
        "models_dir": dataset.models_dir,
        "model_file": dataset.model_file,
        "audit_dir": dataset.audit_dir,
    }
    if not dataset.is_baseline:
        rate = float((dataset.mapping or {}).get("currency_to_inr", 1.0) or 1.0)
        payload["money"] = {**payload.get("money", {}), "gbp_to_inr": rate}
    return Config(_data=payload, path=base.path)


def training_config_for(dataset: Dataset, base: Config) -> Config:
    """`config_for`, plus the `data.file` block that points the pipeline at the upload."""
    if dataset.is_baseline:
        raise ValueError("the baseline is trained by run_demo.py, not from the dashboard")
    if dataset.upload_path is None:
        raise ValueError(f"dataset {dataset.key!r} has no stored upload to train on")

    cfg = config_for(dataset, base)
    payload = cfg.as_dict()
    mapping = dataset.mapping or {}
    payload["data"] = {
        **payload.get("data", {}),
        "source": "file",
        "file": {
            **(payload.get("data", {}).get("file") or {}),
            "path": str(dataset.upload_path),
            "table": dataset.table,
            "mapping": mapping,
            "currency_to_inr": mapping.get("currency_to_inr", 1.0),
        },
    }
    return Config(_data=payload, path=base.path)
