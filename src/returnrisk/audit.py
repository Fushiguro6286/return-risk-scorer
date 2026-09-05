"""Append-only, hash-chained decision ledger.

A risk decision you cannot reconstruct six months later is not a risk decision, it is a
guess that happened to be logged. When a merchant asks "why did you block COD on this
order in March", the answer has to survive the model having been retrained twice since:
you cannot re-score the order and get the same number back, so the number, the inputs it
came from, the threshold in force, and every guardrail that fired must have been written
down at the time.

That is what this module is. One JSON object per decision, appended to a file, never
edited.

**Why hash-chained.** An append-only file is only append-only by convention -- anyone can
open it and change a score. Each record therefore carries the SHA-256 of the record
before it, so altering any past entry breaks every hash after it and `verify()` reports
the exact sequence number where the chain parts. This is tamper *evidence*, not tamper
*proof*: it does not stop someone rewriting the whole file from scratch, and it is not
meant to. It means a silent single-row edit is not possible, which is the realistic
threat for an internal ledger.

**What is recorded.** Enough to replay the decision without the model: the feature row
actually scored, the score, the threshold and which action it belonged to, the policy
verdict with every rule that fired, and the config fingerprint. Plus, later and
separately, the observed outcome -- because a ledger that never learns whether it was
right is just a diary.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import REPO_ROOT, Config

LEDGER_FILE = "decisions.jsonl"
OUTCOME_FILE = "outcomes.jsonl"

#: The hash a chain starts from. Any value works; a constant makes an empty ledger
#: verifiable rather than a special case.
GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    """Stable JSON for hashing.

    `sort_keys` and a fixed separator matter: a dict that serialises differently on two
    machines would break the chain for reasons that have nothing to do with tampering.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def record_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def fingerprint(payload: dict[str, Any]) -> str:
    """A short, stable id for an order's economically meaningful content.

    Used for holdout membership and for spotting the same order being scored twice.
    Deliberately excludes the timestamp -- the same basket scored twice is the same
    basket.
    """
    return record_hash(payload)[:16]


def config_fingerprint(cfg: Config) -> str:
    """Which config produced this decision.

    Costs and thresholds live in YAML; a decision is only reproducible if you know
    which version of that YAML was in force.
    """
    return record_hash(cfg.as_dict())[:16]


@dataclass
class LedgerEntry:
    """One row of the ledger, already hashed and chained."""

    seq: int
    decision_id: str
    recorded_at: str
    prev_hash: str
    entry_hash: str
    body: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "decision_id": self.decision_id,
                "recorded_at": self.recorded_at,
                "prev_hash": self.prev_hash,
                "entry_hash": self.entry_hash,
                **self.body,
            },
            sort_keys=True,
            default=str,
        )


class DecisionLedger:
    """Append-only decision log with a verifiable hash chain.

    Concurrency: a process-level lock plus line-buffered append. Two workers writing the
    same file would race on `prev_hash`, so a multi-process deployment wants a real
    append log (Kafka, a DB with a sequence) behind this same interface -- the interface
    is the point, the JSONL is the demo-scale implementation, and `verify()` would catch
    it if that assumption were violated.
    """

    def __init__(self, directory: Path | str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / LEDGER_FILE
        self.outcome_path = self.dir / OUTCOME_FILE
        self._lock = threading.Lock()
        self._tip: tuple[int, str] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> "DecisionLedger":
        directory = REPO_ROOT / str(cfg["paths"].get("audit_dir", "audit"))
        return cls(directory)

    # -- reading -----------------------------------------------------------------
    def __len__(self) -> int:
        return sum(1 for _ in self.entries())

    def entries(self) -> Iterator[dict[str, Any]]:
        """Every record, in the order it was written. Corrupt lines are skipped here
        and reported by `verify()` -- reading should not explode on a bad byte."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        rows = list(self.entries())
        return rows[-n:][::-1]

    def get(self, decision_id: str) -> dict[str, Any] | None:
        for row in self.entries():
            if row.get("decision_id") == decision_id:
                return row
        return None

    def _tip_state(self) -> tuple[int, str]:
        """Sequence number and hash of the last record. Cached after first read."""
        if self._tip is None:
            seq, prev = 0, GENESIS
            for row in self.entries():
                seq = int(row.get("seq", seq)) + 1
                prev = str(row.get("entry_hash", prev))
            self._tip = (seq, prev)
        return self._tip

    # -- writing -----------------------------------------------------------------
    def append(self, body: dict[str, Any], decision_id: str | None = None) -> LedgerEntry:
        """Write one decision and return the entry, including its id and hash."""
        with self._lock:
            seq, prev = self._tip_state()
            entry = {
                "seq": seq,
                "decision_id": decision_id or uuid.uuid4().hex,
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "prev_hash": prev,
                **body,
            }
            digest = record_hash(entry)
            row = LedgerEntry(
                seq=seq,
                decision_id=str(entry["decision_id"]),
                recorded_at=str(entry["recorded_at"]),
                prev_hash=prev,
                entry_hash=digest,
                body=body,
            )
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(row.to_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._tip = (seq + 1, digest)
            return row

    def record_outcome(
        self, decision_id: str, returned: bool, note: str = ""
    ) -> dict[str, Any]:
        """Close the loop: what actually happened to an order we decided on.

        Kept in a second file rather than mutating the decision row, because the whole
        value of the ledger is that a decision row is never edited after the fact.
        """
        row = {
            "decision_id": decision_id,
            "returned": bool(returned),
            "note": note,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        with self._lock:
            with open(self.outcome_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
        return row

    def outcomes(self) -> dict[str, dict[str, Any]]:
        """Latest recorded outcome per decision."""
        out: dict[str, dict[str, Any]] = {}
        if not self.outcome_path.exists():
            return out
        with open(self.outcome_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[str(row.get("decision_id"))] = row
        return out

    # -- integrity ---------------------------------------------------------------
    def verify(self) -> dict[str, Any]:
        """Walk the chain and report the first sequence number that does not line up."""
        prev = GENESIS
        n = 0
        for expected_seq, row in enumerate(self.entries()):
            n += 1
            stored = str(row.get("entry_hash", ""))
            body = {k: v for k, v in row.items() if k != "entry_hash"}
            if int(row.get("seq", -1)) != expected_seq:
                return {
                    "ok": False,
                    "n_entries": n,
                    "broken_at": expected_seq,
                    "reason": f"sequence number is {row.get('seq')}, expected {expected_seq}",
                }
            if str(row.get("prev_hash")) != prev:
                return {
                    "ok": False,
                    "n_entries": n,
                    "broken_at": expected_seq,
                    "reason": "prev_hash does not match the previous entry",
                }
            if record_hash(body) != stored:
                return {
                    "ok": False,
                    "n_entries": n,
                    "broken_at": expected_seq,
                    "reason": "entry hash does not match its contents - record was edited",
                }
            prev = stored
        return {"ok": True, "n_entries": n, "broken_at": None, "reason": "chain intact"}

    # -- aggregates the policy layer needs ---------------------------------------
    def action_rate(self, on_date: dt.date | None = None) -> float:
        """Share of a day's decisions that carried an automatic action.

        This is what feeds the portfolio cap in `policy.py`. Computed from the ledger
        rather than an in-memory counter so a restart does not reset the budget.
        """
        day = on_date or dt.datetime.now(dt.timezone.utc).date()
        total = acted = 0
        for row in self.entries():
            ts = str(row.get("recorded_at", ""))[:10]
            if ts != day.isoformat():
                continue
            total += 1
            if row.get("final_action") not in (None, "none") and not row.get(
                "requires_human_review", False
            ):
                acted += 1
        return acted / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        """Operational summary -- what the ledger knows about itself."""
        rows = list(self.entries())
        outcomes = self.outcomes()
        by_outcome: dict[str, int] = {}
        by_action: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        saving = 0.0
        for row in rows:
            by_outcome[str(row.get("outcome", "unknown"))] = (
                by_outcome.get(str(row.get("outcome", "unknown")), 0) + 1
            )
            act = str(row.get("final_action", "none"))
            by_action[act] = by_action.get(act, 0) + 1
            for hit in row.get("rules_fired", []) or []:
                key = str(hit.get("rule"))
                by_rule[key] = by_rule.get(key, 0) + 1
            if row.get("final_action") not in (None, "none"):
                saving += float(row.get("expected_saving_inr", 0.0) or 0.0)
        return {
            "n_decisions": len(rows),
            "n_outcomes_recorded": len(outcomes),
            "by_outcome": by_outcome,
            "by_final_action": by_action,
            "guardrails_fired": by_rule,
            "expected_saving_inr_total": round(saving, 2),
            "action_rate_today": round(self.action_rate(), 4),
            "chain": self.verify(),
        }


__all__ = [
    "GENESIS",
    "LEDGER_FILE",
    "OUTCOME_FILE",
    "DecisionLedger",
    "LedgerEntry",
    "config_fingerprint",
    "fingerprint",
    "record_hash",
]
