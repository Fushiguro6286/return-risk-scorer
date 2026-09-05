"""Bring-your-own-data: read an arbitrary transaction file and shape it into the
canonical frame the rest of the pipeline already understands.

The pipeline is written against one schema -- line-level retail transactions with
`invoice, stock_code, description, quantity, invoice_date, unit_price, customer_id,
country`, where a return is a *cancellation invoice* carrying negative quantity. Every
guarantee this repo makes (the leakage allow-list, the 90-day embargo, the greedy
cancellation linking, the right-censoring rule) is expressed in terms of that schema.

So this module does **not** add a second pipeline. It adds a front door:

    your file  ->  read_any  ->  column mapping  ->  validate  ->  canonical frame
                                                                        |
                                              exactly what load_uci() returns
                                                                        v
                                                     ...the existing 12-step pipeline

Two things make an arbitrary file fit.

**Column mapping.** Nobody else's export is named `StockCode`. `suggest_mapping` guesses
from column names and dtypes, and the guess is always presented for confirmation rather
than applied silently -- a mis-mapped date column would produce a plausible-looking model
that is entirely wrong, which is the worst failure mode available here.

**Return convention.** Real exports record returns two ways, and both are supported:

* ``cancellation`` -- the UCI convention. A reversing invoice with negative quantity.
  Used as-is.
* ``flag`` -- an explicit `returned` / `is_return` column, which is what most order
  management systems export. Here the file already knows which order came back, so the
  label is carried through as ground truth. The cancellation-matching heuristic is
  *not* reused for these: it works by pinning a reversal to the most recent prior
  purchase of the same (customer, SKU), and where the true link is known that guess is
  simply wrong -- a customer who buys one SKU twice would have the return attributed to
  the wrong order, mislabelling both. Censoring and the leakage-safe history rules are
  shared with the cancellation path, because those are about time, not discovery.

Flag mode still needs a date on which the return became *known*, because
`prior_return_rate` must only count returns the merchant had already observed at
checkout. If the file has a return-date column, it is used. If it does not, a fixed lag
is assumed and **loudly reported**, because an assumed lag is a real modelling
assumption and not a detail.
"""
from __future__ import annotations

import io
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd

from ..config import Config

#: Extensions we can read. `.db`/`.sqlite` are included because "any database file"
#: usually means exactly that.
TABULAR_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".ndjson"}
)
SQLITE_SUFFIXES: Final[frozenset[str]] = frozenset({".db", ".sqlite", ".sqlite3"})
READABLE_SUFFIXES: Final[frozenset[str]] = TABULAR_SUFFIXES | SQLITE_SUFFIXES | {".zip", ".gz"}

#: Values that mean "yes this came back" in a flag column, however it was exported.
TRUTHY: Final[frozenset[str]] = frozenset(
    {"1", "1.0", "true", "t", "yes", "y", "returned", "return", "refunded", "cancelled",
     "canceled", "rto", "is_return"}
)


@dataclass(frozen=True)
class FieldSpec:
    """One canonical column: what it means, and how to spot it in someone else's file."""

    name: str
    required: bool
    kind: str  # "id" | "date" | "number" | "text"
    description: str
    synonyms: tuple[str, ...]


#: The target schema. `required` here means "the pipeline cannot run without it".
CANONICAL_FIELDS: Final[tuple[FieldSpec, ...]] = (
    FieldSpec(
        "invoice", True, "id",
        "Order / invoice / transaction id. Lines sharing this id form one order.",
        ("invoice", "invoiceno", "invoice_no", "invoice_id", "order", "orderid", "order_id",
         "order_no", "ordernumber", "order_number", "transaction_id", "txn_id", "receipt",
         "bill_no", "billno", "sale_id"),
    ),
    FieldSpec(
        "invoice_date", True, "date",
        "When the order was placed. Everything temporal depends on this.",
        ("invoicedate", "invoice_date", "order_date", "orderdate", "date", "created_at",
         "createdat", "timestamp", "purchase_date", "transaction_date", "placed_at",
         "order_placed_at", "datetime"),
    ),
    FieldSpec(
        "stock_code", True, "id",
        "Product / SKU identifier. Returns are matched per product.",
        ("stockcode", "stock_code", "sku", "product_id", "productid", "product_code",
         "item_id", "itemid", "item_code", "article", "asin", "variant_id", "product"),
    ),
    FieldSpec(
        "quantity", True, "number",
        "Units on this line. Negative marks a reversal in cancellation mode.",
        ("quantity", "qty", "units", "count", "item_count", "quantity_ordered", "no_of_items",
         "amount_qty"),
    ),
    FieldSpec(
        "unit_price", True, "number",
        "Price per unit, before discount. Line value = quantity x unit_price.",
        ("price", "unitprice", "unit_price", "rate", "mrp", "selling_price", "item_price",
         "price_per_unit", "unit_cost", "amount_per_unit"),
    ),
    FieldSpec(
        "customer_id", True, "id",
        "Who placed it. Needed for purchase history and to attribute a return.",
        ("customerid", "customer id", "customer_id", "custid", "cust_id", "user_id", "userid",
         "buyer_id", "client_id", "account_id", "shopper_id", "email"),
    ),
    FieldSpec(
        "country", False, "text",
        "Destination. Becomes a feature; filled with 'Unknown' if absent.",
        ("country", "ship_country", "shipping_country", "destination", "region", "market",
         "country_code", "state", "city"),
    ),
    FieldSpec(
        "description", False, "text",
        "Product name. Used only for readable reason codes.",
        ("description", "product_name", "productname", "item_name", "itemname", "title",
         "product_title", "name", "item_description"),
    ),
    FieldSpec(
        "category", False, "text",
        "Product family / category. Without it, a family is guessed from the SKU prefix.",
        ("category", "product_category", "productcategory", "category_name", "product_type",
         "producttype", "item_category", "department", "product_group", "family", "vertical",
         "segment", "collection"),
    ),
    FieldSpec(
        "returned", False, "text",
        "FLAG MODE ONLY: was this order returned? Any of 1/true/yes/returned.",
        ("returned", "is_return", "isreturn", "is_returned", "return_flag", "returnflag",
         "refunded", "is_refunded", "cancelled", "canceled", "status", "return_status",
         "rto", "order_status"),
    ),
    FieldSpec(
        "return_date", False, "date",
        "FLAG MODE ONLY: when the return became known. Absent means a lag is assumed.",
        ("return_date", "returndate", "returned_on", "return_on", "returned_date",
         "returned_at", "return_created_at", "refund_date", "refunded_at", "cancelled_at",
         "canceled_at", "rto_date", "return_initiated_at"),
    ),
)

FIELDS_BY_NAME: Final[dict[str, FieldSpec]] = {f.name: f for f in CANONICAL_FIELDS}
REQUIRED_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in CANONICAL_FIELDS if f.required)


# --------------------------------------------------------------------------- issues
@dataclass
class Issue:
    """One validation finding. `error` blocks the run; `warning` is reported and kept."""

    level: str  # "error" | "warning" | "info"
    message: str
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message, "fix": self.fix}


@dataclass
class ColumnMapping:
    """How the uploaded file's columns line up with the canonical schema."""

    columns: dict[str, str | None] = field(default_factory=dict)
    return_mode: str = "cancellation"          # "cancellation" | "flag"
    assumed_return_lag_days: float = 21.0      # flag mode, only when no return date
    currency_to_inr: float = 1.0               # uploaded amounts -> INR

    def source_for(self, canonical: str) -> str | None:
        return self.columns.get(canonical)

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not self.columns.get(f)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": dict(self.columns),
            "return_mode": self.return_mode,
            "assumed_return_lag_days": self.assumed_return_lag_days,
            "currency_to_inr": self.currency_to_inr,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnMapping":
        return cls(
            columns=dict(payload.get("columns") or {}),
            return_mode=str(payload.get("return_mode", "cancellation")),
            assumed_return_lag_days=float(payload.get("assumed_return_lag_days", 21.0)),
            currency_to_inr=float(payload.get("currency_to_inr", 1.0)),
        )


# ---------------------------------------------------------------------------- reading
def list_tables(path: str | Path) -> list[str]:
    """Table names in a SQLite file, or sheet names in a workbook. Empty otherwise."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in SQLITE_SUFFIXES:
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]
    if suffix in {".xlsx", ".xls"}:
        return list(pd.read_excel(p, sheet_name=None, engine="openpyxl", nrows=0))
    return []


def _read_delimited(buffer: Any, name: str) -> pd.DataFrame:
    """CSV/TSV with delimiter and encoding sniffing, because exports are messy."""
    seps = ["\t", ";", "|"] if name.lower().endswith(".tsv") else [",", ";", "\t", "|"]
    last: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        for sep in seps:
            try:
                if hasattr(buffer, "seek"):
                    buffer.seek(0)
                # The C engine is far faster on large exports and is the one that
                # supports low_memory; the python engine is only the fallback for
                # files it cannot tokenise.
                df = pd.read_csv(
                    buffer, sep=sep, encoding=encoding, low_memory=False,
                    on_bad_lines="skip",
                )
                # One column usually means the separator guess was wrong.
                if df.shape[1] > 1:
                    return df
                last = ValueError(f"only one column parsed with sep={sep!r}")
            except Exception as exc:
                last = exc
    # Last resort: let pandas sniff the dialect itself.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(buffer, "seek"):
                buffer.seek(0)
            df = pd.read_csv(
                buffer, sep=None, encoding=encoding, engine="python", on_bad_lines="skip"
            )
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            last = exc
    raise ValueError(f"could not parse {name} as delimited text: {last}")


def read_any(
    path: str | Path,
    table: str | None = None,
    sheet: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read one tabular file into a DataFrame, whatever container it arrived in.

    Supports CSV/TSV, Excel, Parquet, JSON/JSONL, SQLite, and a zip or gzip wrapping
    any of those. Raises with a readable message rather than a stack trace, because
    this runs behind a file-upload box.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    suffix = p.suffix.lower()

    if suffix in SQLITE_SUFFIXES:
        tables = list_tables(p)
        if not tables:
            raise ValueError(f"{p.name} contains no readable tables")
        chosen = table or _largest_table(p, tables)
        limit = f" LIMIT {int(max_rows)}" if max_rows else ""
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            return pd.read_sql_query(f'SELECT * FROM "{chosen}"{limit}', conn)

    if suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            inner = [
                n for n in zf.namelist()
                if Path(n).suffix.lower() in TABULAR_SUFFIXES and not n.startswith("__MACOSX")
            ]
            if not inner:
                raise ValueError(f"{p.name} has no readable data file inside it")
            name = table or inner[0]
            blob = zf.read(name)
        return _read_blob(blob, name, sheet=sheet)

    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet or 0, engine="openpyxl")
    if suffix in {".json"}:
        return pd.read_json(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix in {".csv", ".tsv", ".txt", ".gz"}:
        return _read_delimited(p, p.name)

    raise ValueError(
        f"unsupported file type {suffix!r}. Supported: "
        f"{', '.join(sorted(READABLE_SUFFIXES))}"
    )


def _read_blob(blob: bytes, name: str, sheet: str | None = None) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(blob))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(blob), sheet_name=sheet or 0, engine="openpyxl")
    if suffix == ".json":
        return pd.read_json(io.BytesIO(blob))
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(io.BytesIO(blob), lines=True)
    return _read_delimited(io.BytesIO(blob), name)


def _largest_table(path: Path, tables: Iterable[str]) -> str:
    """Pick the table with the most rows -- in an exported DB that is the fact table."""
    best, best_n = None, -1
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for name in tables:
            try:
                n = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            except sqlite3.Error:
                continue
            if n > best_n:
                best, best_n = name, n
    if best is None:
        raise ValueError("no readable table found")
    return best


# --------------------------------------------------------------------------- sniffing
def _norm(name: str) -> str:
    """Fold a column name to letters and digits only.

    Underscores are stripped as well as spaces and punctuation, so that the synonym
    `order_status` and the real-world header `Order Status` normalise to the same
    thing. Keeping underscores here silently defeated every multi-word synonym.
    """
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _looks_like_date(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) > 0.8


def _looks_boolean(s: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in s.dropna().unique()[:20]}
    return 0 < len(vals) <= 4 and vals <= (TRUTHY | {"0", "0.0", "false", "f", "no", "n",
                                                    "kept", "delivered", "completed", "none"})


def suggest_mapping(df: pd.DataFrame) -> tuple[ColumnMapping, list[Issue]]:
    """Guess which source column is which canonical field.

    Name matching first (exact normalised, then substring), dtype heuristics only as a
    tiebreak. The result is a *suggestion*: `validate` still runs, and the UI shows it
    for confirmation. Silently trusting a guess here is how you get a confident model
    trained on the wrong date column.
    """
    issues: list[Issue] = []
    mapping = ColumnMapping()
    used: set[str] = set()
    norm_to_source = {_norm(c): c for c in df.columns}

    for spec in CANONICAL_FIELDS:
        pick: str | None = None
        for syn in spec.synonyms:
            candidate = norm_to_source.get(_norm(syn))
            if candidate is not None and candidate not in used:
                pick = candidate
                break
        if pick is None:  # substring pass
            for syn in spec.synonyms:
                token = _norm(syn)
                if len(token) < 3:
                    continue
                for norm_name, source in norm_to_source.items():
                    if source in used:
                        continue
                    # Short tokens ("sku", "qty") only match as a substring of the
                    # column name -- the reverse direction would let "sku" claim any
                    # column whose name happens to sit inside it.
                    if token in norm_name or (len(token) >= 4 and norm_name in token):
                        pick = source
                        break
                if pick:
                    break
        if pick is not None:
            mapping.columns[spec.name] = pick
            used.add(pick)
        else:
            mapping.columns[spec.name] = None

    # Dtype fallback for the two fields a file cannot do without and that are easy to spot.
    if mapping.columns.get("invoice_date") is None:
        for col in df.columns:
            if col not in used and _looks_like_date(df[col]):
                mapping.columns["invoice_date"] = col
                used.add(col)
                issues.append(
                    Issue("warning", f"Guessed {col!r} as the order date from its values, "
                                     f"not its name.", "Confirm this is the order date.")
                )
                break

    # Return convention: a boolean-ish flag column means flag mode.
    flag_col = mapping.columns.get("returned")
    if flag_col is not None and _looks_boolean(df[flag_col]):
        mapping.return_mode = "flag"
    elif flag_col is not None:
        # Named like a flag but not boolean -- e.g. a free-text status column.
        uniques = df[flag_col].dropna().astype(str).str.lower().unique()
        if any(v in TRUTHY for v in uniques):
            mapping.return_mode = "flag"
            issues.append(
                Issue("warning",
                      f"{flag_col!r} looks like a status column, not a boolean. Rows matching "
                      f"{sorted(TRUTHY & set(uniques))} will count as returned.",
                      "Check that this captures every return.")
            )
        else:
            mapping.columns["returned"] = None
            mapping.return_mode = "cancellation"

    if mapping.return_mode == "cancellation":
        qty = mapping.columns.get("quantity")
        has_negative = qty is not None and bool(
            (pd.to_numeric(df[qty], errors="coerce") < 0).any()
        )
        if not has_negative:
            issues.append(
                Issue(
                    "error",
                    "No return information found: there is no returned/is_return column, and "
                    "no negative quantities that would mark a cancellation.",
                    "Map a 'returned' column, or supply data using cancellation rows "
                    "(negative quantity).",
                )
            )

    for missing in mapping.missing_required():
        issues.append(
            Issue("error", f"Required field {missing!r} could not be matched to any column.",
                  FIELDS_BY_NAME[missing].description)
        )
    return mapping, issues


# ------------------------------------------------------------------------- validating
def validate(df: pd.DataFrame, mapping: ColumnMapping) -> list[Issue]:
    """Check a mapped file is usable *before* spending a minute training on it."""
    issues: list[Issue] = []

    for missing in mapping.missing_required():
        issues.append(
            Issue("error", f"Required field {missing!r} is not mapped.",
                  FIELDS_BY_NAME[missing].description)
        )
    if issues:
        return issues

    if len(df) < 500:
        issues.append(
            Issue("error", f"Only {len(df):,} rows. A temporal split with a 90-day embargo "
                           f"needs meaningfully more than this.",
                  "Supply at least a few thousand transaction lines.")
        )

    date_col = mapping.source_for("invoice_date")
    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    if float(dates.notna().mean()) < 0.5:
        issues.append(
            Issue("error", f"{date_col!r} did not parse as dates for most rows.",
                  "Map the correct order-date column.")
        )
    else:
        span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
        if span_days < 180:
            issues.append(
                Issue("error", f"Data spans only {span_days} days. The split needs train, "
                               f"calibration and test slices plus a 90-day embargo.",
                      "Supply at least ~9 months of history; a year or more is better.")
            )
        elif span_days < 365:
            issues.append(
                Issue("warning", f"Data spans {span_days} days. This will work, but the test "
                                 f"slice will be short and the metrics noisy.", "")
            )

    qty_col = mapping.source_for("quantity")
    price_col = mapping.source_for("unit_price")
    for col, label in ((qty_col, "quantity"), (price_col, "unit_price")):
        numeric = pd.to_numeric(df[col], errors="coerce")
        if float(numeric.notna().mean()) < 0.5:
            issues.append(
                Issue("error", f"{col!r} (mapped to {label}) is not numeric for most rows.",
                      "Map a numeric column, or clean the source.")
            )

    if not mapping.source_for("category"):
        sku_col = mapping.source_for("stock_code")
        codes = df[sku_col].astype(str).str.replace(r"[^0-9A-Z]", "", regex=True).str[:3]
        n_families = int(codes.nunique())
        if n_families <= 2:
            issues.append(
                Issue(
                    "warning",
                    f"No category column, and guessing one from the SKU prefix yields only "
                    f"{n_families} family/families -- your SKUs share a common prefix, so "
                    f"product-level signal will be invisible to the model.",
                    "Map a product category column if you have one. Without it the model "
                    "can still use price, basket and customer-history features.",
                )
            )

    cust_col = mapping.source_for("customer_id")
    missing_cust = float(df[cust_col].isna().mean())
    if missing_cust > 0.5:
        issues.append(
            Issue("warning", f"{missing_cust:.0%} of rows have no customer id. Those rows are "
                             f"dropped -- they cannot carry purchase history.", "")
        )

    if mapping.return_mode == "flag":
        flag_col = mapping.source_for("returned")
        if not flag_col:
            issues.append(Issue("error", "Return mode is 'flag' but no returned column is mapped.", ""))
        else:
            rate = float(_truthy(df[flag_col]).mean())
            if rate == 0:
                issues.append(Issue("error", "No row is marked as returned.",
                                    "Check the returned column and its true-values."))
            elif rate > 0.6:
                issues.append(Issue("warning", f"{rate:.0%} of rows are marked returned, which "
                                               f"is unusually high.", "Check the mapping."))
            else:
                issues.append(Issue("info", f"Return rate in the file: {rate:.2%}.", ""))
        if not mapping.source_for("return_date"):
            issues.append(
                Issue(
                    "warning",
                    f"No return-date column. A fixed {mapping.assumed_return_lag_days:.0f}-day "
                    f"lag will be assumed for when each return became known.",
                    "This is a real modelling assumption: it decides when a past return "
                    "enters a customer's history. Map a return-date column if you have one.",
                )
            )
    return issues


def _truthy(s: pd.Series) -> pd.Series:
    """Interpret a flag column that could be bool, int, or free text."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0) > 0
    return s.astype(str).str.strip().str.lower().isin(TRUTHY)


# ------------------------------------------------------------------------ conversion
def to_canonical(
    df: pd.DataFrame, mapping: ColumnMapping
) -> tuple[pd.DataFrame, list[Issue]]:
    """Produce the exact frame `load_uci` produces, from an arbitrary mapped file.

    In flag mode this synthesises cancellation rows, so that from here on the labeller,
    the embargo and the leakage guards all run on the same shapes they were written and
    tested against.
    """
    notes: list[Issue] = []
    missing = mapping.missing_required()
    if missing:
        raise ValueError(f"cannot convert: unmapped required fields {missing}")

    out = pd.DataFrame(index=df.index)
    out["invoice"] = df[mapping.source_for("invoice")].astype("string").str.strip()
    out["stock_code"] = df[mapping.source_for("stock_code")].astype("string").str.strip()

    desc_col = mapping.source_for("description")
    out["description"] = (
        df[desc_col].astype("string") if desc_col else "ITEM " + out["stock_code"]
    )

    out["quantity"] = pd.to_numeric(df[mapping.source_for("quantity")], errors="coerce")
    out["invoice_date"] = pd.to_datetime(
        df[mapping.source_for("invoice_date")], errors="coerce", format="mixed"
    )
    out["unit_price"] = pd.to_numeric(df[mapping.source_for("unit_price")], errors="coerce")

    # customer_id is numeric downstream (`clean` casts float -> int -> string), so a
    # non-numeric id (an email, a UUID) is factorised to a stable integer instead.
    raw_cust = df[mapping.source_for("customer_id")]
    numeric_cust = pd.to_numeric(raw_cust, errors="coerce")
    if float(numeric_cust.notna().mean()) > 0.9:
        out["customer_id"] = numeric_cust.astype("float64")
    else:
        codes, _ = pd.factorize(raw_cust.astype("string"))
        out["customer_id"] = np.where(codes < 0, np.nan, codes + 1).astype("float64")
        notes.append(
            Issue("info", "Customer ids are not numeric; they were mapped to stable integers.", "")
        )

    cat_col = mapping.source_for("category")
    if cat_col:
        out["category"] = df[cat_col].astype("string").str.strip()

    country_col = mapping.source_for("country")
    out["country"] = (
        df[country_col].astype("string").str.strip() if country_col else pd.Series("Unknown", index=df.index, dtype="string")
    )

    if mapping.currency_to_inr and mapping.currency_to_inr != 1.0:
        notes.append(
            Issue("info", f"Prices will be converted at {mapping.currency_to_inr} to the rupee.", "")
        )

    if mapping.return_mode == "cancellation":
        n_neg = int((out["quantity"] < 0).sum())
        notes.append(
            Issue("info", f"Cancellation mode: {n_neg:,} reversing lines found.", "")
        )
        return out.reset_index(drop=True), notes

    # ---- flag mode: synthesise the cancellation lines the labeller expects ----------
    flag_col = mapping.source_for("returned")
    is_returned = _truthy(df[flag_col])

    date_col = mapping.source_for("return_date")
    if date_col:
        return_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        # A return dated before its order is a data error; fall back to the lag.
        bad = return_dates.notna() & (return_dates < out["invoice_date"])
        if bad.any():
            notes.append(
                Issue("warning", f"{int(bad.sum()):,} return dates fall before their order date "
                                 f"and were replaced with the assumed lag.", "")
            )
            return_dates = return_dates.mask(bad)
        filled = return_dates.isna() & is_returned
        if filled.any():
            notes.append(
                Issue("warning", f"{int(filled.sum()):,} returned rows have no return date; the "
                                 f"{mapping.assumed_return_lag_days:.0f}-day assumption was used "
                                 f"for those.", "")
            )
        return_dates = return_dates.fillna(
            out["invoice_date"] + pd.Timedelta(days=float(mapping.assumed_return_lag_days))
        )
    else:
        return_dates = out["invoice_date"] + pd.Timedelta(
            days=float(mapping.assumed_return_lag_days)
        )
        notes.append(
            Issue(
                "warning",
                f"No return-date column: every return is assumed to become known "
                f"{mapping.assumed_return_lag_days:.0f} days after the order.",
                "This sets when a past return enters a customer's history, so it affects "
                "the prior_return_rate feature. Map a real return date if you have one.",
            )
        )

    if not bool(is_returned.any()):
        raise ValueError("flag mode selected but no row is marked as returned")

    # Carry the ground truth through to the labeller rather than re-deriving it.
    #
    # The obvious alternative -- synthesise a cancellation row per return and let the
    # standard linker figure it out -- is wrong, and quietly so. That linker matches a
    # cancellation to the most recent prior purchase of the same (customer, SKU),
    # because Online Retail II gives it no foreign key to follow. Here we *have* the
    # foreign key. Throwing it away means a customer who buys the same SKU twice gets
    # the return pinned to the wrong order, mislabelling both; on a file with any
    # repeat purchasing that is enough to erase the signal completely.
    purchases = out.copy()
    purchases["_returned_flag"] = is_returned.to_numpy().astype(bool)
    purchases["_return_event_date"] = pd.to_datetime(return_dates).where(is_returned)

    # Cancellation rows are still emitted, so the reversal is visible in the transaction
    # log the way a merchant would expect. They carry no labelling authority.
    cancels = purchases[is_returned].copy()
    cancels["invoice"] = "C" + cancels["invoice"].astype("string")
    cancels["quantity"] = -cancels["quantity"].abs()
    cancels["invoice_date"] = pd.to_datetime(return_dates[is_returned]).to_numpy()
    cancels["_returned_flag"] = False
    cancels["_return_event_date"] = pd.NaT

    n_orders_returned = int(purchases.loc[is_returned, "invoice"].nunique())
    notes.append(
        Issue(
            "info",
            f"Flag mode: {n_orders_returned:,} orders are labelled returned directly from "
            f"the file. The label is ground truth here, not the cancellation-matching "
            f"heuristic the UCI dataset needs.",
            "",
        )
    )
    combined = pd.concat([purchases, cancels], ignore_index=True)
    return combined.sort_values("invoice_date", ignore_index=True), notes


# -------------------------------------------------------------------------- profiling
def profile(df: pd.DataFrame, mapping: ColumnMapping | None = None) -> dict[str, Any]:
    """A quick description of an uploaded file, for the confirmation screen."""
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 1),
    }
    if mapping is None:
        return out
    date_col = mapping.source_for("invoice_date")
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        if dates.notna().any():
            out["date_min"] = str(dates.min().date())
            out["date_max"] = str(dates.max().date())
            out["span_days"] = int((dates.max() - dates.min()).days)
    for canonical, label in (("invoice", "n_orders"), ("customer_id", "n_customers"),
                             ("stock_code", "n_products")):
        col = mapping.source_for(canonical)
        if col and col in df.columns:
            out[label] = int(df[col].nunique())
    if mapping.return_mode == "flag":
        col = mapping.source_for("returned")
        if col and col in df.columns:
            out["flagged_return_rate"] = round(float(_truthy(df[col]).mean()), 4)
    return out


def sample_head(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return df.head(n)


def with_file_currency(cfg: Config) -> Config:
    """Make the money layer use the uploaded file's currency, not the UCI one.

    `money.gbp_to_inr` is 105 because the reference dataset is a British wholesaler
    priced in pounds. Applying it to a file already denominated in rupees inflates every
    order 105-fold, which makes the cost of a missed return dwarf the cost of a false
    alarm and drives the optimal threshold to zero -- the model then "recommends"
    actioning literally every order, and does so with a straight face.

    So for a file run the FX rate is whatever `data.file.currency_to_inr` says, and its
    default is 1.0: amounts are assumed to already be in the merchant's own currency.
    """
    if cfg["data"].get("source") != "file":
        return cfg
    rate = float((cfg["data"].get("file") or {}).get("currency_to_inr", 1.0) or 1.0)
    payload = cfg.as_dict()
    payload["money"] = {**payload["money"], "gbp_to_inr": rate}
    return type(cfg)(_data=payload, path=cfg.path)


__all__ = [
    "CANONICAL_FIELDS",
    "FIELDS_BY_NAME",
    "READABLE_SUFFIXES",
    "REQUIRED_FIELDS",
    "SQLITE_SUFFIXES",
    "TABULAR_SUFFIXES",
    "ColumnMapping",
    "FieldSpec",
    "Issue",
    "list_tables",
    "profile",
    "read_any",
    "sample_head",
    "suggest_mapping",
    "to_canonical",
    "validate",
]
