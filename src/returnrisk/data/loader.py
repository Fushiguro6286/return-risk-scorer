"""Load line-level retail transactions.

Primary source is the real UCI *Online Retail II* dataset (UCI id 502): ~1.07M invoice
lines from a UK online giftware retailer, Dec-2009 to Dec-2011. The Excel workbook is
slow to parse (two sheets, ~40MB), so the first read is cached to Parquet.

The synthetic generator exists ONLY so the pipeline can be smoke-tested without the
download. It is never used for reported metrics.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, Config

CANONICAL_COLUMNS: Final[list[str]] = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
]

_RENAME: Final[dict[str, str]] = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "customer id": "customer_id",
    "country": "country",
}

# Non-product invoice lines: postage, bank charges, samples, manual adjustments.
# They are not orderable goods, so they must not create phantom "returns".
ADMIN_STOCK_CODES: Final[frozenset[str]] = frozenset(
    {"POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES", "C2", "CRUK", "PADS", "B", "GIFT"}
)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map the two spellings UCI ships (2009 vs 2010 sheet) onto one schema."""
    df = df.rename(columns={c: _RENAME.get(str(c).strip().lower(), str(c)) for c in df.columns})
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source is missing columns {missing}; got {list(df.columns)}")
    return df[CANONICAL_COLUMNS]


def download_uci(cfg: Config) -> Path:
    """Fetch the Online Retail II zip if it is not already on disk."""
    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "online_retail_II.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        return zip_path

    import urllib.request

    url = cfg["data"]["uci_url"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = resp.read()
    zip_path.write_bytes(payload)
    return zip_path


def _read_uci_workbook(zip_path: Path) -> pd.DataFrame:
    """Parse every data sheet out of the zipped workbook and stack them."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))]
        if not names:
            raise ValueError(f"no data file inside {zip_path}")
        name = names[0]
        blob = zf.read(name)
    if name.lower().endswith(".csv"):
        frames = [pd.read_csv(io.BytesIO(blob))]
    else:
        sheets = pd.read_excel(io.BytesIO(blob), sheet_name=None, engine="openpyxl")
        frames = list(sheets.values())
    df = pd.concat([_normalise(f) for f in frames], ignore_index=True)

    # The two year-sheets disagree on dtype: 2009 invoice numbers read as int64,
    # 2010 as str (cancellations carry a "C" prefix). Pin the schema before caching
    # or the Parquet writer rejects the mixed-type column.
    for col in ("invoice", "stock_code", "description", "country"):
        df[col] = df[col].astype("string")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    for col in ("quantity", "unit_price", "customer_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def load_uci(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Return the raw UCI line-level frame, using a Parquet cache after the first parse."""
    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "online_retail_II.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    zip_path = download_uci(cfg)
    df = _read_uci_workbook(zip_path)
    df.to_parquet(cache, index=False)
    return df


def make_synthetic(cfg: Config, n_orders: int = 4000) -> pd.DataFrame:
    """Small generated frame with the same schema, for pipeline smoke tests only.

    Deliberately crude: it exists to prove the code path runs end to end, not to
    produce numbers anyone should read.
    """
    rng = np.random.default_rng(cfg["seed"])
    n_cust = max(50, n_orders // 8)
    codes = [str(20000 + i) for i in range(120)]
    countries = ["United Kingdom"] * 6 + ["France", "Germany", "EIRE", "Spain"]
    start = pd.Timestamp("2010-01-04 08:00")
    returny = set(codes[:20])

    rows: list[dict[str, object]] = []
    cancels: list[dict[str, object]] = []
    invoice_no = 500000
    cancel_no = 900000

    for _ in range(n_orders):
        cust = int(rng.integers(0, n_cust))
        when = start + pd.Timedelta(hours=float(rng.uniform(0, 24 * 600)))
        country = str(rng.choice(countries))
        n_lines = int(rng.integers(1, 7))
        invoice_no += 1
        picked = [str(c) for c in rng.choice(codes, size=n_lines, replace=False)]
        lines = []
        for code in picked:
            line = {
                "invoice": str(invoice_no),
                "stock_code": code,
                "description": "ITEM " + code,
                "quantity": int(rng.integers(1, 12)),
                "invoice_date": when,
                "unit_price": float(np.round(rng.gamma(2.0, 1.8) + 0.4, 2)),
                "customer_id": float(cust),
                "country": country,
            }
            lines.append(line)
            rows.append(line)

        risk = 0.05 + 0.03 * n_lines + 0.30 * (len(set(picked) & returny) / n_lines)
        if rng.random() < min(risk, 0.6):
            cancel_no += 1
            victim = lines[int(rng.integers(0, len(lines)))]
            cancels.append(
                {
                    "invoice": "C" + str(cancel_no),
                    "stock_code": victim["stock_code"],
                    "description": victim["description"],
                    "quantity": -int(victim["quantity"]),
                    "invoice_date": when + pd.Timedelta(days=float(rng.uniform(1, 60))),
                    "unit_price": victim["unit_price"],
                    "customer_id": float(cust),
                    "country": country,
                }
            )

    return pd.concat([pd.DataFrame(rows), pd.DataFrame(cancels)], ignore_index=True).sort_values(
        "invoice_date", ignore_index=True
    )


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Type-coerce and drop rows that cannot support a checkout-time prediction."""
    df = df.copy()
    for col in ("invoice", "stock_code", "description", "country"):
        df[col] = df[col].astype("string").str.strip()
    df["stock_code"] = df["stock_code"].str.upper()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df.dropna(subset=["invoice", "stock_code", "invoice_date", "quantity", "unit_price"])

    if cfg["data"]["drop_missing_customer"]:
        # No customer id => no purchase history and no way to attribute a cancellation.
        df = df[df["customer_id"].notna()]
    df["customer_id"] = df["customer_id"].astype("float64").astype("int64").astype("string")

    df["is_cancellation"] = df["invoice"].str.upper().str.startswith("C").fillna(False)
    df = df[~df["stock_code"].isin(ADMIN_STOCK_CODES)]
    df = df[df["unit_price"] > 0]
    df = df[df["quantity"] != 0]

    # A cancellation line must have negative quantity; a purchase line positive.
    df = df[(df["is_cancellation"] & (df["quantity"] < 0)) | (~df["is_cancellation"] & (df["quantity"] > 0))]

    df["line_value"] = df["quantity"] * df["unit_price"]
    return df.sort_values(["invoice_date", "invoice"], ignore_index=True)


def load_file(cfg: Config) -> pd.DataFrame:
    """Read a user-supplied transaction file and shape it like the UCI frame.

    Everything downstream -- labelling, the embargo, the leakage guards -- then runs
    unchanged, which is the whole point: bringing your own data must not mean running a
    second, less-tested pipeline.
    """
    from .ingest import ColumnMapping, read_any, suggest_mapping, to_canonical

    spec = dict(cfg["data"].get("file") or {})
    path = spec.get("path")
    if not path:
        raise ValueError(
            "data.source is 'file' but data.file.path is not set. "
            "Point it at a CSV/Excel/Parquet/SQLite export, or pass --data-file."
        )
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p

    raw = read_any(p, table=spec.get("table"), sheet=spec.get("sheet"))

    mapping_spec = spec.get("mapping")
    if mapping_spec:
        mapping = ColumnMapping.from_dict(dict(mapping_spec))
    else:
        # No mapping committed to config -- fall back to the suggestion, but only if it
        # is unambiguous. A half-guessed mapping must fail loudly, not train silently.
        mapping, issues = suggest_mapping(raw)
        blocking = [i for i in issues if i.level == "error"]
        if blocking:
            detail = "; ".join(i.message for i in blocking)
            raise ValueError(
                f"could not infer a usable column mapping for {p.name}: {detail}. "
                f"Set data.file.mapping in config.yaml, or use the dashboard's "
                f"'Run on your data' tab to map the columns."
            )
    for key in ("return_mode", "assumed_return_lag_days", "currency_to_inr"):
        if key in spec and not mapping_spec:
            setattr(mapping, key, spec[key])

    df, _notes = to_canonical(raw, mapping)
    return df


def load_transactions(cfg: Config) -> pd.DataFrame:
    """Dispatch on `data.source` and apply the documented cleaning rules."""
    source = cfg["data"]["source"]
    if source == "uci":
        df = load_uci(cfg)
    elif source == "synthetic":
        df = make_synthetic(cfg)
    elif source == "file":
        df = load_file(cfg)
    else:
        raise ValueError(
            f"unknown data.source {source!r} (expected 'uci', 'synthetic' or 'file')"
        )
    return clean(df, cfg)
