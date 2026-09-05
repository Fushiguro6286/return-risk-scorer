#!/usr/bin/env python
"""One command that regenerates every artifact in reports/ and the served model.

    python run_demo.py                      # real UCI data, full run
    python run_demo.py --quick              # smaller SHAP sample, everything else identical
    python run_demo.py --synthetic          # pipeline smoke test, no download required
    python run_demo.py --data-file mine.csv # YOUR data, columns inferred
    python run_demo.py --config other.yaml

Fixed seeds throughout, so two runs on the same data produce identical charts.

Bringing your own data
----------------------
`--data-file` accepts CSV, TSV, Excel, Parquet, JSON/JSONL, SQLite (.db/.sqlite) or a
zip wrapping any of those, holding **line-level** transactions. Columns are inferred
from their names and values, and the inferred mapping is printed before training so you
can check it. If a required column cannot be identified the run stops rather than
guessing -- a model trained on the wrong date column looks fine and is worthless.

Use `--return-mode flag` when returns are recorded as a `returned`/`is_return` column
rather than as reversing rows with negative quantity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from returnrisk.config import load_config  # noqa: E402
from returnrisk.pipeline import run  # noqa: E402


def _patched(cfg, data_overrides: dict):
    """Return a copy of the config with `data` keys replaced."""
    payload = cfg.as_dict()
    payload["data"] = {**payload["data"], **data_overrides}
    return type(cfg)(_data=payload, path=cfg.path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--quick", action="store_true", help="shrink SHAP sampling (metrics unchanged)")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="use the generated smoke-test dataset instead of real UCI data",
    )
    ap.add_argument(
        "--data-file",
        default=None,
        metavar="PATH",
        help="run on your own transaction export (csv/xlsx/parquet/json/sqlite/zip)",
    )
    ap.add_argument("--table", default=None, help="table (SQLite), sheet (Excel) or zip member")
    ap.add_argument(
        "--return-mode",
        choices=["cancellation", "flag"],
        default=None,
        help="how returns are recorded in --data-file (default: inferred)",
    )
    ap.add_argument(
        "--return-lag-days",
        type=float,
        default=None,
        help="flag mode with no return-date column: assumed days until a return is known",
    )
    args = ap.parse_args()

    if args.synthetic and args.data_file:
        ap.error("--synthetic and --data-file are mutually exclusive")

    cfg = load_config(args.config)

    if args.synthetic:
        cfg = _patched(cfg, {"source": "synthetic"})
        print("!! synthetic mode: numbers below are a pipeline smoke test, not results\n")

    elif args.data_file:
        cfg = _describe_and_patch(cfg, args)
        if cfg is None:
            return 2

    if cfg["data"].get("source") == "file":
        from returnrisk.data.ingest import with_file_currency

        cfg = with_file_currency(cfg)
        rate = float(cfg["money"]["gbp_to_inr"])
        print(
            f"Money layer: amounts treated as {'already in rupees' if rate == 1.0 else f'x{rate} to the rupee'}"
            f" (data.file.currency_to_inr).\n"
        )

    run(cfg, quick=args.quick)
    return 0


def _describe_and_patch(cfg, args):
    """Read the user's file, show the inferred mapping, and stop if it is not usable.

    Printing the mapping is not decoration. The failure this guards against -- a column
    mapped to the wrong field, producing a model that trains happily and means nothing --
    is silent by nature, so the mapping goes on screen before any training happens.
    """
    from returnrisk.data.ingest import profile, read_any, suggest_mapping, validate

    path = Path(args.data_file)
    if not path.is_absolute():
        path = Path.cwd() / path

    print(f"Reading {path} ...")
    try:
        raw = read_any(path, table=args.table)
    except Exception as exc:
        print(f"\n!! could not read the file: {exc}", file=sys.stderr)
        return None

    mapping, issues = suggest_mapping(raw)
    if args.return_mode:
        mapping.return_mode = args.return_mode
    if args.return_lag_days is not None:
        mapping.assumed_return_lag_days = args.return_lag_days

    info = profile(raw, mapping)
    print(f"  {info['n_rows']:,} rows x {info['n_columns']} columns")
    if "date_min" in info:
        print(f"  {info['date_min']} -> {info['date_max']}  ({info.get('span_days', 0)} days)")
    if "n_orders" in info:
        print(
            f"  {info.get('n_orders', 0):,} orders  ·  "
            f"{info.get('n_customers', 0):,} customers  ·  "
            f"{info.get('n_products', 0):,} products"
        )

    print("\nColumn mapping (inferred):")
    for canonical, source in mapping.columns.items():
        marker = " " if source else "-"
        print(f"  {marker} {canonical:<16} <- {source or '(not found)'}")
    print(f"\n  return mode: {mapping.return_mode}")

    issues = issues + validate(raw, mapping)
    errors = [i for i in issues if i.level == "error"]
    for issue in issues:
        if issue.level == "info":
            continue
        print(f"  [{issue.level}] {issue.message}")
        if issue.fix:
            print(f"           -> {issue.fix}")

    if errors:
        print(
            "\n!! cannot run on this file. Fix the errors above, or map the columns "
            "explicitly in config.yaml under data.file.mapping.",
            file=sys.stderr,
        )
        return None

    print()
    return _patched(
        cfg,
        {
            "source": "file",
            "file": {
                **(cfg["data"].get("file") or {}),
                "path": str(path),
                "table": args.table,
                "mapping": mapping.to_dict(),
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
