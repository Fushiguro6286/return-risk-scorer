#!/usr/bin/env python
"""Generate `data/sample/merchant_export_sample.csv` -- the bring-your-own-data demo file.

    python scripts/make_sample_export.py

This is a **foreign-format** export on purpose. It shares no column names with the
canonical schema, records returns as a text status column rather than as cancellation
rows, uses email addresses as customer ids, and is priced in rupees. It exists to prove
the ingest front door works on a file that looks nothing like Online Retail II.

The return process is generated with structure the model can actually learn -- a
per-customer propensity, plus effects from order value, discount depth and basket size --
because a sample file with no signal would demonstrate the plumbing and nothing else.
It is still synthetic: no number produced from it means anything about the real world.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

N_ORDERS = 16_000
N_CUSTOMERS = 700
SEED = 11


def main() -> int:
    rng = np.random.default_rng(SEED)

    # Real categories, with genuinely different return behaviour -- apparel comes back,
    # consumables do not. The file carries the category explicitly, which is what most
    # merchant exports actually look like.
    families = {
        "Apparel": 1.00, "Footwear": 0.83, "Jewellery": 0.58, "Home Decor": 0.34,
        "Kitchenware": 0.22, "Stationery": 0.12, "Consumables": 0.01,
    }
    skus, sku_family, sku_risk, base_price = [], {}, {}, {}
    for fi, (fam, fam_risk) in enumerate(families.items()):
        for i in range(24):
            sku = f"SKU-{2000 + fi * 100 + i}"
            skus.append(sku)
            sku_family[sku] = fam
            sku_risk[sku] = float(np.clip(fam_risk + rng.normal(0, 0.05), 0.0, 1.1))
            base_price[sku] = float(rng.gamma(2.4, 190.0) + 120.0)

    family_names = list(families)
    family_weights = np.array([0.20, 0.16, 0.12, 0.16, 0.14, 0.12, 0.10])
    family_weights = family_weights / family_weights.sum()
    family_skus = {f: [s for s in skus if sku_family[s] == f] for f in family_names}

    cities = ["Mumbai"] * 4 + ["Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Kolkata"]
    # A customer-level propensity is the single most predictive real-world signal, and
    # it is what makes `prior_return_rate` worth having as a feature.
    cust_propensity = rng.beta(0.6, 4.5, size=N_CUSTOMERS)
    cust_city = rng.integers(0, len(cities), size=N_CUSTOMERS)

    start = pd.Timestamp("2023-01-02 09:00")
    rows: list[dict[str, object]] = []
    order_no = 700_000

    for _ in range(N_ORDERS):
        cust = int(rng.integers(0, N_CUSTOMERS))
        when = start + pd.Timedelta(hours=float(rng.uniform(0, 24 * 730)))
        order_no += 1
        n_lines = int(rng.integers(1, 7))
        # Baskets are category-coherent: people buy shoes with shoes, not one item from
        # each of six departments. Drawing SKUs uniformly across families would average
        # the basket's return risk back to the global mean and erase the signal.
        home = str(rng.choice(family_names, p=family_weights))
        pool = family_skus[home]
        picked = [str(c) for c in rng.choice(pool, size=min(n_lines, len(pool)), replace=False)]
        if rng.random() < 0.25 and n_lines > 1:  # occasional cross-category add-on
            picked[-1] = str(rng.choice(skus))
        n_lines = len(picked)

        # A deep discount is a real return driver: impulse buys come back.
        discount = float(np.clip(rng.beta(1.8, 5.0), 0, 0.7))

        lines = []
        order_value = 0.0
        for sku in picked:
            units = int(rng.integers(1, 10))
            rate = round(base_price[sku] * (1.0 - discount), 2)
            order_value += rate * units
            lines.append((sku, units, rate))

        # Return probability: customer propensity dominates, then product mix,
        # discount depth, basket size, and a mild high-value effect.
        p = (
            0.90 * cust_propensity[cust]
            + 1.60 * float(np.mean([sku_risk[s] for s in picked]))
            + 1.20 * discount
            + 0.030 * n_lines
            + 0.110 * float(np.clip(np.log1p(order_value) - 7.4, 0, 3))
        )
        p = float(np.clip(p * 0.22, 0.005, 0.92))
        returned = rng.random() < p
        ret_date = when + pd.Timedelta(days=float(rng.uniform(4, 55))) if returned else pd.NaT

        city = cities[int(cust_city[cust])]
        email = f"buyer{cust:04d}@example.com"
        for sku, units, rate in lines:
            rows.append(
                {
                    "Order Number": str(order_no),
                    "Placed On": when.strftime("%Y-%m-%d %H:%M:%S"),
                    "Item SKU": sku,
                    "Item Name": f"Product {sku}",
                    "Product Category": sku_family[sku],
                    "Units": units,
                    "Rate (INR)": rate,
                    "Buyer Email": email,
                    "Ship City": city,
                    "Order Status": "RETURNED" if returned else "DELIVERED",
                    "Returned On": "" if pd.isna(ret_date) else ret_date.strftime("%Y-%m-%d"),
                }
            )

    df = pd.DataFrame(rows).sort_values("Placed On", ignore_index=True)
    out = REPO_ROOT / "data" / "sample" / "merchant_export_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    orders = df.drop_duplicates("Order Number")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"  {len(df):,} lines  ·  {len(orders):,} orders  ·  {df['Buyer Email'].nunique():,} customers")
    print(f"  {orders['Placed On'].min()[:10]} -> {orders['Placed On'].max()[:10]}")
    print(f"  return rate {(orders['Order Status'] == 'RETURNED').mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
