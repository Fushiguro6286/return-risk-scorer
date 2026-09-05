# Model Card — Return-Risk Scorer

**Version** 1.0 · trained 2026-08-31 · `models/return_risk_model.joblib`
**Owner** Razorpay AI Buildathon, Track 02 (AI Risk Manager) submission
**Status** Prototype. Not deployed. Not validated on any production book.

---

## 1. What it does

Given an e-commerce order **at checkout**, it outputs a calibrated probability that the order
will be returned within 90 days, a recommended intervention chosen to minimise expected rupee
cost, and up to three plain-language reasons.

**Intended use:** helping a merchant's risk/ops team decide *where to spend finite attention* —
which parcels to hand-check, which promos to withhold, which orders to require prepayment on.

**Intended users:** the merchant's own risk, ops and finance teams. Not the end customer.

---

## 2. The label, stated exactly

> `returned = 1` if a later **cancellation invoice from the same customer** reverses at least one
> stock code of the order, within **90 days** of the order.

This is a proxy, and the gap matters:

- Online Retail II records **cancellations**, not physical returns. An order cancelled before
  dispatch and a parcel sent back after delivery are indistinguishable in this data. The model
  therefore predicts "this order gets reversed", which is close to but not identical to
  "the goods come back".
- Linkage is heuristic (most-recent-prior-purchase, same customer, same stock code, quantity
  consumed greedily). **17.9% of cancellation lines could not be linked** to an originating
  order and were dropped — they are not counted as returns anywhere.
- Orders in the final 90 days of the file were **dropped as right-censored** (6,573 of 36,620),
  because their outcome window runs past the end of the data.

---

## 3. Training data

| | |
|---|---|
| Source | **UCI Online Retail II** (id 502), Chen (2019), CC BY 4.0 |
| Domain | A single UK-based online **wholesaler** of giftware |
| Raw | 1,067,371 invoice lines, 2009-12-01 → 2011-12-09 |
| After cleaning | 820,645 lines → **30,047 orders**, 5,265 customers |
| Order value | mean £465, median £304 — **wholesale scale, not retail baskets** |
| Base rate | **18.53%** returned |
| Train window | 2009-12-01 → 2010-11-01 (15,694 orders) |
| Calibration | 2010-11-01 → 2010-12-23 (3,923 orders) |
| **Held-out test** | **2011-03-24 → 2011-09-09 (7,511 orders)** |

Rows without a customer ID are dropped (no history, no cancellation attribution). Non-product
lines (postage, bank charges, manual adjustments, samples) are excluded.

---

## 4. Model

LightGBM (400 trees, lr 0.05, 31 leaves, `min_child_samples=100`), 27 checkout-time features,
**isotonic calibration** fitted on the held-out calibration slice via `FrozenEstimator`.

**No SMOTE, no oversampling, no class weights.** They distort the predicted base rate, and the
entire cost layer is a probability multiplied by a rupee figure.

---

## 5. Metrics — held-out temporal test set

**Always read against the base rate: 17.57%.**

| Metric | Value | Note |
|---|---|---|
| **AUC-PR (headline)** | **0.341** | **1.94×** the 0.176 base-rate floor |
| ROC-AUC | 0.705 | secondary |
| Brier | 0.1318 | 0.1322 → 0.1288 on the calib slice, pre/post isotonic |
| Expected calibration error | 0.0104 | |
| Precision @ t\*=0.122 | **0.252** | |
| Recall @ t\*=0.122 | **0.787** | |
| F1 @ t\* | 0.382 | |
| Accuracy | 0.553 | **meaningless here** — always-predict-no scores 0.824 |
| Flag rate @ t\* | 54.8% | |

**Money (test set, "withhold promo" action):** ₹2.88 Cr → ₹2.11 Cr, **₹76.6 L saved (26.6%)**.
Profitable across **36/36** cells of the cost-sensitivity grid.

### Per-segment (where it is weakest)

| Segment | n | Base rate | AUC-PR | Lift | Precision |
|---|---|---|---|---|---|
| Returning customers | 6,910 | 17.8% | 0.350 | 1.96× | 25.8% |
| **New customers** | **601** | **14.8%** | **0.219** | **1.48×** | **19.2%** |
| Orders <£150 | 1,853 | 7.9% | 0.124 | 1.57× | 15.4% |
| Orders £1k+ | 472 | 29.9% | 0.562 | 1.88× | 36.3% |
| UK | 6,777 | 17.5% | 0.342 | 1.95× | 25.7% |
| EU | 696 | 18.2% | 0.364 | 2.00× | 22.3% |

---

## 6. Known limitations

1. **Precision is 25%.** Three of every four flagged orders would not have been returned. This
   is a prioritisation signal, not a verdict.
2. **Weakest on new customers** (1.48× lift) — precisely the group a COD block hurts most, and
   the group with no prior-return history for the model to use.
3. **Weak on small orders** (<£150: 1.57× lift, 15.4% precision) and they absorb disproportionate
   friction (flagged 19.2%, converting at 15.4% vs 36.3% for £1k+ orders).
4. **One merchant, one country, one era.** 2009–2011, UK giftware wholesale. Seasonality
   (`days_to_christmas` is a top-10 feature) is specific to this book.
5. **Currency is an assumption.** GBP→INR at a **fixed ₹105**, and margin at **28%**. Both live
   in `config.yaml`; neither was estimated from data.
6. **Intervention effectiveness is assumed, not measured** (25–65% by action). These are the
   single largest source of uncertainty in every rupee figure and can only be established by a
   live A/B test. The sensitivity sweep is there because of this.
7. **The label is cancellations, not physical returns** (§2).
8. **Selective labels** (§7) — the failure mode that will actually bite.
9. **No adversarial robustness testing.** If customers learned the rule, discount-seeking
   behaviour would shift. Not modelled.
10. **Stability is evidenced over six months only.** No claim beyond that.

---

## 7. The selective-labels caveat

**Acting on this model degrades the data used to retrain it.** Intervene on a high-risk order
and the counterfactual is never observed. Measured on this data:

| | Observed base rate | AUC-PR on clean data |
|---|---|---|
| Truth | 17.74% | 0.338 |
| **Naive retrain on censored labels** | **3.94%** | **0.150** |
| Retrain w/ 5% always-approve holdout + IPW | 17.21% | 0.255 |

**A naive quarterly retrain loses 56% of the model's discriminative power** and does so silently —
the offline metrics look fine because they are computed on the same censored data.

**Mandatory mitigation:** run the **always-approve holdout** (default 5% of flagged orders let
through unactioned, up-weighted 1/p at retraining). It costs ~₹3.55 L per test window in returns
deliberately absorbed. **This is not optional. Do not run this model in production without it.**

---

## 8. Do not use this for

- ❌ **As the sole reason to deny anyone service, refuse an order, or close an account.** At 25%
  precision that would wrongly penalise three customers for every one return prevented. Actions
  must stay reversible and proportionate: a pack check, a withheld promo, a prepayment request.
- ❌ **Any decision about a person rather than an order** — creditworthiness, employment,
  insurance, or a "customer risk score" that follows someone around.
- ❌ **Automated irreversible action without a human in the loop** on high-value orders.
- ❌ **A different merchant, category or country** without refitting and re-running the full
  evaluation. The method transfers; these coefficients do not.
- ❌ **Any offensive purpose.** This is a defensive tool. It does not generate fraud, evade
  detection, or profile individuals.

---

## 9. Fairness

Checked flag rate and precision by region, customer type and order value. No protected
attribute (age, gender, ethnicity, religion) is available in this dataset or used as a feature.
Country is used — legitimate for logistics, but it means **flag rates differ by destination**
(UK 52.9%, EU 72.3%) and this should be monitored if deployed.

**The one disparity found:** small orders absorb more friction per prevented return (20.8-point
precision gap vs £1k+ orders). Mitigation available and implemented — the per-order value-aware
policy (`money.CostModel.value_aware_flags`).

---

## 10. Monitoring, if this were deployed

| Watch | Trip wire |
|---|---|
| Monthly AUC-PR on the always-approve holdout | drop >20% from 0.34 |
| Observed vs predicted return rate (calibration) | ECE > 0.03 |
| Flag rate | ±10pp drift from 54.8% |
| Flag rate **by segment** | any segment's over-flagging ratio moving >25% |
| Realised ₹ saved vs forecast | any negative month |
| Holdout size | never below 5% of flagged volume |

Retrain quarterly, always on holdout-corrected data, always re-running the leakage tests.

---

## 11. Reproducibility

```bash
pip install -r requirements.txt
python run_demo.py     # fixed seed 42; regenerates every number in this card
pytest tests/ -q       # 232 tests
```

Every constant lives in `config.yaml`. Data: Chen, D. (2019), *Online Retail II*, UCI ML
Repository, <https://doi.org/10.24432/C5CG6D>, CC BY 4.0.
