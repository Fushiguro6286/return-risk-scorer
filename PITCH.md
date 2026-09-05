# 5-Minute Pitch — Return-Risk Scorer

Track 02 · AI Risk Manager · defense-only

**Rule for the recording: almost no time on model architecture.** Nobody is hiring for
"I used LightGBM". The argument is cost reasoning, honesty, and knowing how this breaks.

**Setup before recording**

```bash
python run_demo.py                  # artifacts fresh
uvicorn app.api:app --port 8000     # for the live /decide call
streamlit run app/dashboard.py      # leave open on tab 3
```

Have ready: `reports/money_confusion_matrix.png`, the dashboard (tabs 3 and 4), a
terminal with the `/decide` curl already typed, `MODEL_CARD.md`, `ARCHITECTURE.md`.

---

## [0:00–0:30] The problem — 30s

> "Returns are the biggest silent margin leak in Indian e-commerce. A returned order costs the
> reverse leg, the condition write-down, and the margin you already booked — and you find out
> weeks after you shipped it.
>
> I built a scorer that says, at checkout, which orders are likely to come back — and more
> usefully, **what to do about it and what that's worth in rupees.**
>
> Real data: UCI Online Retail II, 1.07 million invoice lines, 30,000 orders. 18.5% of them get
> reversed. Everything you're about to see is on a held-out **future** six months the model
> never saw."

*Do not say the model name. Do not open a notebook.*

---

## [0:30–1:30] The money confusion matrix — 60s ★ THE MONEY SHOT

**Full-screen `reports/money_confusion_matrix.png`.**

> "This is the whole project in one picture. It's a confusion matrix, but every cell is in
> **rupees**, not counts.
>
> Bottom-left — **₹32 lakh** — returns we missed. Top-right — **₹38 lakh** — friction we put on
> good customers who were never going to return anything. That top-right cell is the one most
> demos pretend is free. It isn't, and I price it.
>
> Bottom-right: 1,039 orders we caught. Those would have cost **₹2.55 crore** unflagged; after
> intervening they cost **₹1.40 crore**. That's where the money comes from.
>
> Bottom line: doing nothing costs **₹2.88 crore**. This policy costs **₹2.11 crore**.
> **₹76.6 lakh saved — 26.6% of the return bill.**
>
> And one number I want to be honest about up front: I assume an intervention only prevents
> **45%** of the returns it catches. It's not a cure. If I'd assumed 100% — which is the easy
> thing to do — that saving would look three times bigger and be fiction."

**Beat. Let the ₹76.6 L land.**

---

## [1:30–3:00] Per-action thresholds, live — 90s ★ THE DIFFERENTIATOR

**Switch to the dashboard, tab ③ "Policy simulator".**

> "Here's the part I actually care about. 'What's the threshold' is the wrong question — it
> depends entirely on **what you're going to do**.
>
> Three interventions, three completely different costs of being wrong:
> - A **manual pack check** — ₹40 of warehouse labour, the customer never knows.
> - **Withholding a promo** — some good buyers abandon the cart.
> - **Blocking cash-on-delivery** — maximum friction on someone who did nothing wrong.
>
> So each one gets its **own** optimal threshold."

**Show the table (README §5.1 or tab ③'s action dropdown):**

| Action | FP cost | t\* | Flagged |
|---|---|---|---|
| Pack check | ₹40 | 0.056 | 98.6% |
| Withhold promo | ₹1,066 | 0.122 | 54.8% |
| Block COD | ₹2,393 | 0.197 | 36.8% |

> "**Monotone**: the more a mistake costs, the higher the bar, the fewer orders you touch. That
> falls straight out of the algebra, and there's a unit test that fails the build if it ever
> stops holding."

**Now drag the slider. Slowly. This is the shot.**

> "Watch the rupees-saved number. At 0.05 — flag almost everything — **₹69.8 lakh**. Push it to
> 0.35 — very precise, 47% — and it collapses to **₹36.6 lakh**, because you're not catching
> enough.
>
> The peak is right here at **0.122: ₹76.57 lakh**." *(land on t\*)*
>
> "Left of it you buy recall you can't afford. Right of it you leave returns on the table.
> The merchant doesn't tune a threshold — they tell me what a false alarm costs them, and the
> threshold falls out."

*Make sure the ₹-saved metric visibly changes on camera. That is the acceptance criterion.*

---

## [3:00-4:00] Honest metrics + the weakness - 60s

> "Now the numbers, including the ones that don't flatter me.
>
> **AUC-PR 0.34** against a **17.6%** base rate — 1.94× lift. That is a **modest** number and I
> want to be straight about it: predicting returns at checkout is genuinely hard. Nothing in the
> data tells you a customer is about to change their mind.
>
> If I showed you 0.95 here, you should assume I leaked something. I ran a leakage audit on the
> SHAP importances specifically to check myself — top feature carries 13% of the explanation,
> nothing dominant, nothing post-checkout.
>
> Accuracy is 55%, and I'm showing you that on purpose: predicting 'no return' for everything
> scores **82%** accuracy and is worth exactly zero rupees. That's why AUC-PR is the headline.
>
> Calibrated with isotonic regression — Brier improves, calibration error is 1% — because the
> whole money layer is a probability times a cost. If the probability doesn't mean what it says,
> the rupees are fiction.
>
> **And here's where it's weakest.** On **new customers** the lift drops from 1.94× to **1.48×**.
> Which makes sense — the third-strongest feature is the customer's own return history, and a
> first-time buyer doesn't have one. So: **never COD-block a first-time buyer on this model's
> say-so.** That's written into the model card as a hard constraint."

---

## [4:00-4:40] The system refuses to act - 40s ★ THE GOVERNANCE BEAT

*Cut to the terminal. Run the `/decide` call for customer 17850 — the one the model
wants to block.*

> "Everything so far is what the model thinks. Here is what the system actually does.
>
> Same order. The model says 22% risk, above the threshold, block COD. And the system
> says **no**.
>
> *(point at `rules_fired`)*
>
> This customer has 155 prior orders at a 5% return rate. My model is a 25%-precision
> signal — three in four flagged orders were never coming back. It does not get to
> outrank a clean history that long. That's one of six guardrails, all declared in
> YAML: a value floor, this loyalty shield, a severity cap on new customers, a
> two-lakh limit above which a human decides, a daily cap on how much of the book we
> can action, and a 5% holdout that we deliberately let through so our retraining
> labels stay unbiased.
>
> Two of those exist because my own model card admits a weakness. New customers are my
> **worst** segment — 1.48× lift — and a COD block hurts them most. So the system
> cannot hit them with the harshest action. Writing that in a model card is free.
> Enforcing it in code costs money.
>
> *(switch to dashboard tab 4)*
>
> And every decision lands here. Append-only, each record hash-chained to the one
> before it — edit any past decision and this integrity check names the row. That's the
> difference between a model and something a merchant could actually deploy: when
> someone asks in March why you blocked an order in January, you can answer.
>
> There's one LLM in this system. It writes the merchant note and the dispute evidence
> pack — and it writes them **after** the decision is already final and recorded. It
> cannot decide anything, and if it states a rupee figure I did not give it, the output
> is thrown away and a template is used instead. Clone this repo with no API key and
> the whole demo still works."

**Beat to land:** *the model recommends, the policy decides, and the ledger remembers.*
Say the words "the system refused" out loud.

## [4:40-5:00] Selective labels + what's next - 20s ★ THE MATURITY BEAT

**Show `reports/selective_labels.png`.**

> "Last thing, and it's the one I'd want to be asked about.
>
> **The moment you switch this on, it starts poisoning its own training data.** Block COD on a
> risky order and the customer either prepays or walks — either way you **never find out** whether
> it would have been returned. The label is censored, and censored exactly on the orders the model
> was most confident about.
>
> I simulated it. Apply the policy for one period, retrain the way most teams would, then test on
> clean data:
>
> The observed return rate collapses from **17.7% to 3.9%**. The retrained model drops from
> **0.338 to 0.150 AUC-PR — it loses more than half its power.** And it doesn't crash. It quietly
> learns that its own high-risk signature is *safe*, stops flagging, returns climb, and the
> dashboard still looks fine because it's scored on the same broken data.
>
> **The fix:** let 5% of flagged orders through unactioned. Those are the only unbiased outcomes
> you have left, so you re-weight them by 1-over-p at retraining. Base rate recovers to **17.2%**,
> AUC-PR to **0.255**.
>
> It costs real money — about **₹3.6 lakh a window** in returns I'm deliberately eating. But I'd
> rather tell a merchant '₹3.6 lakh buys you a model that still works next year' than hand them
> something that silently degrades.
>
> **What I'd do next:** A/B test the effectiveness assumptions — 45% is my biggest guess and the
> only way to know is to run it. Then fix the small-order fairness gap: sub-£150 orders get flagged
> at 19% but only convert at 15%, so they absorb friction they don't earn. The per-order
> value-aware policy already in the repo closes that.
>
> Everything reproduces from a clean clone with one command. Thank you."

---

## Timing

| Segment | Budget | Cumulative |
|---|---|---|
| Problem | 0:30 | 0:30 |
| **Money confusion matrix** | 1:00 | 1:30 |
| **Per-action thresholds + live slider** | 1:30 | 3:00 |
| Honest metrics + weakness | 1:00 | 4:00 |
| **The system refuses to act (governance + ledger)** | 0:40 | 4:40 |
| Selective labels + next | 0:20 | 5:00 |

If you overrun, cut the selective-labels segment down to one sentence — the governance
beat is the one that separates this from a notebook, and it is the only place the demo
shows the system overruling its own model.

---

## If asked

**"Why is AUC-PR only 0.34?"**
Because that's what the problem is worth at checkout. 1.94× lift on an 18% base rate turns into
₹76.6 lakh, and the money layer is what converts a mediocre-looking ranking into a real decision.
A higher number here would make me suspect leakage in my own work.

**"Why does the pack check flag 98.6%?"**
Because at ₹40 against a ~₹15,000 expected return cost, the arithmetic genuinely says check
everything — this is a wholesaler averaging £465 an order. It's a real answer, and it's exactly
why I built the capacity-constrained mode: review the riskiest 1% and you get **67% precision**,
3.8× base rate. No warehouse can check 98.6% of a book.

**"Isn't 25% precision bad?"**
For a decision, yes — which is why nothing here denies anyone service. For prioritisation at
78.7% recall it's worth ₹76.6 lakh. The action set is deliberately reversible.

**"How do you know there's no leakage?"**
Three things: an allow-list with a test that plants a fake post-checkout feature and asserts the
guard fires; a row-by-row test that a customer's prior return only counts once the *cancellation
date* has passed, not the order date; and a test that deleting all future orders doesn't change
any past feature.

**"Where is the AI? This looks like gradient boosting."**
The detector is a calibrated LightGBM, deliberately — a decision that spends money has to be
reproducible and defensible, and an LLM in that seat would be neither. The LLM does the job it
is actually good at: it writes the merchant note and the dispute evidence pack over a decision
that is already final and already in the ledger. It cannot change one, and if it states a figure
I did not give it, the output is discarded and a template is used. That split is the design, not
a limitation.

**"What stops this from blocking a good customer?"**
Six guardrails, and the honest answer that nothing stops it entirely at 25% precision. The
loyalty shield protects long clean histories outright, new customers can only ever get the
gentlest action, anything over two lakh goes to a human, and there is a daily cap on how much
of the book we can touch. Every one of those is in YAML, and every one that fires writes a
sentence into the audit trail.

**"How would you know if this went wrong in production?"**
The ledger. `/audit/stats` reports the guardrail hit counts and the action rate; a drift event
shows up as the daily cap firing. And the 5% always-approve holdout means I still have unbiased
labels to measure against — that is the failure mode that kills these systems quietly, and it is
why the holdout runs in the serving path rather than living in a README.

**"Did you use the real data?"**
Yes — UCI Online Retail II, downloaded and parsed by `run_demo.py`. The synthetic generator
exists only as a smoke test so CI doesn't need a 45MB download, and it's never used for any
reported number.
