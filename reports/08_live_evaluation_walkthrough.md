# Live Evaluation Walkthrough

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail
Author: Eashwaradhinesh K

A defence document for the live evaluation. Covers the project narrative,
every design decision and why it was made, the failures found and fixed, and
the questions an evaluator is most likely to ask.

---

## Part 1 — The 60-second summary

> I built a recommendation system for an e-commerce platform that has no real
> interaction data yet — the early-stage scenario in the brief. I generated
> 6,000 customers, a 2,500-product catalogue and 138,000 interactions, but
> deliberately engineered the properties that make recommendation hard: 99%
> sparsity, Pareto-distributed demand, a long tail, and reserved cold-start
> cohorts.
>
> I built four baselines, a TF-IDF content recommender, a PyTorch NCF model, and
> a hybrid that fuses them. I benchmarked all six on a time-based split with
> no-leakage assertions. The hybrid won at NDCG@10 = 0.1095, 69.9% above the
> popularity baseline.
>
> The most interesting part wasn't the winning number. The benchmark caught two
> real failures. The deep model initially scored exactly zero on every ranking
> metric, and the hybrid was losing to one of its own inputs. Both are fixed and
> both are documented.

---

## Part 2 — Why the project looks like this

### Q: Why e-commerce and not movies?

The brief specifies the domain as *"E-Commerce / OTT Platforms / Online Learning
/ Retail / Media Streaming"* and mandates an item schema with `price` and an
interaction schema with `purchase`. Those two fields only make natural sense in
a commerce setting. The mandated fields drove the domain choice.

### Q: Walk me through the pipeline.

```
generate_synthetic_data.py     users, items, interactions
        ↓
preprocessing.py               rating / normalised / implicit matrices,
                               popularity, content and profile features
        ↓
content_based_nlp.py           TF-IDF + content similarity
baseline_recommenders.py       popularity, user-CF, item-CF, SVD
collaborative_filtering.py     re-ranking + cold-start logic
ncf_recommender.py             PyTorch NCF (explicit + implicit)
        ↓
hybrid_recommender.py          four-signal fusion  ← the production model
explainability.py              three explanation types
        ↓
app_fastapi.py  /  streamlit_recommendation_dashboard.py
```

---

## Part 3 — Synthetic data (15 marks)

### Q: Why not just generate random data?

Because a recommender trained on uniformly random data learns nothing, and an
EDA over random data cannot demonstrate any of the properties the brief asks us
to handle. The generator has to *engineer* the pathologies.

### Q: What did you engineer, and how?

| Property | Mechanism | Result |
|---|---|---|
| Popularity bias | shifted Zipf: `1/(rank+25)^1.25` | top 10% hold 62.9% |
| Long tail | falls out of the Zipf law | Gini 0.716 |
| Sparsity | limited interactions per user | 99.08% |
| Cold start | reserved 8% users / 10% items | 480 / 255 |
| User heterogeneity | lognormal activity draw | median 15, max 201 |
| Latent structure | category affinity × segment price band | 60.7% in-category |

### Q: Why the `+25` offset in the Zipf law?

**This is the best single question to be asked, because the answer shows real
debugging.**

With a pure `1/rank^alpha` law, the top item's *expected* exposure exceeded the
entire user base — 327% of it. But a customer can only interact with an item
once, so the head **saturated**: the top items ended up seen by nearly every
customer, which destroys their discriminative value entirely. Measured, the top
item drew 4,299 interactions across 6,000 customers.

The offset flattens the very top while leaving the tail untouched. After the
fix the top item sits at 32% of the user base and the Pareto shape is intact.

### Q: How do you know the data is realistic?

Ten checks in `notebooks/eda_validation.py`, all passing. The one I'd point to
is the rating distribution: real marketplace ratings are **J-shaped**, mean near
4, dominated by 5-star reviews. My first calibration produced a symmetric
distribution centred on 3.10 — the classic give-away of synthetic data. I
retuned the intercept and slope to get mean 3.91 with 67% at 4–5 stars.

### Q: Anything deliberately unrealistic?

Yes, and it's documented. Conversion is 16.4% of clicks against a real-world
2–3%. At a realistic rate, 138k interactions yield only ~4,000 purchases across
6,000 customers — far too sparse to train an implicit model or build a test set
with enough relevant items per customer for stable metrics. It's a documented
trade-off between behavioural realism and having enough signal to model.

---

## Part 4 — Deep learning (20 marks)

### Q: Why a neural model when SVD already factorises the same matrix?

Matrix factorisation scores an item as a dot product of user and item vectors —
a fixed, linear interaction. The dimensions can only combine additively.

My generator embedded a **non-linear** rule: a customer buys when the category
matches **and** the price sits in their segment's band. A dot product cannot
represent a conjunction. An MLP over concatenated embeddings can.

### Q: Why concatenate rather than multiply the embeddings?

An element-wise product hard-codes a multiplicative interaction and reduces the
model back to matrix factorisation. Concatenation leaves the interaction
function itself to be learned by the MLP. That's the entire architectural point
of NCF.

### Q: Your NCF scored NDCG of 0.0000. Explain that.

**Be ready for this — it's in the reports and an evaluator will find it.**

The first benchmark returned exactly zero on every ranking metric, with
catalogue coverage of 0.006 — the same 14 items to every customer.

It was a framing error, not a tuning problem. The explicit model minimises MSE
against observed star ratings. It's a *good rating predictor* — RMSE 0.911 — but
it has never once been shown an item a customer did **not** interact with. So
scoring the unseen catalogue is completely out of distribution, and it falls
back on the global item bias, which is identical for everybody.

The fix was **negative sampling**: 4 negatives per positive, drawn from items
the customer never touched. Now discriminating "would engage" from "would not"
*is* the training objective.

| | Accuracy | F1 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Explicit | n/a | n/a | 0.0000 | 0.006 |
| Implicit + negatives | 0.858 | 0.582 | 0.0576 | 0.013 |

The lesson: **a model can be accurate on the metric you trained it on and
worthless at the task you actually need.**

### Q: Why exclude browsed-but-not-carted items from the negatives?

They're ambiguous, not confirmed negatives. Labelling them 0 would teach the
model that browsing implies disinterest, which is false — a customer may have
been interrupted, or may buy later.

### Q: Your implicit model peaks at epoch 19 but accuracy is only 0.858. Is it
### under-trained?

No — it's capacity-bound. I ran a regularisation sweep across embedding
dimension, weight decay and dropout; every configuration landed at ~0.858
accuracy. What changed was *when*: with the defaults it peaked at epoch 1 and
overfitted immediately. Weight decay 1e-3 with dropout 0.40 reaches the same
ceiling over 20 epochs, which gives a genuine convergence curve and a meaningful
early-stopping signal.

---

## Part 5 — Hybrid logic (15 marks)

### Q: How did you choose the fusion weights?

By measurement, after the first version failed.

The original blend — SVD 0.35 / content 0.25 / NCF 0.40 — scored NDCG@10 =
0.0947, **worse than plain item-based CF at 0.1245**. The hybrid was losing to
one of its own potential inputs, because the collaborative slot held SVD (the
weakest signal) while item-based CF wasn't in the fusion at all.

| Configuration | NDCG@10 |
|---|---|
| item-based CF alone | 0.1245 |
| svd + content + ncf (original) | 0.0947 |
| item_cf + content + ncf | 0.1282 |
| **item_cf .45 / svd .10 / content .10 / ncf .35** | **0.1293** |

### Q: Why keep content at only 10% when it scores 0.0116 alone?

Coverage. Content-based has **75.4% catalogue coverage** against item-CF's
22.2%. It's the only signal that reaches the long tail at all. Dropping it costs
NDCG — the sweep shows `item_cf .70 / ncf .30` with no content scores 0.1181,
worse than the blend that keeps it.

### Q: Why keep SVD when item-CF is better?

It generalises to customers whose exact co-rating neighbours are absent. Item-CF
needs overlap in specific items; SVD works in latent space. Removing it also
costs NDCG in the sweep.

### Q: What is "retrieve then re-rank"?

The diversity correction applies only to the top 200 candidates by relevance,
not across the whole catalogue.

Applied catalogue-wide, a near-zero-exposure item the customer has no affinity
for outranks a genuinely relevant one purely for being obscure. I measured this:
with the correction applied catalogue-wide, top-10 mean exposure dropped from
518 interactions to **1**, and long-tail share went to **100%**. The system was
recommending obscurity for its own sake.

Restricting the pool means every item in the final list was relevant first and
diverse second.

---

## Part 6 — Evaluation (15 marks)

### Q: How do you guarantee no leakage?

Two rules, both **asserted in code**:

```python
assert test_df["timestamp"].min() > cutoff
assert train_df["timestamp"].max() <= cutoff
```

And critically — every model is **refitted from scratch on the training
period**. Reusing the artifacts in `data/processed/` would have been much
quicker, but those were fitted on the full history including the test window, so
every metric would be inflated.

The candidate set also excludes everything the customer touched during training.
Without that, a model scores points for "predicting" purchases it was shown.

### Q: Why is relevance "rated ≥ 4 OR purchased"?

A bought-but-unrated item is unambiguous evidence of relevance. Only about 68%
of interactions carry a rating, so using ratings alone would discard a large
share of genuine positives and understate every model's recall.

### Q: Why report coverage and novelty as well as NDCG?

Because a model can post excellent precision while only ever recommending 32
items out of 2,500. Commercially that's a failure — the rest of the inventory is
invisible and will never sell. The popularity baseline has 1.3% coverage: it
returns the same list to everybody, and accuracy metrics alone can't see that.

### Q: Classical CF beat your deep model. Isn't that a bad result?

It's an honest one, and I'd rather report it than hide it. Item-based CF scores
0.1044 against NCF's 0.0663, at 1/60th the inference cost.

That's unsurprising at this scale — 138k interactions across 6,000 customers
isn't enough signal to justify 278,000 parameters. NCF's advantage emerges with
much larger and richer data.

The deep model still earns its place: it carries 0.35 of the hybrid weight, and
the hybrid beats item-CF alone. But recommending NCF *instead of* item-CF
wouldn't be supported by this evidence, and my business recommendations say so.

---

## Part 7 — Explainability (10 marks)

### Q: How do the explanations work?

Three types, all **derived from the actual scores that produced the ranking**:

1. **Why this item** — signal attribution reporting weight × score, not raw
   score, because a signal with a high score but a low weight contributes little
   to the decision.
2. **Evidence** — the customer's own comparable purchases, plus similar-customer
   statistics.
3. **Content justification** — the specific matching TF-IDF terms, which is what
   turns "similarity 0.83" into something a merchandiser can audit.

### Q: Why not just write a nice template sentence?

Because a plausible-sounding explanation that doesn't correspond to the real
reason is worse than none. It manufactures false confidence and will eventually
contradict the model in front of a customer.

### Q: Your similar-customer evidence is often empty. Is that broken?

No — it's structural, and I chose to state it rather than hide it. The re-ranker
deliberately promotes long-tail items, and an item is long-tail precisely
because few people have interacted with it. **The diversity mechanism erodes the
collaborative evidence available to justify its own output.**

So the system says exactly that:

> "None of your 50 closest customer matches have rated this item — it is a
> low-exposure product surfaced for catalogue variety, so the recommendation
> rests on content and model signals rather than crowd evidence."

Telling a customer "customers like you loved this" about an item no comparable
customer has ever rated would be a straightforward misrepresentation.

---

## Part 8 — Application layer (15 marks)

### Q: Why build both FastAPI and Streamlit?

The brief requires one. FastAPI proves production thinking — validation, typed
contracts, batch endpoint, health checks. Streamlit demos live and makes the
explainability visible, which is hard to convey through curl.

Critically, **both call the same `HybridRecommender` class.** Duplicating fusion
logic into each application is how they drift apart and start returning
different answers for the same customer.

### Q: What was wrong with your previous API?

Two real bugs, both now fixed:

1. It looked for NCF artifacts in `app/saved_models/` — a directory that never
   existed — wrapped in a bare `try/except`. The service started, reported
   healthy, and **served every recommendation with the deep-learning signal
   silently switched off.**
2. It called `torch.load()` and then `.eval()` on the result, but the file was a
   `state_dict`, not a model object.

Now: missing artifacts produce a loud warning naming the exact file and fix
command, `/health` reports `ncf_available` explicitly, and weights are
renormalised over the signals that actually produced scores.

### Q: Why does every response carry a `strategy` field?

So a caller can always distinguish a personalised result from a fallback.
Presenting them identically is how a dashboard ends up claiming a brand-new
customer has a learned taste profile.

---

## Part 9 — Likely challenges

### "Your metrics are low. NDCG@10 of 0.11 isn't very good."

Absolute ranking metrics on sparse implicit data are always low — a customer has
~4.4 relevant items in the test period out of a 2,500-item catalogue, so
Precision@10 is capped around 0.44 even for a perfect model. What matters is the
**relative** comparison on identical data and splits: the hybrid is 69.9% above
the popularity baseline.

### "This is all made-up data. What does it prove?"

It proves the pipeline, the methodology and the architecture. It does **not**
prove real-world performance, and I say so in every report. The brief explicitly
puts us in a scenario where real data is unavailable — that's the premise, not a
shortcut. What transfers is the harness; what doesn't is the numbers.

### "Why should I believe your evaluation isn't leaking?"

Because the assertions are in the code and will crash the run if violated, and
because I refit every model on train-only data rather than reusing the stored
artifacts. That choice makes the benchmark slower and the numbers lower — which
is the direction an honest fix moves them.

### "What would you do differently with more time?"

Add online exploration to break the popularity feedback loop — the current
inverse-propensity penalty dampens it but can't break it, because the system
never explores. And build the ingestion-quality layer: bot traffic in particular
directly poisons the popularity model and the item-item similarity matrix.

---

## Part 10 — Numbers to have ready

| | |
|---|---|
| Users / items / interactions | 6,000 / 2,500 / 138,345 |
| Sparsity | 99.08% |
| Top 10% of items' share of demand | 62.9% |
| Gini | 0.716 |
| Cold-start users / items | 480 (8.0%) / 255 (10.2%) |
| Rating mean | 3.91 |
| Train / test split | 96,656 / 41,689 (90-day holdout) |
| **Hybrid NDCG@10** | **0.1095** |
| Item-CF NDCG@10 | 0.1044 |
| NCF NDCG@10 | 0.0663 |
| Popularity NDCG@10 | 0.0645 |
| **Lift over baseline** | **+69.9%** |
| NCF RMSE (explicit) | 0.9108 |
| NCF accuracy (implicit) | 0.8583 |
| NCF parameters | 278,817 |
| Hybrid weights | item_cf .45 / svd .10 / content .10 / ncf .35 |
| Hybrid coverage | 20.2% |
| Content coverage | 75.4% |

---

## Part 11 — Demo script (5 minutes)

1. **Data** — run `python notebooks/eda_validation.py`, show the ten passing
   checks and the three mandatory plots.
2. **Streamlit** — pick a highly active customer, show the recommendations and
   the category mix.
3. **Explainability tab** — show the signal attribution bar chart and the
   evidence from the customer's own purchases.
4. **Cold-start tab** — switch to a new customer; point out the price points
   match their segment, and the explanation is visibly different and honest.
5. **New product** — pick a cold-start item with zero sales, show content
   neighbours working with no collaborative signal at all.
6. **FastAPI** — open `/docs`, hit `/recommend/3181`, show the `strategy` field
   and the signal breakdown in the JSON.
7. **Close on the benchmark table** — the hybrid winning, and the two failures
   the harness caught.
