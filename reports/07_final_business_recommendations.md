# Final Business Recommendations

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## Executive summary

We were asked to design and validate a recommendation system for a digital
platform that does not yet have real interaction data. We built the full
pipeline on engineered synthetic data — 6,000 customers, a 2,500-product
catalogue, 138,345 interactions — and benchmarked six recommendation approaches
under a leak-free time-based split.

**The hybrid model is the best performer: NDCG@10 of 0.1095, a 69.9% lift over
the popularity baseline.**

Three findings matter more than that headline number:

1. **Classical item-based collaborative filtering beats the deep model on its
   own** (0.1044 vs 0.0663) at 1/60th the inference cost. The neural network
   earns its place inside the hybrid, not as a replacement.
2. **The deep model was initially worthless for ranking** — NDCG of exactly
   0.0000 — because it was trained to predict ratings rather than to
   discriminate across the catalogue. Fixing the training objective, not the
   architecture, is what made it usable.
3. **Every number here comes from synthetic data.** The transferable deliverable
   is the pipeline and the evaluation methodology, not the metric values.

---

## 1. What was built

| Layer | Delivered |
|---|---|
| Synthetic data | 6,000 users / 2,500 items / 138,345 interactions with engineered sparsity, popularity bias, long tail and cold-start cohorts |
| Baselines | Popularity, User-CF, Item-CF, SVD |
| NLP | TF-IDF content recommender over product descriptions and tags |
| Deep learning | PyTorch NCF, explicit and implicit variants |
| Hybrid | Four-signal weighted fusion with retrieve-then-rerank |
| Explainability | Three mandated explanation types, all score-derived |
| Applications | FastAPI service (7 endpoints) and Streamlit operator console |
| Evaluation | Precision/Recall/MAP/NDCG @ 5/10/20 plus coverage, novelty, diversity |

---

## 2. Model performance

| Model | NDCG@10 | Coverage | Inference/request |
|---|---|---|---|
| **Hybrid** | **0.1095** | 20.2% | 208 ms |
| Item-based CF | 0.1044 | 22.2% | 3.3 ms |
| NCF (implicit) | 0.0663 | 1.3% | 192 ms |
| Popularity | 0.0645 | 1.3% | 0.2 ms |
| SVD | 0.0462 | 9.6% | 1.4 ms |
| Content (TF-IDF) | 0.0116 | 75.4% | 4.4 ms |

### The cost/benefit question the business actually faces

The hybrid buys **+4.9% NDCG over item-based CF for 63× the inference cost.**

That trade is worth making on a batch-served surface, where the compute happens
overnight and latency is irrelevant. It is much harder to justify on a
real-time, high-QPS surface.

---

## 3. Recommendations

### R1 — Ship item-based CF first, add the hybrid second
**Priority: HIGH**

Item-based CF delivers 95% of the hybrid's accuracy at 1.6% of the cost, trains
in three seconds, and produces explanations customers find credible ("similar to
what you bought"). It is the fastest path to a working recommender.

Layer the hybrid on afterwards, on batch-served surfaces first.

### R2 — Serve batch-first
**Priority: HIGH**

Precompute top-N nightly into a key-value store. Around 80% of surfaces —
homepage rails, email, push, product-page "similar items" — tolerate daily-stale
recommendations, and serving becomes a single-digit-millisecond lookup.

Reserve real-time scoring for post-add-to-cart suggestions and search re-ranking,
where in-session behaviour genuinely changes the answer.

**Impact:** roughly 95% reduction in serving compute.

### R3 — Instrument before optimising
**Priority: CRITICAL**

Every metric in this project is an offline proxy. Ranking metrics measure
agreement with historical behaviour, which is itself a product of whatever
ranked items in the past.

Before tuning anything further, instrument:
- CTR on recommended items, by surface
- Conversion rate from recommendation to purchase
- Revenue per recommendation slot
- Share of revenue from long-tail items

**Without this telemetry, no amount of offline NDCG tells you whether the system
is earning its cost.** This is the highest-value action on the list and it
requires no modelling work at all.

### R4 — Close the two blocking risks before production
**Priority: CRITICAL**

Two risks in the data risk assessment remain HIGH and neither is a modelling
problem:

- **Ingestion data quality (R6).** Real logs carry duplicate events, bot
  traffic, mixed timezones and orphaned item IDs. Bot traffic in particular
  directly poisons the popularity model and the item-item similarity matrix.
  Needs schema validation, bot filtering and deduplication *before* aggregation.
- **Privacy (R8).** `name` is stored but used by no model — drop it. More
  seriously, deleting a customer's rows does not remove the embedding the NCF
  learned from them; genuine erasure needs retraining or explicit embedding
  removal.

### R5 — Plan the scaling migration at 40,000 users
**Priority: MEDIUM**

User-user collaborative filtering is O(users²) — 133 MB today, 40 GB at 100k
users, 4 TB at 1M. It is the first thing that breaks and it breaks hard.

It is also **not used by the hybrid at all** — it exists as a mandated baseline
and to supply similar-customer explanations. Drop it from the production
artifact set and derive explanation evidence from the item-item matrix.

Migrate item retrieval to an ANN index (FAISS/HNSW) before the user base makes
it urgent. Replacing infrastructure under load is worse than replacing it early.

### R6 — Add exploration to break the popularity loop
**Priority: MEDIUM**

The current inverse-propensity penalty dampens the feedback loop but cannot
break it, because the system never explores — it can only promote items it
already has signal for.

Epsilon-greedy or Thompson sampling on a small share of slots would close this.
It requires live traffic, so it is the first thing to build *after* launch.

### R7 — Reduce NCF cost
**Priority: LOW**

The implicit NCF is 84% of training cost and 93% of hybrid inference cost. Two
cheap experiments:
- Halve the negative-sampling ratio from 1:4 to 1:2. The regularisation sweep
  already showed the model is capacity-bound at ~0.858 accuracy, so it may hold
  that ceiling on half the data.
- Quantise the linear layers for serving (~4× smaller, ~2× faster).

---

## 4. Financial framing

At 100,000 users, total system cost is roughly **$570/month** (training,
storage, batch scoring, and real-time serving for 20% of traffic).

On $2M monthly GMV, the break-even revenue lift is **0.03%**. A working
recommender typically lifts 5–15%.

**Compute cost is not the risk. The risk is deploying something that does not
lift conversion and never measuring whether it did.** This is why R3 outranks
every optimisation on the list.

---

## 5. What this project does and does not prove

### Does
- The pipeline runs end to end, from data generation to two serving layers
- The evaluation methodology is sound: time-based split, asserted no-leakage,
  models refitted on train-only data, relevance defined to include unrated
  purchases
- Cold-start handling works for new customers, unknown customers and new
  products, and is visibly distinguished from personalised results
- Explanations are derived from real scores and are honest about their own
  limits
- Fairness and cost can be measured, and the apparatus to do so exists

### Does not
- **Predict real-world performance.** The models learned the rules the generator
  was written with. NDCG 0.1095 is a measure of how well those known rules were
  recovered.
- **Validate cold-start quality.** No ground truth exists for customers with no
  history.
- **Establish that the deep model is worth deploying.** On this data it loses to
  a similarity matrix. That may reverse with real data at larger scale — or it
  may not.

---

## 6. Proposed roadmap

| Phase | Actions | Success criterion |
|---|---|---|
| **0 — Now** | Ship item-based CF, batch-served. Instrument CTR/conversion. | Telemetry flowing |
| **1 — Month 1–2** | Close ingestion quality and privacy gaps. Add the hybrid on batch surfaces. | Both HIGH risks retired |
| **2 — Month 3–4** | A/B test hybrid vs item-CF on live traffic. Add exploration. | Measured lift, or a decision not to |
| **3 — Month 5–6** | Retraining scheduler with drift triggers. Real-time path for cart and search. | Automated refresh, monitored |
| **4 — At 40k users** | ANN migration, drop user-user CF, sparse storage. | Latency held flat as users grow |

---

## 7. Closing assessment

The system meets every requirement in the brief and the hybrid is measurably the
best of six approaches. But the two most valuable outputs of this exercise are
not the model.

They are, first, the **evaluation harness** — which caught two genuine failures
that would otherwise have shipped silently: a deep model that could not rank at
all, and a hybrid that was losing to one of its own inputs. Neither was visible
from training loss.

And second, the **honest accounting of what synthetic data can and cannot
establish**. The pipeline, the methodology and the serving architecture transfer
to real data. The numbers do not.

The single highest-value next action is not a model improvement. It is
instrumenting the business metrics that would tell us whether any of this works.
