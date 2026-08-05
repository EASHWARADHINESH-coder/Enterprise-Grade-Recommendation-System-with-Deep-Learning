# System Architecture & Scalability Document

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Architecture overview

```
                        ┌──────────────────────────┐
                        │   SYNTHETIC DATA LAYER   │
                        │  generate_synthetic_data │
                        │  users / items / events  │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │    DATA ACCESS LAYER     │
                        │      data_loader.py      │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  FEATURE ENGINEERING     │
                        │     preprocessing.py     │
                        │  • rating matrix         │
                        │  • normalised matrix     │
                        │  • implicit matrix       │
                        │  • popularity features   │
                        │  • content features      │
                        │  • user profiles         │
                        └────────────┬─────────────┘
                                     │
        ┌───────────────┬────────────┼────────────┬───────────────┐
        │               │            │            │               │
┌───────▼──────┐ ┌──────▼─────┐ ┌────▼─────┐ ┌────▼──────┐ ┌──────▼──────┐
│  POPULARITY  │ │  ITEM CF   │ │   SVD    │ │  TF-IDF   │ │  NCF (torch)│
│   baseline   │ │  cosine    │ │ latent   │ │  content  │ │  embeddings │
└───────┬──────┘ └──────┬─────┘ └────┬─────┘ └────┬──────┘ └──────┬──────┘
        │               │            │            │               │
        └───────────────┴────────────┼────────────┴───────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │    SCORE FUSION LAYER    │
                        │   hybrid_recommender.py  │
                        │  weighted linear blend   │
                        │  + retrieve-then-rerank  │
                        │  + cold-start fallback   │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  EXPLAINABILITY LAYER    │
                        │    explainability.py     │
                        │  signal attribution      │
                        │  evidence retrieval      │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
        ┌───────────▼──────────┐        ┌─────────────▼──────────┐
        │   FastAPI service    │        │  Streamlit dashboard   │
        │  /recommend/{id}     │        │  operator console      │
        │  /similar-items/{id} │        │  explainability panel  │
        │  /explain/{u}/{i}    │        │  cold-start demo       │
        │  /recommend/batch    │        │                        │
        └──────────────────────┘        └────────────────────────┘
```

**Key structural decision:** all recommendation logic lives in
`hybrid_recommender.py`, exposed as a loadable service class. The two
application layers are transport only. Duplicating fusion logic into each
application is how they drift apart and start returning different answers for
the same customer — which is exactly what happened in the previous revision of
this project, where the `reports/` folder held byte-identical copies of every
source module.

---

## 2. Batch vs real-time recommendation flow

The system supports both. Which path a request takes depends on latency budget
and freshness requirement, not on preference.

### 2.1 Real-time path (online serving)

```
request → load cached artifacts → score signals → fuse → re-rank → explain → respond
```

| Stage | Cost | Notes |
|---|---|---|
| Artifact load | one-off at startup | ~400 MB resident |
| Item-CF scoring | O(rated_items × catalogue) | matrix slice + dot product |
| SVD scoring | O(catalogue) | single row lookup |
| Content scoring | O(liked_items × catalogue) | similarity row mean |
| NCF scoring | one batched forward pass | ~10 ms CPU |
| Fusion + re-rank | O(candidate_pool) | pool = 200 |

**Measured: ~190 ms per request** for a warm customer on CPU, dominated by the
NCF forward pass and the similarity slice.

Use for: product pages, search result re-ranking, session-based surfaces — any
context where the customer's *current* behaviour must influence what they see.

### 2.2 Batch path (offline scoring)

```
nightly job → score all users → write top-N to a key-value store → serve by lookup
```

`POST /recommend/batch` implements this. Serving becomes an O(1) key lookup, so
p99 latency drops to single-digit milliseconds.

Use for: email campaigns, push notifications, homepage slots, any surface where
recommendations can be hours stale without harm.

### 2.3 Recommended hybrid deployment

| Surface | Path | Freshness |
|---|---|---|
| Homepage rails | batch | daily |
| Email / push | batch | daily |
| Product detail "similar items" | batch (item-item is user-independent) | daily |
| Post-add-to-cart suggestions | real-time | immediate |
| Search re-ranking | real-time | immediate |

Around 80% of traffic can be served from precomputed results. Reserve the
real-time path for surfaces where in-session signal genuinely changes the answer.

---

## 3. Retraining strategy

Different components decay at very different rates, so a single retraining
cadence would be either wasteful or negligent.

| Component | Cadence | Trigger | Cost |
|---|---|---|---|
| Popularity model | hourly | scheduled | seconds |
| Item-item similarity | daily | scheduled | ~3 s at current scale |
| SVD factors | daily | scheduled | ~5 s |
| TF-IDF + content similarity | **on catalogue change** | new items listed | ~10 s |
| NCF | weekly | scheduled + drift alarm | ~8 min CPU |
| Fusion weights | monthly | offline sweep | ~10 min |

**Why TF-IDF is event-driven rather than scheduled:** a newly listed product is
invisible to the content recommender until the vectoriser has seen it. That is
the item cold-start window, and it should be minutes, not a day. Refit on
catalogue change.

**Why NCF is weekly rather than daily:** it is the most expensive component and
the slowest to shift. Embeddings for established customers change little
week-to-week; the marginal accuracy from daily retraining does not repay the
compute.

### 3.1 Drift triggers

Retrain ahead of schedule when any of these fire:

- NDCG@10 on a rolling holdout drops > 10% from the trailing 7-day mean
- Catalogue coverage falls below 15%
- Cold-start cohort exceeds 15% of active users (indicates acquisition outpacing
  model refresh)
- Category distribution of interactions shifts by > 20% (seasonality, new
  department launch)

### 3.2 Safe deployment

Every retrain must clear a gate before replacing the incumbent:

1. Train the challenger on data up to `T - 90d`.
2. Evaluate on the held-out 90-day window.
3. Promote **only if** NDCG@10 ≥ incumbent AND coverage ≥ incumbent × 0.9.
4. Shadow-serve for 24 h, comparing distributions rather than metrics.
5. Ramp 5% → 25% → 100% with automatic rollback on CTR regression.

The coverage floor in step 3 exists because an accuracy-only gate will happily
promote a model that has quietly collapsed onto the head of the catalogue.

---

## 4. Data growth handling

### 4.1 What breaks first

| Component | Complexity | At 6k users | At 100k users | At 1M users |
|---|---|---|---|---|
| **User-user similarity** | O(U²) | 133 MB | **40 GB** | **4 TB** |
| Item-item similarity | O(I²) | 20 MB | 20 MB* | 20 MB* |
| SVD | O(U × I × k) | 5 s | ~90 s | intractable dense |
| NCF embeddings | O((U + I) × d) | 1.1 MB | 13 MB | 128 MB |
| TF-IDF | O(I × V) | 50 MB | 50 MB* | 50 MB* |

\* catalogue size grows far more slowly than the user base.

**User-user collaborative filtering is the first thing that breaks**, and it
breaks hard — it is quadratic in the fastest-growing dimension. This is the
single most important scaling fact about the current design.

### 4.2 Scaling path

**Phase 1 — up to ~50k users (current design holds)**
No change needed. Everything fits in memory on a single machine.

**Phase 2 — 50k to 500k users**
- Drop user-user CF entirely. Item-item CF gives better accuracy here anyway
  (0.1044 vs SVD's 0.0462) and does not grow with the user base.
- Replace exact item-item similarity with an ANN index (FAISS / HNSW).
- Move SVD to `implicit` ALS or sparse randomised SVD.
- Precompute batch recommendations nightly; reserve real-time for high-value
  surfaces.

**Phase 3 — beyond 500k users**
- Two-stage retrieval: ANN candidate generation (~1,000 items) then neural
  re-ranking of that shortlist only. Never score the full catalogue online.
- Shard embedding tables; serve NCF from a dedicated inference service.
- Stream interactions through Kafka; maintain features in a feature store
  rather than recomputing matrices.
- Move from full retraining to incremental embedding updates.

### 4.3 Storage

| Artifact | Current | Growth |
|---|---|---|
| Raw interactions | 14 MB | linear in events |
| User-user similarity | 133 MB | **quadratic — remove at Phase 2** |
| Item-item similarity | 20 MB | quadratic in catalogue (slow) |
| Content similarity | 25 MB | quadratic in catalogue (slow) |
| NCF weights | 1.1 MB | linear |

---

## 5. Monitoring KPIs

### 5.1 Model health

| Metric | Target | Alarm |
|---|---|---|
| NDCG@10 (rolling holdout) | ≥ 0.10 | < 0.08 |
| Catalogue coverage | ≥ 20% | < 15% |
| Long-tail share | 5–15% | < 2% or > 40% |
| Cold-start fallback rate | < 10% of requests | > 20% |
| NCF score variance across users | > 0 | ≈ 0 (**model collapse**) |

The last one is not standard, and it is here because of a real failure found in
this project: the explicit NCF returned near-identical scores to every customer
(coverage 0.006). Per-user score variance collapsing toward zero is the earliest
detectable symptom, and it is invisible to accuracy metrics.

### 5.2 System health

| Metric | Target | Alarm |
|---|---|---|
| p50 latency | < 100 ms | > 200 ms |
| p99 latency | < 500 ms | > 1 s |
| Error rate | < 0.1% | > 1% |
| Artifact load success | 100% | any failure |
| Memory resident | < 2 GB | > 4 GB |

### 5.3 Business KPIs

| Metric | Why it matters |
|---|---|
| CTR on recommended items | direct engagement signal |
| Conversion rate from recommendations | the metric that pays for the system |
| Revenue per recommendation slot | compares surfaces against each other |
| Share of revenue from long-tail items | is the diversity work paying off? |
| Category diversity per customer session | filter-bubble early warning |

### 5.4 Fairness monitoring

| Metric | Alarm |
|---|---|
| Coverage gap between highest and lowest activity band | > 2× |
| Recommended price spread across segments | growing faster than observed purchase spread |
| Cold-start customers receiving fallback after 30 days | > 5% |

---

## 6. Failure modes and degradation

The service is designed to fail loudly, not quietly. This is a direct response
to a defect in the previous revision, where a missing artifact path caused the
deep-learning signal to be silently dropped — the API served recommendations
with the NCF weight effectively zero and reported success.

| Failure | Detection | Behaviour |
|---|---|---|
| NCF artifacts missing | startup check | **loud warning**, hybrid runs on remaining signals with weights renormalised |
| Any core artifact missing | startup check | service refuses to start; `/health` returns 503 |
| Unknown customer | request-time | global popularity fallback, **labelled in the response** |
| Customer with < 3 interactions | request-time | profile-based cold start, **labelled in the response** |
| Item not in catalogue | request-time | 404, not a silent empty list |

Every response carries a `strategy` field. A caller must always be able to
distinguish a personalised result from a fallback — presenting the two
identically is how a dashboard ends up claiming a brand-new customer has a
learned taste profile.

---

## 7. Technology choices

| Layer | Technology | Rationale |
|---|---|---|
| Synthetic data | Faker, NumPy | required by brief; seeded for reproducibility |
| Data processing | pandas | standard, adequate at this scale |
| Classical ML | scikit-learn | `TruncatedSVD`, cosine similarity, TF-IDF |
| Deep learning | PyTorch | required by brief |
| API | FastAPI | async, automatic OpenAPI docs, Pydantic validation |
| Dashboard | Streamlit | fastest path to an operator console |
| Version control | Git | required by brief |

**On `surprise`:** the brief names it as an option for matrix factorisation. It
has no wheel for Python 3.13 and fails to build. `TruncatedSVD` provides the
same latent-factor method without the dependency, and the substitution is noted
in the code.

---

## 8. Deployment recommendation

For an early-stage platform matching the brief's scenario:

1. **Serve batch recommendations from day one.** Nightly job, results in Redis.
   Covers homepage, email and product pages at single-digit millisecond latency.
2. **Add the real-time path only for cart and search.** These are where session
   context genuinely changes the answer.
3. **Ship item-based CF first, add NCF second.** Item-CF is more accurate here,
   trains in seconds, and is explainable. NCF earns its place inside the hybrid,
   not as a replacement.
4. **Instrument before optimising.** Without CTR and conversion telemetry, every
   offline metric in this report is a proxy for something unmeasured.
5. **Plan the Phase 2 migration at 40k users**, not at 50k. Replacing user-user
   CF under load is worse than replacing it early.
