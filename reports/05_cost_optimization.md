# Cost Optimization Report

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Purpose

A recommendation system that costs more to run than the revenue it generates is
a liability regardless of its NDCG. This report measures where compute and
storage actually go, identifies what can be cut without losing accuracy, and
projects cost at scale.

All measurements are from this project running on **CPU only** (no GPU),
Python 3.13, PyTorch 2.12.

---

## 2. Measured compute cost

### 2.1 Training

| Component | Time | Share |
|---|---|---|
| Synthetic data generation | ~8 s | 1% |
| Preprocessing (all matrices) | ~12 s | 2% |
| TF-IDF + content similarity | ~10 s | 2% |
| Popularity model | < 1 s | — |
| User-user similarity | ~9 s | 2% |
| Item-item similarity | ~3 s | 1% |
| SVD (50 components) | ~5 s | 1% |
| **NCF explicit** | **39 s** | 7% |
| **NCF implicit** | **473 s** | **84%** |
| **Total full pipeline** | **~9.5 min** | |

**The implicit NCF alone is 84% of total training cost.**

The reason is negative sampling: 45,907 positives expand to 229,535 training
rows at a 1:4 ratio, and the model runs 20 epochs over that. Training cost
scales linearly with the negative ratio.

### 2.2 Inference (800 customers, full catalogue scoring)

| Model | Time | Per request |
|---|---|---|
| Popularity | 0.16 s | 0.2 ms |
| SVD | 1.1 s | 1.4 ms |
| Item-based CF | 2.6 s | 3.3 ms |
| Content (TF-IDF) | 3.5 s | 4.4 ms |
| **NCF** | **154 s** | **192 ms** |
| **Hybrid (all four)** | **166 s** | **208 ms** |

**NCF is ~93% of hybrid inference cost** and 60× more expensive than the next
most costly signal.

### 2.3 The uncomfortable ratio

| | Item-based CF | Hybrid |
|---|---|---|
| NDCG@10 | 0.1044 | 0.1095 |
| Inference per request | 3.3 ms | 208 ms |
| **Cost per NDCG point** | **32 ms** | **1,900 ms** |

**The hybrid buys +4.9% NDCG for 63× the inference cost.**

That is the single most important number in this report. It does not mean the
hybrid is wrong — it means the deployment decision depends entirely on the
surface. On a homepage rail served from a nightly batch, 208 ms is free. On a
real-time search re-rank at 10,000 QPS, it is the difference between 4 servers
and 250.

---

## 3. Storage cost

| Artifact | Size | Growth |
|---|---|---|
| Raw CSVs | 22 MB | linear in events |
| `user_similarity.pkl` | **133 MB** | **O(users²)** |
| `content_similarity.pkl` | 25 MB | O(items²) |
| `item_similarity.pkl` | 20 MB | O(items²) |
| `predicted_ratings.pkl` | 52 MB | O(users × items) |
| `implicit_item_similarity.pkl` | 21 MB | O(items²) |
| TF-IDF matrix + vectoriser | 8 MB | O(items × vocab) |
| NCF weights (both variants) | 2.2 MB | O((U+I) × d) |
| Other matrices | ~80 MB | |
| **Total** | **~365 MB** | |

**`user_similarity.pkl` is 36% of storage and is the only artifact growing
quadratically in the fastest-moving dimension.**

---

## 4. Optimizations already applied

### 4.1 float32 instead of float64
**Saving: ~50% of matrix memory, zero accuracy cost.**

All similarity and rating matrices are stored as float32. At 6,000 × 2,500 the
rating matrix alone is 15 million cells; float64 doubles the memory for
precision that recommendation scores do not need.

### 4.2 Gumbel top-k sampling in data generation
**Saving: minutes → seconds.**

The original generator drew each interaction with `df.sample(1)` inside a
`while` loop — 125,000 individual pandas sampling calls. The rewrite uses the
Gumbel top-k trick, which is mathematically equivalent to sequential weighted
sampling without replacement but runs as a single vectorised operation per
customer.

### 4.3 Vectorised mean-centring
**Saving: ~30× on the normalisation step.**

The original used a row-wise `.apply()` with a `.loc` lookup per row. Replaced
with `groupby().transform()`.

### 4.4 Batched NCF scoring
**Saving: ~100× on inference.**

Scoring items individually would issue thousands of tiny forward passes per
request. `score_all_items` builds one tensor and does a single forward pass.

### 4.5 Precomputed user history
**Saving: ~40 ms per request.**

`HybridRecommender` precomputes each customer's interaction set once at load.
Filtering the 138k-row interaction log per request was the largest avoidable
cost in the serving path.

### 4.6 Retrieve-then-rerank
**Saving: re-ranking cost is O(200), not O(2,500).**

Applied for correctness reasons, but it also cuts re-ranking work by 92%.

### 4.7 Lazy loading of the user-similarity matrix
**Saving: 133 MB resident.**

The largest artifact is loaded only when a caller actually requests
similar-customer evidence, not held resident for every request.

---

## 5. Further optimizations available

### 5.1 Delete the user-user similarity matrix
**Saving: 133 MB storage, 9 s training. Accuracy cost: zero.**

User-user CF is **not used by the hybrid at all** — it exists only as a mandated
baseline and to supply similar-customer evidence for explanations. Item-based CF
outperforms it and does not grow with the user base.

**Recommendation:** keep it for the baseline comparison in this submission; drop
it in production and derive explanation evidence from the item-item matrix
instead.

### 5.2 Reduce the NCF negative sampling ratio
**Potential saving: ~50% of NCF training time.**

At 1:4 the implicit training set is 229k rows. At 1:2 it would be 138k. The
original NCF paper uses 4, but the regularisation sweep already showed this model
is **capacity-bound at ~0.858 accuracy regardless of settings** — it may well
hold that ceiling at a lower ratio.

**Untested.** Worth one experiment before scaling up.

### 5.3 Quantise NCF weights for serving
**Potential saving: 4× model size, ~2× inference speed.**

`torch.quantization.quantize_dynamic` on the linear layers. Typical accuracy
cost for recommendation ranking is negligible.

### 5.4 Sparse matrix storage
**Potential saving: ~90% on the rating matrices.**

The rating matrix is 99.08% zeros but stored dense (52 MB). `scipy.sparse.csr`
would cut this to a few MB. The trade-off is that pandas `.loc` row lookups
become more awkward, so this is worth doing at Phase 2, not now.

### 5.5 Batch-first serving
**Potential saving: ~95% of serving compute.**

Precomputing top-N nightly turns a 208 ms hybrid call into a single-digit
millisecond key lookup. Around 80% of surfaces tolerate daily-stale
recommendations.

---

## 6. Cost projection at scale

Assumptions: AWS-equivalent pricing, ~$0.10/vCPU-hour, ~$0.10/GB-month storage.

### 6.1 Current scale (6k users, 2.5k items)

| Item | Monthly |
|---|---|
| Training: full pipeline daily (~10 min/day) | ~$0.50 |
| Storage: 365 MB | ~$0.04 |
| Serving: batch-first, 1 small instance | ~$15 |
| **Total** | **~$16/month** |

Negligible. A single small instance runs everything.

### 6.2 100k users, 20k items

| Item | Monthly | Note |
|---|---|---|
| Training | ~$45 | NCF dominates; weekly not daily |
| Storage | ~$2 | after dropping user-user CF |
| Batch scoring (nightly, all users) | ~$120 | |
| Real-time serving (20% of traffic) | ~$400 | |
| **Total** | **~$570/month** | |

**Without the optimizations above:** user-user similarity alone would be 40 GB,
making the daily job infeasible on a single machine and pushing cost past
$3,000/month.

### 6.3 1M users, 50k items

| Item | Monthly |
|---|---|
| Training (GPU, weekly) | ~$300 |
| Storage (with sparse + ANN index) | ~$25 |
| Batch scoring | ~$900 |
| Real-time serving (two-stage retrieval) | ~$2,200 |
| Feature store + streaming | ~$800 |
| **Total** | **~$4,200/month** |

At this scale the architecture must change (see the scalability document) — this
is not the current design scaled up, it is the Phase 3 design costed.

---

## 7. Cost/benefit framing

For an e-commerce platform, a recommendation system typically needs to lift
revenue by 1–3% to justify itself.

At 100k users with, say, $2M monthly GMV:
- System cost: ~$570/month
- Break-even lift: **0.03%**
- Typical observed lift from a working recommender: 5–15%

**The system pays for itself many times over — provided it actually works.**
That conditional is the whole point of the evaluation report: cost is trivial
relative to revenue impact, so the risk is not overspending, it is shipping
something that does not lift conversion and never measuring it.

**The most expensive mistake available here is not compute. It is deploying
without CTR and conversion telemetry**, because then no amount of offline NDCG
tells you whether the spend was worthwhile.

---

## 8. Recommendations

**Immediate (this submission):**
1. Keep the current design — cost is negligible at this scale.
2. Document that user-user CF is a baseline only, not a production component.

**Before production:**
3. Serve batch-first; reserve real-time for cart and search only.
4. Drop user-user similarity from the production artifact set (−133 MB, −9 s).
5. Quantise the NCF for serving.
6. Instrument CTR and conversion **before** optimising anything else.

**At 40k users:**
7. Migrate to ANN-based item retrieval.
8. Move rating matrices to sparse storage.
9. Move NCF training to GPU or reduce the negative ratio.
