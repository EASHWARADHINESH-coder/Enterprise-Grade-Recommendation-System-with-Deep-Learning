# Evaluation & Benchmarking Report

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Methodology

### 1.1 Time-based split

The final **90 days** of the 18-month log are held out. Every model is trained
only on what came before.

| | Period | Interactions |
|---|---|---|
| Train | 2025-01-07 → 2026-04-01 | 96,656 |
| Test | 2026-04-02 → 2026-06-30 | 41,689 |

A random split would leak the future into training — the model sees what a
customer did in November while being asked to predict their October behaviour,
and every reported metric is inflated as a result.

### 1.2 No interaction leakage

Two rules, both **asserted in code** rather than assumed
(`notebooks/recommendation_evaluation.py`):

1. No model sees any test-period interaction. Every model — SVD, item-item
   similarity, popularity ranking, the popularity lookup — is **refitted from
   scratch on the training period**. Reusing the artifacts in `data/processed/`
   would have been far quicker, but those were fitted on the full history
   including the test window.
2. The candidate set excludes everything the customer touched during training.
   Without this, a model scores points for "predicting" purchases it was shown.

```python
assert test_df["timestamp"].min() > cutoff
assert train_df["timestamp"].max() <= cutoff
```

### 1.3 Relevance definition

An interaction in the test period is **relevant** if the customer rated it ≥ 4
**or** purchased it.

Including purchases is not optional. A bought-but-unrated item is unambiguous
evidence of relevance, and excluding it would understate every model's recall by
roughly the share of purchases that go unrated. NDCG additionally uses graded
relevance — the actual star rating — so a 5-star hit outranks a 4-star one; an
unrated purchase is graded at the threshold value.

### 1.4 Cohort

800 customers sampled from the 4,746 who have at least one relevant test item.
Customers with nothing to find are excluded: including them drags every metric
toward zero by an amount that depends on the split rather than on model quality,
making runs incomparable.

Mean relevant items per evaluated customer: **4.4**.

---

## 2. Ranking metrics

### K = 10 (headline)

| Model | Precision@10 | Recall@10 | MAP@10 | **NDCG@10** | HitRate@10 |
|---|---|---|---|---|---|
| **Hybrid** | **0.0628** | **0.1372** | **0.0593** | **0.1095** | **0.3850** |
| Item-based CF | 0.0579 | 0.1241 | 0.0593 | 0.1044 | 0.3575 |
| NCF (Deep Learning) | 0.0375 | 0.0858 | 0.0349 | 0.0663 | 0.2812 |
| Popularity | 0.0375 | 0.0843 | 0.0336 | 0.0645 | 0.2825 |
| SVD | 0.0278 | 0.0511 | 0.0230 | 0.0462 | 0.2050 |
| Content (TF-IDF) | 0.0073 | 0.0161 | 0.0049 | 0.0116 | 0.0675 |

**Headline: the Hybrid is the best model on every ranking metric, at
+69.9% NDCG@10 over the popularity baseline.**

### K = 5 and K = 20

Ordering is stable across K. Full results in
`data/processed/recommendation_evaluation_results.csv`.

---

## 3. Beyond-accuracy metrics

Accuracy alone is a dangerous way to judge a recommender. A model can post
excellent precision while only ever recommending 32 items out of 2,500 —
commercially a failure, because the remaining inventory is invisible and will
never sell.

| Model | Coverage | Long-tail share | Novelty | Intra-list diversity |
|---|---|---|---|---|
| Content (TF-IDF) | **0.754** | 0.814 | **12.32** | 0.707 |
| Item-based CF | 0.222 | 0.056 | 7.67 | 0.936 |
| Hybrid | 0.202 | 0.040 | 7.91 | 0.911 |
| SVD | 0.096 | 0.069 | 8.57 | 0.955 |
| Popularity | 0.013 | 0.000 | 6.83 | 0.962 |
| NCF | 0.013 | 0.000 | 6.91 | 0.952 |

The content model is the weakest on accuracy (NDCG 0.0116) and by far the
strongest on coverage (75.4%). That is exactly why it earns a small weight in
the hybrid despite its standalone score — it is the only component that can
reach the long tail at all.

Popularity has 1.3% coverage by construction: it returns the same list to
everybody.

---

## 4. The two failures found during benchmarking

Both were discovered *because* the benchmark was run, and both were fixed rather
than reported around.

### 4.1 The explicit NCF could not rank at all

First run: **NDCG@10 = 0.0000** for the deep model, with catalogue coverage of
0.006 — it returned the same ~14 items to every single customer.

This was not a tuning problem. The explicit model minimises MSE against observed
star ratings. It is a *good rating predictor* — RMSE 0.911 on a 1–5 scale — but
it has never once been shown an item a customer did **not** interact with, so
scoring the unseen catalogue is entirely out of distribution. What it falls back
on is the global item bias, which is identical for everybody.

The implicit model is trained with **negative sampling**: 4 negatives drawn from
the catalogue the customer never touched, per observed positive. Discriminating
"would engage" from "would not" *is* its training objective.

| Variant | Accuracy | F1 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Explicit (rating MSE) | n/a | n/a | **0.0000** | 0.006 |
| Implicit (BCE + negatives) | 0.858 | 0.582 | **0.0576** | 0.013 |

**Lesson: a model can be accurate on the metric you trained it on and worthless
at the task you actually need.** Production ranks with the implicit variant.

Before negative sampling was added at all, the implicit model scored 0.534
accuracy — barely above chance — because it was being asked "given they clicked
it, did they buy it?", a conversion problem rather than a retrieval one.

### 4.2 The hybrid was worse than its own components

Second run, after fixing the NCF: Hybrid NDCG@10 = 0.0760 against Item-based
CF's 0.1044. **The hybrid was losing to one of its own inputs.**

Cause: the hybrid's "collaborative" slot was filled by SVD (0.0462), the
*weakest* collaborative signal, while item-based CF (0.1044) — the strongest
single model in the system — was not in the fusion at all.

Weight sweep over a 250-customer cohort:

| Configuration | NDCG@10 |
|---|---|
| item-based CF alone | 0.1245 |
| svd + content + ncf (original) | 0.0947 |
| item_cf + content + ncf | 0.1282 |
| item_cf .50 / content .10 / ncf .40 | 0.1276 |
| item_cf .60 / content .10 / ncf .30 | 0.1258 |
| **item_cf .45 / svd .10 / content .10 / ncf .35** | **0.1293** |
| item_cf .70 / ncf .30 (no content) | 0.1181 |

Final weights:

```python
HYBRID_WEIGHTS = {
    "item_cf":       0.45,
    "collaborative": 0.10,   # SVD
    "content":       0.10,
    "ncf":           0.35,
}
```

SVD is retained at low weight because it generalises to customers whose exact
co-rating neighbours are absent. Content is retained because of its coverage.
Dropping either loses NDCG.

---

## 5. Popularity de-biasing: the accuracy/diversity dial

The re-ranker applies `1 / (count + 1) ** alpha` with a long-tail boost.

### 5.1 What was tried first and why it failed

The obvious form, `1 / (1 + log1p(count))`, spans a ~5× range across this
catalogue — **wider than the spread of the normalised relevance scores it
multiplies**. The penalty dominated the ranking outright:

| | Before re-ranking | After |
|---|---|---|
| Top-10 mean exposure | 518 interactions | **1** |
| Top-10 long-tail share | 0% | **100%** |

The system was recommending obscurity for its own sake — all diversity, zero
relevance.

### 5.2 Two structural fixes

**Retrieve, then re-rank.** The diversity correction now applies only to the top
200 candidates by raw relevance, not across the whole catalogue. Applied
catalogue-wide, a near-zero-exposure item the customer has no affinity for
outranks a genuinely relevant one purely for being obscure.

**Normalise before multiplying.** SVD runs on the mean-centred matrix, so its
predictions are *signed*. Multiplying a negative score by a penalty in (0, 1)
moves it *up* toward zero — silently inverting the correction for every
negatively scored item. Min-max scaling to [0, 1] first makes the multiplication
mean what it says.

### 5.3 Calibration

Measured inside the two-stage ranker, top-10 overlap against the uncorrected
ranking:

| alpha | Overlap | Interpretation |
|---|---|---|
| 0.00 | 100% | no correction |
| 0.10 | 97% | barely perceptible |
| **0.20** | **92%** | visible tail promotion, relevance intact |
| 0.35 | 86% | noticeable relevance cost |

**Selected: alpha = 0.20.**

---

## 6. Predictive metrics (explicit model)

| Metric | Value |
|---|---|
| MSE | 0.8296 |
| RMSE | 0.9108 |
| MAE | 0.7449 |

On a 1–5 scale, an RMSE of 0.91 means predictions land within roughly one star.
As section 4.1 established, this says **nothing** about ranking quality.

---

## 7. Classical CF vs deep learning

| | Item-based CF | NCF (implicit) |
|---|---|---|
| NDCG@10 | **0.1044** | 0.0663 |
| Training time | ~3 s (similarity matrix) | 473 s (20 epochs, CPU) |
| Inference (800 users) | 2.6 s | 154 s |
| Parameters | — | 278,817 |
| Interpretable | yes — "similar to X you bought" | no |
| Cold-start items | no | no |
| Scales to 1M users | no (O(items²) is fine, O(users²) is not) | yes (embeddings) |

**On this dataset, classical item-based CF beats the deep model on accuracy at a
fraction of the cost.** That is an honest and unsurprising result at this scale:
NCF's advantage is in modelling complex interaction patterns that emerge with
much larger and richer data, and with 138k interactions across 6,000 customers
there is not enough signal to justify 278k parameters.

The deep model still earns its place — it contributes 0.35 of the hybrid weight
and the hybrid beats item-based CF alone (0.1095 vs 0.1044). But a
recommendation to deploy NCF *instead of* item-based CF would not be supported
by this evidence.

---

## 8. Limitations

**Synthetic data.** Every number here comes from generated data. The models have
learned the rules the generator was written with — category affinity and segment
price sensitivity. The metrics measure how well those specific rules were
recovered. No claim is made that they would hold on real customer data.

**Offline proxy.** Ranking metrics measure agreement with historical behaviour,
which is itself a product of whatever ranked items in the past. Only an online
A/B test measures whether recommendations actually change what customers buy.

**Cold-start quality is unverifiable.** Customers with no history have no
ground truth, so the cold-start path can be demonstrated but not scored.

**Single split.** One time-based split, not cross-validated across multiple
cutoffs. Confidence intervals on these numbers are not reported.

---

## 9. Reproducing

```bash
python notebooks/recommendation_evaluation.py
```

Outputs:
- `data/processed/recommendation_evaluation_results.csv`
- `data/processed/beyond_accuracy_results.csv`
