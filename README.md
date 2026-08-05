# Enterprise-Grade Recommendation System with Deep Learning

**Domain:** E-Commerce / Retail
**Author:** Eashwaradhinesh K

An end-to-end recommendation system built for a digital platform operating
without real interaction data — the early-stage / new-market scenario where a
recommender must be designed, prototyped and validated *before* live traffic
exists.

Built with **Python, pandas, scikit-learn, PyTorch, FastAPI and Streamlit**.

---

## Headline result

| Model | NDCG@10 | Coverage | Inference/request |
|---|---|---|---|
| **Hybrid (4-signal fusion)** | **0.1095** | 20.2% | 208 ms |
| Item-based CF | 0.1044 | 22.2% | 3.3 ms |
| NCF (deep learning) | 0.0663 | 1.3% | 192 ms |
| Popularity baseline | 0.0645 | 1.3% | 0.2 ms |
| SVD | 0.0462 | 9.6% | 1.4 ms |
| Content (TF-IDF) | 0.0116 | 75.4% | 4.4 ms |

**The hybrid is the best model on every ranking metric — a 69.8% lift over the
popularity baseline** — measured on a leak-free time-based split.

---

## Problem statement

A digital platform in an early-stage or new-market position needs a
recommendation system, but real user interaction data is limited, unavailable or
privacy-restricted. The system must therefore:

1. Generate realistic synthetic data
2. Build baseline and deep learning recommenders
3. Handle cold-start scenarios
4. Evaluate recommendation quality rigorously
5. Expose recommendations through an application layer

---

## Quickstart

```bash
pip install -r requirements.txt
```

Run the pipeline in order — each stage writes artifacts the next one consumes:

```bash
python synthetic_data/generate_synthetic_data.py
```
```bash
python src/preprocessing.py
```
```bash
python src/content_based_nlp.py
```
```bash
python models/baseline_recommenders.py
```
```bash
python src/collaborative_filtering.py
```
```bash
python models/ncf_recommender.py
```
```bash
python notebooks/eda_validation.py
```
```bash
python notebooks/recommendation_evaluation.py
```

Then launch either application layer:

```bash
streamlit run app/streamlit_recommendation_dashboard.py
```
```bash
python app/app_fastapi.py
```

FastAPI docs at <http://127.0.0.1:8000/docs>.

Total pipeline runtime: **~10 minutes on CPU**, dominated by NCF training.

---

## Project structure

```
data/
  raw/                users.csv, items.csv, interactions.csv
  processed/          model artifacts (.pkl, .pt)
synthetic_data/
  generate_synthetic_data.py     data generation with engineered realism
src/
  data_loader.py                 loading layer
  preprocessing.py               feature engineering
  content_based_nlp.py           TF-IDF content recommender
  collaborative_filtering.py     CF improvements + re-ranking
  evaluation_metrics.py          Precision/Recall/MAP/NDCG + diversity
models/
  baseline_recommenders.py       Popularity, User-CF, Item-CF, SVD
  ncf_recommender.py             PyTorch Neural Collaborative Filtering
  hybrid_recommender.py          score fusion service class
  explainability.py              three mandated explanation types
notebooks/
  01_advanced_eda.ipynb
  02_feature_engineering.ipynb
  03_ml_pipeline.ipynb
  04_deep_learning_model.ipynb
  05_explainability_and_governance.ipynb
  eda_validation.py              standalone EDA script
  recommendation_evaluation.py   benchmarking harness
app/
  app_fastapi.py                 REST service
  streamlit_recommendation_dashboard.py
reports/
  01_synthetic_data_design_and_validation.md
  02_evaluation_and_benchmarking.md
  03_system_architecture_and_scalability.md
  04_data_risk_assessment.md
  05_cost_optimization.md
  06_governance_and_fairness.md
  07_final_business_recommendations.md
  figures/                       7 generated plots
```

Every module uses only standard libraries (`os`, `sys`, `pickle`, `numpy`,
`pandas`, `sklearn`, `torch`, `faker`) and computes its own paths relative to
`__file__`, so the project runs from any folder or drive with no configuration.

---

## 1. Synthetic data

**6,000 users · 2,500 items · 138,345 interactions** — all above the mandated
minimums.

The generator does not emit random rows. It deliberately engineers the
properties that make recommendation hard:

| Property | Achieved |
|---|---|
| Sparsity | 99.08% |
| Popularity bias | top 10% of items hold 62.9% of demand |
| Long tail | Gini 0.716 |
| Cold-start users | 480 (8.0%) |
| Cold-start items | 255 (10.2%) |
| Category preference | 60.7% of purchases in declared category |
| Price sensitivity | INR 572 (Budget) → INR 3,361 (Luxury) |
| Rating realism | J-shaped, mean 3.91, 67% at 4–5 stars |

Interactions follow a real engagement funnel:

```
click (100%) → add_to_cart (47.5%) → purchase (16.4%)
                      ↓
                rating (68.5%)
```

A single latent affinity score drives all four outcomes, so the signals stay
mutually consistent — a customer never rates five stars something they bounced
off in two seconds.

**Two latent factors** are embedded for the models to recover: category affinity
and segment-driven price sensitivity. They interact **non-linearly** — a
customer buys when the category matches *and* the price fits — which is
precisely what a dot product cannot represent and a neural network can.

All ten validation checks pass. See
[`reports/01_synthetic_data_design_and_validation.md`](reports/01_synthetic_data_design_and_validation.md).

---

## 2. Models

### Baselines
Popularity (blended count + rating + conversion), User-based CF, Item-based CF,
and Truncated SVD on the mean-centred rating matrix.

### Content-based (NLP)
TF-IDF over product description, tags, category and brand — with bigrams and
`min_df=2` to drop model codes. Needs no interaction history, which makes it the
answer to **item cold start**.

### Deep learning (PyTorch NCF)
```
user_id → embedding (32d) ─┐
                           ├→ concat (64d) → MLP [128, 64, 32] → score
item_id → embedding (32d) ─┘
```
Concatenation rather than element-wise product is the whole point: a product
hard-codes a multiplicative interaction and reduces to matrix factorisation;
concatenation leaves the interaction function to be learned. Dropout, weight
decay and early stopping with best-weight restoration.

### Hybrid
Four-signal weighted fusion, with weights chosen by measurement:

```python
HYBRID_WEIGHTS = {
    "item_cf":       0.45,   # strongest single signal
    "collaborative": 0.10,   # SVD, generalises where neighbours are absent
    "content":       0.10,   # weak on accuracy, best coverage (75%)
    "ncf":           0.35,   # deep model
}
```

---

## 3. Two failures the benchmark caught

Both were found *because* the evaluation harness was built properly, and both
are documented in full in
[`reports/02_evaluation_and_benchmarking.md`](reports/02_evaluation_and_benchmarking.md).

### The deep model could not rank at all

First benchmark: **NDCG@10 = 0.0000**, catalogue coverage 0.006 — the same ~14
items returned to every customer.

Not a tuning problem. The explicit model minimises MSE on observed ratings, so
it is a good *rating predictor* (RMSE 0.911) but has never been shown an item a
customer did **not** interact with. Scoring the unseen catalogue is out of
distribution, and it falls back on a global item bias identical for everybody.

The implicit variant is trained with **negative sampling** — 4 negatives per
positive, drawn from items the customer never touched — so discriminating "would
engage" from "would not" *is* its objective.

| Variant | Accuracy | NDCG@10 | Coverage |
|---|---|---|---|
| Explicit (rating MSE) | n/a | **0.0000** | 0.006 |
| Implicit (BCE + negatives) | 0.858 | **0.0576** | 0.013 |

**A model can be accurate on the metric you trained it on and worthless at the
task you need.**

### The hybrid was losing to its own input

Second benchmark: Hybrid 0.0760 vs Item-based CF 0.1044. The hybrid's
"collaborative" slot held SVD (0.0462) — the weakest signal — while item-based
CF, the strongest model in the system, was not in the fusion at all.

| Configuration | NDCG@10 |
|---|---|
| svd + content + ncf (original) | 0.0947 |
| item_cf + content + ncf | 0.1282 |
| **item_cf .45 / svd .10 / content .10 / ncf .35** | **0.1293** |

---

## 4. Evaluation methodology

- **Time-based split.** Final 90 days held out (train 96,656 / test 41,689).
- **No leakage, asserted in code.** Every model is refitted from scratch on the
  training period — reusing the stored artifacts would leak, since those were
  fitted on the full history.
- **Relevance = rated ≥ 4 OR purchased.** Excluding unrated purchases would
  understate every model's recall.
- **Graded NDCG** using the actual star rating.
- **Beyond-accuracy metrics**: coverage, long-tail share, novelty, intra-list
  diversity — because a model can post excellent precision while making 99% of
  the catalogue invisible.

---

## 5. Explainability

Three mandated explanation types, all **derived from the actual scores that
produced the ranking** — never from a template.

**Signal attribution** reports weight × score, not raw score, because a signal
with a high score but a low weight contributes little to the decision:

```
item-based collaborative filtering   51.3%
the deep learning model              38.7%
latent-factor collaborative filtering 7.3%
content similarity                    2.7%
```

**Evidence** from the customer's own comparable purchases, and from similar
customers.

**Content justification** exposing the actual matching TF-IDF terms, which is
what turns "similarity 0.83" into something a merchandiser can audit.

The system is also honest about its own limits. Similar-customer evidence is
often empty, and the cause is structural: the re-ranker promotes long-tail
items, and an item is long-tail precisely because few people have rated it. Told
plainly rather than hidden:

> "None of your 50 closest customer matches have rated this item — it is a
> low-exposure product surfaced for catalogue variety, so the recommendation
> rests on content and model signals rather than crowd evidence."

---

## 6. Cold-start handling

| Scenario | Strategy | Signal used |
|---|---|---|
| New customer (registered, no activity) | `cold_start_user_profile` | declared category + segment price band |
| Unknown customer | `global_popularity_fallback` | de-biased popularity |
| New product (just listed) | content similarity | TF-IDF over description |

Every response carries a `strategy` field. A caller must always be able to tell
a personalised result from a fallback — presenting them identically is how a
dashboard ends up claiming a brand-new customer has a learned taste profile.

---

## 7. Application layer

### FastAPI

| Endpoint | Purpose |
|---|---|
| `GET /recommend/{user_id}` | personalised top-N with explanations and signal breakdown |
| `GET /similar-items/{item_id}` | content-similar products |
| `GET /explain/{user_id}/{item_id}` | full three-part explanation |
| `GET /users/{user_id}` | profile and engagement summary |
| `GET /items/{item_id}` | product detail |
| `POST /recommend/batch` | batch scoring for offline jobs |
| `GET /health` | readiness, including `ncf_available` |

Responses carry business context: basket value, average price point, long-tail
share.

### Streamlit

Five tabs: Recommendations, Explainability, Similar Items, Cold-Start Demo,
Customer History. Customers are selectable by cohort (highly active / typical /
cold start) rather than from a raw 6,000-entry dropdown.

Both applications call the **same** `HybridRecommender` class. Duplicating
fusion logic into each application is how they drift apart and start returning
different answers for the same customer.

---

## 8. Reports

| Report | Contents |
|---|---|
| [Synthetic Data Design & Validation](reports/01_synthetic_data_design_and_validation.md) | schema, engineered properties, 10 validation checks |
| [Evaluation & Benchmarking](reports/02_evaluation_and_benchmarking.md) | methodology, full results, the two failures found |
| [System Architecture & Scalability](reports/03_system_architecture_and_scalability.md) | batch vs real-time, retraining, data growth, monitoring KPIs |
| [Data Risk Assessment](reports/04_data_risk_assessment.md) | 10 risks scored, mitigations, residual exposure |
| [Cost Optimization](reports/05_cost_optimization.md) | measured costs, optimisations applied, projections to 1M users |
| [Governance & Fairness](reports/06_governance_and_fairness.md) | explainability, fairness audit, regulatory position, failure modes |
| [Final Business Recommendations](reports/07_final_business_recommendations.md) | executive summary, 7 recommendations, roadmap |

---

## 9. Tech stack

| Layer | Technology |
|---|---|
| Synthetic data | Faker, NumPy |
| Data processing | pandas |
| Classical ML | scikit-learn (TruncatedSVD, cosine similarity, TF-IDF) |
| Deep learning | PyTorch |
| Evaluation | custom metric implementations |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit |
| Visualisation | matplotlib, seaborn |

**On `surprise`:** named in the brief as an option for matrix factorisation, it
has no wheel for Python 3.13 and fails to build. `TruncatedSVD` provides the
same latent-factor method without the dependency.

---

## 10. Limitations

**Every number in this project comes from synthetic data.** The models learned
the rules the generator was written with — category affinity and segment price
sensitivity — and the metrics measure how well those specific rules were
recovered. No claim is made that they transfer to real customers.

What does transfer is the pipeline, the evaluation methodology and the serving
architecture.

Other known gaps, all documented rather than hidden:

- Cold-start quality is unverifiable offline — no ground truth exists
- The popularity feedback loop is dampened but not broken; closing it needs
  online exploration
- Deleting a customer's rows does not remove their learned NCF embedding
- Ingestion-quality controls (bot filtering, deduplication) are specified but
  not built
- Single time-based split; no confidence intervals reported

---

## Interview summary

> Built an enterprise-grade e-commerce recommendation system on engineered
> synthetic data — 6,000 customers, 2,500 products, 138k interactions with
> deliberately induced sparsity, popularity bias, long tail and cold-start
> cohorts. Benchmarked six approaches under a leak-free time-based split; the
> four-signal hybrid won at NDCG@10 = 0.1095, a 69.8% lift over the popularity
> baseline. The evaluation harness caught two failures that training loss could
> not see: a deep model scoring NDCG 0.0000 because it was trained to predict
> ratings rather than to rank, fixed with negative sampling; and a hybrid losing
> to its own strongest input, fixed by re-weighting on measured evidence. Shipped
> as both a FastAPI service and a Streamlit console sharing one recommendation
> engine, with explanations derived from real score attribution.
