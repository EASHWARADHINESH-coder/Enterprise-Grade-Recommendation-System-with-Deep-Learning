# Enterprise-Grade Recommendation System with Deep Learning

**Domain:** E-Commerce / Retail
**Author:** Eashwaradhinesh K

An end-to-end recommendation engine that combines collaborative filtering,
content-based NLP, and a PyTorch deep learning model into a single ranked
output — served through both a REST API and an interactive dashboard.

---

## 1. Problem Statement and Overview

### The business problem

You are a Senior Data Scientist at a digital commerce platform. The company is
in an **early-stage / new-market position**, which creates a specific difficulty:

- Real user interaction data is limited or unavailable
- Models must be prototyped on synthetic data
- The system design must survive future scale

The company still needs a working recommendation system **before** live traffic
exists, because recommendations drive engagement, retention and revenue.

### What this project builds

A complete recommendation pipeline that:

1. Generates realistic synthetic e-commerce data
2. Builds baseline and deep learning recommenders
3. Handles cold-start users and products
4. Evaluates recommendation quality honestly
5. Serves recommendations through an application layer

### Why it is hard

Recommendation is not a normal prediction task. The data has properties that
break naive approaches:

| Property | In this dataset | Why it matters |
|---|---|---|
| **Sparsity** | 99.08% of cells empty | Direct similarity has almost nothing to work with |
| **Popularity bias** | Top 10% of items hold 62.9% of demand | Models collapse onto best-sellers |
| **Long tail** | Gini 0.716 | Most inventory is nearly invisible |
| **Cold-start users** | 480 (8.0%) | No history to personalise from |
| **Cold-start items** | 255 (10.2%) | No sales history to learn from |

### Headline result

| Model | NDCG@10 | Catalogue coverage |
|---|---|---|
| **Hybrid (4-signal fusion)** | **0.1095** | 20.2% |
| Item-based CF | 0.1044 | 22.2% |
| NCF (Deep Learning) | 0.0663 | 1.3% |
| Popularity baseline | 0.0645 | 1.3% |
| SVD | 0.0462 | 9.6% |
| Content (TF-IDF) | 0.0116 | 75.4% |

**The hybrid is best on every ranking metric — +69.8% over the popularity
baseline**, measured on a time-based split with no interaction leakage.

---

## 2. End-to-End Workflow

```
STEP 1 - GENERATE DATA
  6,000 users | 2,500 items | 138,345 interactions
  Engineered: sparsity, popularity bias, long tail, cold start
                          |
                          v
STEP 2 - VALIDATE (EDA)
  Prove the engineered properties actually landed
  10/10 validation checks pass
                          |
                          v
STEP 3 - FEATURE ENGINEERING
  rating matrix | normalised matrix | implicit matrix
  popularity features | content features | user profiles
                          |
                          v
STEP 4 - TRAIN MODELS
  Popularity -> User-CF -> Item-CF -> SVD -> TF-IDF -> NCF
                          |
                          v
STEP 5 - FUSE INTO HYBRID
  Weighted blend -> retrieve top 200 -> re-rank for diversity
  Cold-start fallback when history is insufficient
                          |
                          v
STEP 6 - EVALUATE
  Time-based split | leakage assertions | models refit on train only
  Precision@K | Recall@K | MAP@K | NDCG@K + coverage & diversity
                          |
                          v
STEP 7 - EXPLAIN & SERVE
  Why this item | similar users/items | content justification
  FastAPI (port 8000) + Streamlit (port 8501)
```

### Run the whole pipeline

```bash
# 1. Generate synthetic data (~10 seconds)
python synthetic_data/generate_synthetic_data.py

# 2. Validate the data (~30 seconds, writes 7 figures)
python notebooks/eda_validation.py

# 3. Build features (~30 seconds)
python src/preprocessing.py

# 4. Build content model (~20 seconds)
python src/content_based_nlp.py

# 5. Train baselines (~60 seconds)
python models/baseline_recommenders.py

# 6. Apply CF improvements (~40 seconds)
python src/collaborative_filtering.py

# 7. Train deep learning models (~8 minutes on CPU)
python models/ncf_recommender.py

# 8. Benchmark everything (~6 minutes)
python notebooks/recommendation_evaluation.py
```

Total: roughly **16 minutes** on CPU. The generator is seeded, so a rebuild
reproduces every number exactly.

### Run the applications

```bash
# Backend API  ->  http://localhost:8000/docs
python app/app_fastapi.py
```

```bash
# Frontend dashboard  ->  http://localhost:8501
streamlit run app/streamlit_recommendation_dashboard.py
```

---

## 3. Stack Used

| Layer | Technology | Used for |
|---|---|---|
| **Language** | Python 3.13 | everything |
| **Data generation** | Faker, NumPy | synthetic users, items, interactions |
| **Data processing** | pandas | matrices, feature tables, aggregation |
| **Classical ML** | scikit-learn | TruncatedSVD, cosine similarity, TF-IDF |
| **Deep learning** | PyTorch 2.12 | Neural Collaborative Filtering |
| **NLP** | scikit-learn TF-IDF | item descriptions to content vectors |
| **API** | FastAPI + Uvicorn | REST service with auto-generated docs |
| **Validation** | Pydantic | request/response schemas |
| **Dashboard** | Streamlit | operator console |
| **Visualisation** | Matplotlib, Seaborn | EDA figures |
| **Notebooks** | Jupyter | the five mandated notebooks |
| **Version control** | Git | source management |

**A note on `surprise`:** the business case names it as an option for matrix
factorisation. It has no wheel for Python 3.13 and fails to build, so
scikit-learn's `TruncatedSVD` provides the same latent-factor method without the
dependency.

---

## 4. Project Structure

```
Enterprise-Grade-Recommendation-System-with-Deep-Learning/
|
+-- synthetic_data/
|   +-- generate_synthetic_data.py      Creates users, items, interactions
|
+-- data/
|   +-- raw/                            Generated CSVs (committed)
|   |   +-- users.csv                   6,000 customers
|   |   +-- items.csv                   2,500 products
|   |   +-- interactions.csv            138,345 events
|   +-- processed/                      Model artifacts (regenerated, not committed)
|
+-- src/                                Core reusable logic
|   +-- data_loader.py                  Reads the three raw tables
|   +-- preprocessing.py                Builds matrices and feature tables
|   +-- content_based_nlp.py            TF-IDF content recommender
|   +-- collaborative_filtering.py      CF improvements + re-ranking
|   +-- evaluation_metrics.py           Precision/Recall/MAP/NDCG from scratch
|
+-- models/                             Model implementations
|   +-- baseline_recommenders.py        The 4 mandated baselines
|   +-- ncf_recommender.py              PyTorch Neural Collaborative Filtering
|   +-- hybrid_recommender.py           Score fusion service class
|   +-- explainability.py               Why an item was recommended
|
+-- app/                                Application layer
|   +-- app_fastapi.py                  REST API (port 8000)
|   +-- streamlit_recommendation_dashboard.py   Dashboard (port 8501)
|
+-- notebooks/                          The 5 mandated notebooks
|   +-- 01_advanced_eda.ipynb
|   +-- 02_feature_engineering.ipynb
|   +-- 03_ml_pipeline.ipynb
|   +-- 04_deep_learning_model.ipynb
|   +-- 05_explainability_and_governance.ipynb
|   +-- eda_validation.py               Script version of the EDA
|   +-- recommendation_evaluation.py    The benchmark harness
|
+-- reports/                            Written deliverables
|   +-- 01_synthetic_data_design_and_validation.md
|   +-- 02_evaluation_and_benchmarking.md
|   +-- 03_system_architecture_and_scalability.md
|   +-- 04_data_risk_assessment.md
|   +-- 05_cost_optimization.md
|   +-- 06_governance_and_fairness.md
|   +-- 07_final_business_recommendations.md
|   +-- 08_live_evaluation_walkthrough.md
|   +-- figures/                        7 EDA plots
|
+-- requirements.txt
+-- README.md
```

**Design rule:** all recommendation logic lives in `hybrid_recommender.py`. The
API and the dashboard are transport layers only — they both load the same
service class, so they can never return different answers for the same customer.

---

## 5. Module Guide

Every module explained in plain language.

### `synthetic_data/generate_synthetic_data.py`

**Creates the dataset.**

This does not just emit random rows. A recommender trained on uniform random
data learns nothing, so the generator deliberately builds in the properties that
make recommendation realistic:

- **Popularity bias** — item exposure follows a Zipf power law
- **Long tail** — most products get very few interactions
- **Cold start** — 8% of users and 10% of items are held at near-zero activity
- **Latent structure** — customers prefer their category *and* their segment's
  price band, and these two interact non-linearly

Every interaction is an engagement funnel: click → add to cart → purchase, with
an optional star rating. A single hidden "affinity" score drives all four, so the
signals stay consistent — nobody rates 5 stars something they bounced off in two
seconds.

### `src/data_loader.py`

**Reads the three CSV files.**

Small but important: it parses `timestamp` into a real datetime so the
time-based split cannot silently degrade into comparing strings. Paths are
computed from the file's own location, so the project runs from any folder or
drive.

### `src/preprocessing.py`

**Turns raw events into the structures models need.**

| Output | What it is |
|---|---|
| `user_item_matrix` | who rated what |
| `normalized_user_item_matrix` | ratings with each user's average removed |
| `implicit_feedback_matrix` | who carted or bought what |
| `item_popularity_features` | exposure, conversion rate, long-tail flag |
| `item_content_features` | price, category, brand as numbers |
| `user_profile_features` | segment, age, gender as numbers |

**Why mean-centre ratings?** A generous rater and a harsh rater who rank items
identically look like completely different people to a model. Subtracting each
user's average removes that personal offset, so the model learns *relative*
preference.

### `src/content_based_nlp.py`

**Recommends by product text, not by who bought it.**

Runs TF-IDF over the description, tags, category and brand, then compares items
by cosine similarity. This is the answer to **item cold start**: a product listed
this morning has a full description, therefore a full content vector, before a
single customer has clicked it.

It also exposes the exact matching terms between two items, which turns
"similarity 0.83" into an explanation a human can audit.

### `src/collaborative_filtering.py`

**Fixes the ways textbook collaborative filtering fails.**

| Problem | Treatment |
|---|---|
| Ratings missing on ~31% of rows | implicit CF from cart/purchase |
| Popularity bias | inverse-propensity penalty |
| Long-tail starvation | explicit boost for under-exposed items |
| Cold-start users | fallback to registration profile |

The re-ranker uses a **two-stage** design: retrieve the top 200 by relevance
first, *then* apply the diversity correction inside that pool. Applying it across
the whole catalogue lets an obscure irrelevant item outrank a genuinely good one
purely for being obscure.

### `src/evaluation_metrics.py`

**The ranking metrics, written from scratch.**

Precision@K, Recall@K, MAP@K and NDCG@K, plus beyond-accuracy metrics: catalogue
coverage, long-tail share, novelty, and intra-list diversity.

**Why coverage matters:** a model can post excellent precision while only ever
recommending 32 items out of 2,500. Commercially that is a failure — the rest of
the inventory is invisible and will never sell.

### `models/baseline_recommenders.py`

**The four mandated baselines.**

1. **Popularity** — most-interacted items, blended with rating and conversion
2. **User-based CF** — what similar customers bought
3. **Item-based CF** — items similar to what you already rated
4. **SVD** — compresses the sparse matrix into 50 latent factors

These exist to set the bar. A deep model that cannot beat a popularity list is
not worth its serving cost.

### `models/ncf_recommender.py`

**The PyTorch deep learning model.**

```
user_id -> embedding (32d) --+
                              +--> concat --> MLP [128, 64, 32] --> score
item_id -> embedding (32d) --+
```

**Why not just use SVD?** SVD scores an item as a dot product — a fixed, linear
combination. This dataset was built with a non-linear rule: a customer buys when
the category matches **and** the price fits. A dot product cannot represent an
"and". An MLP over concatenated embeddings can.

Two variants are trained:
- **Explicit** — predicts the star rating (MSELoss)
- **Implicit** — predicts engagement (BCEWithLogitsLoss, with negative sampling)

Only the implicit one is used for ranking. Section 7 explains why.

### `models/hybrid_recommender.py`

**Combines everything into one ranked list.**

| Signal | Weight | What it contributes |
|---|---|---|
| Item-based CF | 0.45 | strongest single signal |
| Deep learning (NCF) | 0.35 | non-linear interaction patterns |
| Latent-factor CF (SVD) | 0.10 | generalises to sparse users |
| Content (TF-IDF) | 0.10 | catalogue reach and cold start |

Each covers the others' blind spots. Weights were chosen by **measurement**, not
intuition — the sweep is in
[reports/02_evaluation_and_benchmarking.md](reports/02_evaluation_and_benchmarking.md).

Packaged as a loadable service class so the API and dashboard share one
implementation.

### `models/explainability.py`

**Answers "why was this recommended?"**

Produces the three mandated explanation types:

1. **Why this item** — which signal drove the decision, and by how much
2. **Supporting evidence** — the customer's own comparable purchases, and how
   many similar customers rated it well
3. **Content justification** — the actual matching description terms

Every explanation is derived from the **real scores that produced the ranking**,
never from a template. A plausible-sounding sentence that does not match the
actual reason is worse than no explanation at all.

### `app/app_fastapi.py`

**The REST API.**

| Endpoint | Returns |
|---|---|
| `GET /recommend/{user_id}` | top-N with explanations and signal breakdown |
| `GET /similar-items/{item_id}` | content-similar products |
| `GET /explain/{user_id}/{item_id}` | full three-part explanation |
| `GET /users/{user_id}` | customer profile |
| `GET /items/{item_id}` | product detail |
| `POST /recommend/batch` | many customers in one call |
| `GET /health` | readiness and loaded-model status |

Every response carries a `strategy` field, so a caller can always tell a
personalised result from a cold-start fallback.

### `app/streamlit_recommendation_dashboard.py`

**The operator console.**

Five tabs: **Recommendations**, **Explainability**, **Similar Items**,
**Cold-Start Demo**, and **Customer History**.

Built for the person who has to answer *"why did the site show that to this
customer?"* — a merchandiser, or an analyst investigating a complaint.

---

## 6. Results

### Ranking metrics (K = 10, 800 customers, held-out test period)

| Model | Precision | Recall | MAP | **NDCG** | Hit Rate |
|---|---|---|---|---|---|
| **Hybrid** | **0.0628** | **0.1372** | **0.0593** | **0.1095** | **0.3850** |
| Item-based CF | 0.0579 | 0.1241 | 0.0593 | 0.1044 | 0.3575 |
| NCF | 0.0375 | 0.0858 | 0.0349 | 0.0663 | 0.2812 |
| Popularity | 0.0375 | 0.0843 | 0.0336 | 0.0645 | 0.2825 |
| SVD | 0.0278 | 0.0511 | 0.0230 | 0.0462 | 0.2050 |
| Content | 0.0073 | 0.0161 | 0.0049 | 0.0116 | 0.0675 |

### Deep learning model

| Variant | Metric | Value |
|---|---|---|
| Explicit (rating prediction) | RMSE | 0.9108 |
| Explicit | MAE | 0.7449 |
| Implicit (engagement) | Accuracy | 0.8583 |
| Implicit | F1 | 0.5816 |

### Data validation

**10 of 10 checks pass** — volume, sparsity, popularity bias, long tail,
cold start, rating realism, and category preference.

---

## 7. Problems Faced in the Project

Every one of these was found by running and measuring the system, not by
inspection. They are documented because the fixes are the most useful part of
the project.

### Problem 1 — The deep learning model could not rank at all

**Symptom:** NCF scored **NDCG@10 = 0.0000** with catalogue coverage of 0.006.
It returned the same ~14 items to every single customer.

**Diagnosis:** this was not a tuning problem. The explicit model minimises error
against *observed star ratings*. It had never once been shown an item a customer
did **not** interact with, so scoring the unseen catalogue was completely outside
its training distribution. It fell back on a global item bias — identical for
everybody.

**Fix:** train an implicit variant with **negative sampling** — 4 items the
customer never touched, per observed positive. Now "would this customer engage or
not?" *is* the training objective.

| Variant | NDCG@10 | Coverage |
|---|---|---|
| Explicit | 0.0000 | 0.006 |
| Implicit + negatives | **0.0576** | 0.013 |

**Lesson:** a model can be accurate on the metric you trained it on and useless
at the task you actually need. RMSE 0.91 told us nothing about ranking quality.

### Problem 2 — The hybrid was worse than its own components

**Symptom:** Hybrid NDCG@10 = 0.0760, but Item-based CF alone scored 0.1044. The
combined model was **losing to one of its own inputs**.

**Diagnosis:** the hybrid's collaborative slot was filled by SVD (0.0462), the
*weakest* collaborative signal, while item-based CF — the strongest model in the
whole system — was not in the fusion at all.

**Fix:** re-ran the weight sweep and rebuilt the fusion around measured
performance.

| Configuration | NDCG@10 |
|---|---|
| Original (svd + content + ncf) | 0.0947 |
| item_cf swapped in for svd | 0.1282 |
| **item_cf .45 / svd .10 / content .10 / ncf .35** | **0.1293** |

### Problem 3 — The diversity re-ranker destroyed relevance

**Symptom:** after fixing a sign bug, the re-ranker pushed the top-10 mean item
exposure from 518 interactions down to **1**, with 100% long-tail share. The
system was recommending obscurity for its own sake.

**Diagnosis — two separate bugs:**

1. **Sign inversion.** SVD runs on the mean-centred matrix, so its scores are
   negative for many items. Multiplying a negative score by a penalty below 1
   moves it *up* toward zero — inverting the correction for every negatively
   scored item.
2. **Wrong scope.** The penalty `1 / (1 + log(count))` spans a ~5x range, wider
   than the spread of the relevance scores it multiplies, so it dominated the
   ranking outright.

**Fix:** normalise scores to [0, 1] *before* multiplying, switch to a gentler
power-form penalty with a tunable strength, and apply it only to the top 200
candidates rather than the whole catalogue.

### Problem 4 — The API was silently serving without the deep model

**Symptom:** none. That was the problem.

**Diagnosis:** the API loaded NCF artifacts from `app/saved_models/` — a
directory that never existed — inside a bare `try/except`. It started, reported
healthy, and served every recommendation with the deep learning weight
effectively set to zero. It also called `.eval()` on a `state_dict`, which would
have failed even with the correct path.

**Fix:** artifacts now load from the right location, a missing model produces a
**loud warning** naming the file and the command to fix it, `/health` reports
`ncf_available` explicitly, and the fusion renormalises its weights so a degraded
hybrid is at least internally consistent.

### Problem 5 — The Gini coefficient came out negative

**Symptom:** the long-tail validation check reported **Gini = −0.716**, which is
mathematically impossible.

**Diagnosis:** the Gini integral is defined against the *ascending* Lorenz curve.
The plotted concentration curve ("top X% of items drive Y% of demand") is sorted
descending and sits above the diagonal. Integrating that one returns the value
sign-flipped. The magnitude was right, which is exactly why it was easy to miss.

**Fix:** compute the Gini from a separately sorted ascending curve. Result:
**0.716**, and the check passes.

### Problem 6 — The `.gitignore` had never worked

**Symptom:** 403 MB of model artifacts and compiled Python files were tracked in
git despite being listed in `.gitignore`.

**Diagnosis:** the file was saved as **UTF-16**. Git only parses UTF-8, so every
rule in it had been silently inert since the repository was created.

**Fix:** rewritten as UTF-8, artifacts untracked. The repository went from a
would-be 403 MB down to **10.9 MB**.

### Problem 7 — The code could not run on this machine

**Symptom:** every module failed on import.

**Diagnosis:** all paths were hardcoded to an absolute directory that did not
exist on the current machine. A duplicate copy of every source file existed in
`reports/` purely to satisfy imports.

**Fix:** every module now computes its paths from its own file location using
`os.path`, so the project runs from any folder or drive. The duplicate files were
deleted.

---

## 8. Conclusion and Future Development

### Conclusion

This project delivers a complete recommendation system for an e-commerce
platform with no real interaction data — from synthetic data generation through
to a served, explainable API.

**What the evidence supports:**

- The **hybrid is the best model**, at NDCG@10 = 0.1095 and **+69.8% over the
  popularity baseline**, measured with a time-based split and leakage assertions.
- Fusing signals genuinely helps. The hybrid beats its strongest single component
  (0.1095 vs 0.1044) because each signal covers the others' blind spots.
- **Cold start is handled**, not ignored. New customers are served from their
  registration profile, and new products from their description — both with
  honest explanations that say so.
- The system is **reproducible**. A from-scratch rebuild reproduces every metric
  exactly, because the generator is seeded.

**An honest finding worth stating plainly:** classical item-based CF beats the
deep model on its own (0.1044 vs 0.0663), at a fraction of the training and
inference cost. At this data scale — 138k interactions across 6,000 customers —
there is not enough signal to justify 278,817 parameters. The deep model earns
its place *inside* the hybrid, but a recommendation to deploy NCF *instead of*
item-based CF would not be supported by this evidence.

**The main limitation:** every number here comes from synthetic data. The models
learned the rules the generator was written with. What transfers to production is
the **architecture, the evaluation methodology, and the serving design** — not
the metric values themselves.

### Future Development

**Short term**
- Add exploration (epsilon-greedy or Thompson sampling) to break the popularity
  feedback loop
- Build ingestion validation: schema contracts, bot filtering, deduplication
- Stand up the retraining scheduler with the documented drift triggers

**Medium term**
- Replace exact similarity with an approximate nearest-neighbour index (FAISS or
  HNSW) — user-user CF is O(users²) and is the first thing that breaks at scale
- Move to two-stage retrieval: ANN candidate generation, then neural re-ranking
  of a shortlist, so the full catalogue is never scored online
- Run an online A/B test — the only way to validate cold-start quality, which has
  no offline ground truth by definition

**Longer term**
- Swap TF-IDF for transformer embeddings (sentence-transformers) to capture
  meaning rather than word overlap
- Add sequence models (GRU4Rec, SASRec) to use interaction *order*, which the
  current design discards entirely
- Introduce a feature store and streaming ingestion so features are computed once
  and shared between training and serving
- Add Docker packaging and CI for reproducible deployment

**Data**
- Replace synthetic data with real interaction logs and re-tune every
  hyperparameter — the current values are optimal for this generator, not for
  e-commerce in general
- Add seasonality, promotions, stock levels and delivery time, none of which the
  current dataset models

---

## Documentation

| Report | Covers |
|---|---|
| [Synthetic Data Design & Validation](reports/01_synthetic_data_design_and_validation.md) | how the data was engineered and verified |
| [Evaluation & Benchmarking](reports/02_evaluation_and_benchmarking.md) | full methodology and results |
| [System Architecture & Scalability](reports/03_system_architecture_and_scalability.md) | batch vs real-time, retraining, monitoring |
| [Data Risk Assessment](reports/04_data_risk_assessment.md) | what could go wrong and what was done |
| [Cost Optimization](reports/05_cost_optimization.md) | compute and storage economics |
| [Governance & Fairness](reports/06_governance_and_fairness.md) | equitable treatment across segments |
| [Final Business Recommendations](reports/07_final_business_recommendations.md) | what to deploy and why |
| [Live Evaluation Walkthrough](reports/08_live_evaluation_walkthrough.md) | section-by-section defence document |
