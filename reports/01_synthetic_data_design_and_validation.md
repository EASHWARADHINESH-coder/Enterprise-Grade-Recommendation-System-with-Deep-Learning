# Synthetic Data Design & Validation Report

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Why synthetic data at all

The business case places us at a digital platform in an early-stage or new-market
position, where real interaction data is limited, unavailable, or restricted. The
organisation still needs to design, prototype and validate a recommendation
system before committing to full-scale deployment.

That framing sets the bar for this dataset. It is not enough to emit rows with
plausible column names. A recommender trained on uniformly random data learns
nothing, and an exploratory analysis of uniformly random data cannot demonstrate
any of the properties the brief asks us to handle. The generator therefore has
to *engineer* the specific pathologies that make recommendation hard.

---

## 2. Schema

The three mandated tables, implemented exactly as specified on page 4 of the
brief.

### `users.csv` — 6,000 rows

| Field | Type | Notes |
|---|---|---|
| `user_id` | int | primary key |
| `name` | str | Faker-generated |
| `age` | int | correlated with segment, not uniform |
| `gender` | str | Male / Female / Other |
| `location` | str | 16 Indian metro / tier-1 cities |
| `user_segment` | str | **the commercially meaningful field** |
| `preferred_category` | str | declared interest, drives cold start |
| `signup_date` | date | supports tenure features |

### `items.csv` — 2,500 rows

| Field | Type | Notes |
|---|---|---|
| `item_id` | int | primary key |
| `title` | str | brand + adjective + subcategory + model code |
| `category` | str | 10 departments |
| `subcategory` | str | 8 per category |
| `brand` | str | 8 per category |
| `description` | str | **free text, consumed by TF-IDF** |
| `price` | float | per-category log-normal |
| `content_tags` | str | pipe-separated attributes |
| `base_quality` | float | intrinsic quality, drives ratings |
| `launch_date` | date | cold-start items launched recently |
| `is_cold_start_item` | int | reserved flag |
| `price_percentile` | float | for segment price matching |

### `interactions.csv` — 138,345 rows

| Field | Type | Notes |
|---|---|---|
| `user_id`, `item_id` | int | composite key, unique per pair |
| `click` | int | always 1 — every row is at minimum a page view |
| `view_time_seconds` | float | dwell time, scales with affinity |
| `add_to_cart` | int | 0/1 |
| `purchase` | int | 0/1, reachable only from cart |
| `quantity`, `revenue` | int, float | basket economics |
| `rating` | float | 1–5, **NaN when not rated** |
| `timestamp` | datetime | 18-month window |
| `implicit_feedback` | int | cart OR purchase |

All volumes clear the mandated minimums (5,000 / 2,000 / 100,000).

---

## 3. The engagement funnel

The brief lists `rating / click / watch_time / purchase` as the interaction
signals. In a retail domain these are not independent columns — they are stages
of one funnel:

```
click (100%)  ->  add_to_cart (47.5%)  ->  purchase (16.4%)
                          |
                     rating (68.5% of all rows)
```

A single latent **affinity** score per (user, item) pair drives all four
outcomes. This matters: it keeps the signals mutually consistent, so a customer
never rates five stars something they bounced off in two seconds. Generating the
columns independently would produce contradictory rows and a dataset no model
could learn a coherent pattern from.

---

## 4. Engineered properties

### 4.1 Popularity bias — Zipf exposure

Item exposure follows a **shifted** power law:

```
exposure(rank) = 1 / (rank + 25) ** 1.25
```

The offset is not cosmetic. With a pure `1/rank^alpha` law, the top item's
expected exposure exceeded the entire user base. Because a user can only
interact with an item once, the head *saturated* — the most popular items were
seen by nearly every customer, which destroys their discriminative value
entirely. Measured before the fix, the top item drew 4,299 interactions across a
6,000-user base (72% of all customers).

| Parameter | Top item as % of user base | Top 10% of items' share |
|---|---|---|
| `alpha=1.05, offset=0` | 327% (saturating) | 76.6% |
| `alpha=1.25, offset=25` | 32% | 65.7% |

**Result:** the top 10% of items hold **62.9%** of all interactions — a textbook
Pareto shape with a believable head.

### 4.2 Long tail

**Gini coefficient: 0.716.**

A note on computing this: the Gini integral is defined against the *ascending*
Lorenz curve. The concentration curve normally plotted ("top X% of items drive
Y% of demand") is sorted descending and sits above the diagonal. Integrating
that one instead returns the value with the sign flipped — an easy error to miss
because the magnitude looks correct. The first version of the EDA reported
−0.716 for exactly this reason.

**80.1%** of the catalogue sits at or below the 80th percentile of demand.

### 4.3 Sparsity

| | |
|---|---|
| Possible user-item cells | 15,000,000 |
| Observed interactions | 138,345 |
| **Sparsity** | **99.08%** |

Even the densest 60×60 corner (most active users × most popular items) is mostly
empty. This is what makes naive similarity fail and latent-factor methods
necessary.

### 4.4 Cold start

Deliberately carved out so the fallback logic is *testable*:

| Cohort | Count | Share |
|---|---|---|
| Users with < 3 interactions | 480 | 8.0% |
| Items with < 3 interactions | 255 | 10.2% |

Cold-start items are additionally given launch dates inside the last 30 days, so
they are coherent as "newly listed inventory" rather than arbitrarily starved.

### 4.5 Latent structure

Two genuine latent factors the models are meant to recover:

**Category affinity** — 60.7% of purchases fall inside the customer's declared
category, against ~10% under random choice across 10 categories.

**Segment price sensitivity** — median purchase price rises monotonically with
segment:

| Segment | Median purchase price |
|---|---|
| Budget Shopper | INR 572 |
| Value Seeker | INR 827 |
| Mainstream Buyer | INR 1,395 |
| Premium Buyer | INR 2,724 |
| Luxury Enthusiast | INR 3,361 |

These two factors interact **non-linearly** — a customer buys when the category
matches *and* the price fits. A dot product cannot represent a conjunction,
which is precisely the gap the neural model exists to fill.

### 4.6 Temporal structure

Interactions span 2025-01-07 to 2026-06-30, weighted toward recent months via a
Beta(1.6, 1) recency draw. The final 90 days hold **30.1%** of all interactions,
which is what makes holding them out a meaningful test rather than an arbitrary
cut. Interactions can never predate an item's launch date.

---

## 5. Rating realism

Real marketplace ratings are **J-shaped** — the mean sits near 4 and five-star
reviews dominate. A symmetric distribution centred on 3 is the classic
give-away of synthetic data.

| Stars | Count | Share |
|---|---|---|
| 1 | 297 | 0.3% |
| 2 | 5,523 | 5.8% |
| 3 | 25,325 | 26.7% |
| 4 | 35,016 | 36.9% |
| 5 | 28,663 | 30.2% |

**Mean 3.91, with 67% at 4–5 stars.** The first calibration produced a mean of
3.10 with a symmetric shape; the intercept and slope were retuned to reproduce
the observed skew.

Ratings are present on 68.5% of rows, and are far denser on purchases (88%) than
on browse-only interactions (65%) — reflecting that buyers review and browsers
rarely do.

---

## 6. Documented deviations from reality

Two calibration choices deliberately depart from real-world figures. Both are
trade-offs, not oversights.

**Conversion rate.** Real e-commerce converts at roughly 2–3% of sessions; this
dataset runs at 16.4%. At a realistic 3%, a 138k interaction log yields only
~4,000 purchases across 6,000 customers — far too sparse to train an implicit
model or to build a test set with enough relevant items per user to compute
stable ranking metrics. The higher rate buys usable signal density.

**Cart rate.** Similarly, 47.5% against a real-world 10–15%. The implicit
feedback matrix is built from cart-or-purchase, so this rate directly determines
how much positive signal the NCF has to learn from.

Both would need recalibration before any claim about absolute conversion
performance. Neither affects the *relative* model comparison, which is what this
project is actually measuring.

---

## 7. Validation results

Run `python notebooks/eda_validation.py`:

| Check | Result |
|---|---|
| Volume ≥ 5,000 users | **PASS** (6,000) |
| Volume ≥ 2,000 items | **PASS** (2,500) |
| Volume ≥ 100,000 interactions | **PASS** (138,345) |
| Sparsity > 95% | **PASS** (99.08%) |
| Popularity bias: top 10% hold > 40% | **PASS** (62.9%) |
| Long tail present (Gini > 0.5) | **PASS** (0.716) |
| Cold-start users present | **PASS** (480) |
| Cold-start items present | **PASS** (255) |
| Ratings skew high (mean > 3.5) | **PASS** (3.91) |
| Category preference real (> 30%) | **PASS** (60.7%) |

**All ten checks pass.**

Figures written to `reports/figures/`:

1. `01_user_activity_distribution.png` — mandatory plot
2. `02_item_popularity_distribution.png` — mandatory plot
3. `03_interaction_matrix_sparsity.png` — mandatory plot
4. `04_cold_start_distribution.png`
5. `05_engagement_funnel.png`
6. `06_latent_structure.png`
7. `07_temporal_coverage.png`

---

## 8. Reproducibility

Every random draw comes from a single seeded generator
(`np.random.default_rng(42)`, `Faker.seed(42)`). Re-running
`synthetic_data/generate_synthetic_data.py` reproduces the dataset exactly.

All tuning parameters are named constants at the top of that file, with the
measured effect of each documented in the comments beside it.
