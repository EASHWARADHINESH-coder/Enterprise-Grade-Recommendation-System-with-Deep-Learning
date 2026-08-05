# Governance & Fairness Report

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Why this matters

A recommendation system allocates **visibility**. It decides which products
6,000 customers see and which 2,500 sellers get shelf space. Both sides can be
harmed by a system optimised only for click-through:

- **Customers** can be trapped in narrowing loops, or systematically shown worse
  options than comparable customers.
- **Sellers** can be starved of exposure by a feedback loop that has nothing to
  do with product quality.

Neither harm shows up in NDCG. Both are measurable, and this report measures
them.

---

## 2. Explainability

### 2.1 The three mandated explanation types

The brief requires the system to explain why an item is recommended, which
similar users or items support it, and what content similarity justifies it. All
three are implemented in `models/explainability.py`.

**Design principle:** every explanation is derived from the **actual scores that
produced the ranking**, never from a template. A plausible-sounding sentence
that does not correspond to the real reason is worse than no explanation — it
manufactures false confidence and will eventually contradict the model in front
of a customer.

### 2.2 Signal attribution

Raw per-signal scores would mislead: a signal with a high score but a low fusion
weight contributes little. What is reported is **weight × score** — the share of
the decision each signal is genuinely responsible for.

Example output for a real recommendation:

| Signal | Share |
|---|---|
| item-based collaborative filtering | 51.3% |
| the deep learning model | 38.7% |
| latent-factor collaborative filtering | 7.3% |
| content similarity | 2.7% |

### 2.3 Disclosure of ranking interventions

When an item was promoted by the long-tail boost, the explanation says so:
*"it is an under-exposed product we are surfacing for variety."*

Concealing that would misrepresent a merchandising decision as a personalisation
decision.

### 2.4 An honest limitation we chose not to hide

Similar-customer evidence is frequently **empty**, and the cause is structural
rather than a bug: the re-ranker deliberately promotes long-tail items, and an
item is long-tail precisely because few people have interacted with it.

**The diversity mechanism actively erodes the collaborative evidence available
to justify its own output.**

Measured: for typical long-tail recommendations, 0–1 of the 50 nearest customers
have rated the item at all.

The system reports this plainly:

> "None of your 50 closest customer matches have rated this item — it is a
> low-exposure product surfaced for catalogue variety, so the recommendation
> rests on content and model signals rather than crowd evidence."

The alternative — a vague sentence implying crowd support that does not exist —
would be a straightforward misrepresentation.

### 2.5 Cold-start explanations are visibly different

A new customer receives an honest explanation of the fallback, not the same
confident sentence with different nouns:

> "You are new here, so this is based on your registration profile rather than
> purchase history: it sits in Automotive, the category you selected, and is
> priced at INR 326, which fits the typical Budget Shopper budget."

Every API response also carries a `strategy` field
(`hybrid_fusion` / `cold_start_user_profile` / `global_popularity_fallback`) so
downstream systems can distinguish personalised results from fallbacks
programmatically.

---

## 3. Fairness audit

Method: 240 randomly sampled customers with ≥ 5 interactions, top-10
recommendations each. Full reproduction in
`notebooks/05_explainability_and_governance.ipynb`.

### 3.1 Price allocation across segments

The question is not *whether* recommended price varies by segment — it should,
because purchasing behaviour genuinely varies. The question is whether the
system **amplifies** the gap beyond observed behaviour.

| Segment | Median observed purchase price | Recommended price point |
|---|---|---|
| Budget Shopper | INR 572 | tracks observed |
| Value Seeker | INR 827 | tracks observed |
| Mainstream Buyer | INR 1,395 | tracks observed |
| Premium Buyer | INR 2,724 | tracks observed |
| Luxury Enthusiast | INR 3,361 | tracks observed |

**Assessment:** recommended price ordering follows observed purchasing
behaviour. Recommending a INR 90,000 item to a budget shopper serves nobody —
it wastes a slot and reads as tone-deaf.

**The risk to watch** is amplification over time: if the system recommends only
cheap goods, budget customers only ever buy cheap goods, and the next model
trains on an even narrower band. This is a ratchet, and it is invisible in a
single snapshot.

**Monitoring requirement:** track the ratio of *recommended* price spread to
*observed purchase* price spread. If recommendations diverge faster than
behaviour, the ratchet is turning.

### 3.2 Gender

**`gender` is not a model input.** It is retained in the schema solely to make
this audit possible.

Any measured difference in recommendations by gender therefore comes from
correlated behaviour, not from the model reading the field. This is the right
design — but it does **not** make the system fair by construction. A model can
reproduce a demographic disparity perfectly well through proxies (category
preference, price band) without ever seeing the protected attribute.

**Monitoring requirement:** category-diversity spread across gender groups.
Alarm if it exceeds 2×.

### 3.3 Service quality by activity level

The most consequential fairness axis in any recommender. Light users get worse
recommendations *and* are least likely to tolerate them.

| Activity band | Treatment |
|---|---|
| 0–2 interactions | cold-start fallback (profile-based) |
| 3–10 | full hybrid, but thin collaborative signal |
| 11–40 | full hybrid |
| 40+ | full hybrid, strongest signal |

**Structural inequity:** a customer with 200 interactions gets a materially
better service than one with 5. This is inherent to collaborative filtering, not
a defect in this implementation — but it is a real harm, and it compounds,
because worse recommendations mean less engagement means still worse
recommendations.

**Mitigations in place:**
- Content-based signal, which needs no collaborative history, is in the fusion
  at 10% weight.
- Cold-start fallback uses registration data rather than blind popularity.
- The cold-start threshold is 3 interactions, not 1 — a customer with a single
  click carries almost no signal, and treating them as "warm" produces
  confidently wrong recommendations from one data point.

### 3.4 Item-side fairness (seller equity)

Catalogue coverage at K=10:

| Model | Coverage | Items never recommended |
|---|---|---|
| Popularity | 1.3% | 2,468 of 2,500 |
| NCF | 1.3% | 2,468 |
| SVD | 9.6% | 2,260 |
| Hybrid | 20.2% | 1,996 |
| Item-based CF | 22.2% | 1,946 |
| Content (TF-IDF) | 75.4% | 615 |

**Even the best-covering model leaves 615 products never recommended to
anybody.** The popularity baseline leaves 2,468 — 99% of the catalogue invisible.

For a marketplace, this is a commercial and contractual problem, not just an
ethical one: sellers paying for placement on a platform where the recommender
never surfaces their inventory have a legitimate grievance.

**Mitigations in place:** inverse-propensity penalty (α = 0.20), long-tail boost
(1.15×), coverage tracked as a first-class metric, and a coverage floor in the
model-promotion gate so a challenger cannot ship if it collapses onto the head.

---

## 4. The popularity feedback loop

The central structural risk. Recommending popular items makes them more popular,
which makes the model recommend them harder.

**Current mitigation** is a static inverse-propensity correction. This dampens
the loop but does not break it, because the system never explores: it can only
promote items it already has some signal for.

**Properly closing this requires online exploration** — epsilon-greedy or
Thompson sampling — which needs live traffic and is out of scope for an offline
prototype.

**This is a known, unclosed gap, not an oversight.**

---

## 5. Regulatory position

Recommendation systems that shape what customers see fall under transparency
obligations in a growing number of jurisdictions (EU DSA Article 27, and
consumer-protection regimes elsewhere).

| Requirement | Status |
|---|---|
| Explain main ranking parameters | **Met** — signal attribution exposed via API and dashboard |
| Disclose paid placement | N/A — no paid placement in this system |
| Offer a non-profiled option | **Not implemented** — would need a popularity-only mode |
| Right to erasure | **Partially** — deleting rows does not remove learned embeddings |
| Data minimisation | **Not met** — `name` is stored but unused by any model |

### The erasure gap is worth stating precisely

Deleting a customer's interaction rows does **not** remove the representation
the NCF has learned of them. The embedding table retains a vector trained on
their behaviour. Genuine erasure requires either retraining or explicit
embedding removal, and neither is currently implemented.

This is the most commonly missed privacy obligation in embedding-based
recommenders, and it is a real gap here.

---

## 6. Where the system fails

Stated plainly, because a governance report that lists only successes is not a
governance report.

| Failure mode | Cause | Status |
|---|---|---|
| Empty crowd evidence for long-tail items | diversity boost promotes rarely-rated items | disclosed in explanation text |
| Explicit NCF cannot rank | trained only on observed pairs | production uses implicit variant |
| Light users get worse service | inherent to collaborative filtering | partially mitigated |
| 615 items never recommended | head concentration | partially mitigated |
| Cold-start quality unverifiable | no ground truth | needs online A/B test |
| Popularity loop | no exploration | **unclosed** |
| Embedding erasure | no removal mechanism | **unclosed** |
| Filter-bubble ratchet | no longitudinal tracking | **unmonitored** |

---

## 7. On the synthetic data

Every fairness number in this report is computed on generated data.

The generator embedded the relationship between segment and price *by design*.
So when the audit finds that recommended price tracks segment, it is partly
measuring the generator, not only the model. A real fairness audit requires real
customer data and, ideally, an independent reviewer.

**What this audit does demonstrate** is that the measurement apparatus exists,
runs, and produces interpretable output — which is what can be built before real
data is available.

---

## 8. Recommendations

**Before production:**
1. Drop `name` from the modelling path (data minimisation)
2. Implement an embedding-erasure procedure
3. Add a non-profiled browsing mode
4. Stand up longitudinal filter-bubble monitoring

**Ongoing:**
5. Monitor recommended-vs-observed price spread ratio by segment
6. Alarm on coverage < 15% and on category-diversity spread > 2× across groups
7. Re-run this audit on every model promotion, not annually
8. Add exploration to break the popularity loop
