# Data Risk Assessment

**Enterprise-Grade Recommendation System with Deep Learning**
Domain: E-Commerce / Retail

---

## 1. Purpose

This report identifies what could go wrong with the data underpinning this
recommendation system, how likely each risk is, what it would cost, and what has
been done about it. It covers both the synthetic data used for this prototype
and the real data that would replace it in production.

Risks are scored **Likelihood × Impact**, each Low / Medium / High.

---

## 2. Risk register — synthetic data (this prototype)

### R1. Synthetic data does not reflect real customer behaviour
**Likelihood: High · Impact: High · Priority: CRITICAL**

The models have learned the rules the generator was written with — category
affinity and segment price sensitivity. Every metric in this project measures
how well those *specific, known* rules were recovered. Real customer behaviour is
messier, non-stationary, and driven by factors absent here entirely: brand
loyalty, seasonality, promotions, delivery speed, reviews, social proof.

**Consequence:** a model that scores NDCG@10 = 0.1095 here could perform
arbitrarily worse on real data. The number does not transfer.

**Mitigation in place:**
- Stated explicitly in every report and in the governance notebook.
- Architecture and evaluation methodology are the transferable deliverables, not
  the metric values.
- The pipeline is data-agnostic: swapping in real CSVs with the same schema
  requires no code change.

**Residual risk: HIGH — irreducible without real data.**

---

### R2. Deliberate calibration deviations
**Likelihood: Certain · Impact: Medium · Priority: HIGH**

Two parameters knowingly depart from reality:

| Parameter | This dataset | Real e-commerce |
|---|---|---|
| Purchase conversion | 16.4% of clicks | 2–3% of sessions |
| Add-to-cart rate | 47.5% of clicks | 10–15% |

**Why:** at a realistic 3% conversion, a 138k interaction log yields ~4,000
purchases across 6,000 customers — too sparse to train an implicit model or to
build a test set with enough relevant items per customer for stable ranking
metrics.

**Consequence:** any absolute claim about conversion performance is invalid. The
implicit model in particular is trained on a far denser positive signal than it
would see in production.

**Mitigation:** documented in the data validation report and the EDA notebook;
relative model comparison is unaffected.

**Residual risk: MEDIUM — acceptable, and disclosed.**

---

### R3. Overfitting to generator artefacts
**Likelihood: Medium · Impact: Medium · Priority: MEDIUM**

Hyperparameters and fusion weights were tuned against data produced by a known
process. The hybrid weights (`item_cf 0.45 / svd 0.10 / content 0.10 / ncf
0.35`) are optimal *for this generator*, not for e-commerce in general.

**Mitigation:** the weight sweep is reproducible and documented, so it can be
re-run on real data. Weights are a single named constant, not scattered
throughout the code.

**Residual risk: MEDIUM.**

---

## 3. Risk register — production data

### R4. Popularity feedback loop
**Likelihood: High · Impact: High · Priority: CRITICAL**

Recommending popular items makes them more popular, which makes the model
recommend them harder. The catalogue collapses onto a shrinking head, long-tail
inventory becomes unsellable, and the system slowly turns into an expensive
best-sellers list.

**Evidence this is real:** the popularity baseline already has catalogue
coverage of **1.3%** — it returns essentially the same list to everybody.

**Mitigation in place:**
- Inverse-propensity penalty `1 / (count + 1) ** 0.20`.
- Explicit long-tail boost (1.15×).
- Coverage and long-tail share are tracked as first-class metrics, not
  afterthoughts.
- Coverage floor in the model-promotion gate: a challenger cannot ship if
  coverage drops below 90% of incumbent.

**Residual risk: MEDIUM.** Fully closing this requires online exploration
(bandits), which is out of scope here.

---

### R5. Cold-start quality is unverifiable
**Likelihood: Certain · Impact: Medium · Priority: HIGH**

8% of customers and 10.2% of the catalogue fall into cold-start cohorts. By
definition these have no history, so there is **no ground truth** against which
to score the fallback. The cold-start path can be demonstrated but not
measured.

**Mitigation:** fallbacks use registration data (declared category + segment
price band) rather than blind popularity, which is at least defensible on
first principles. Verification requires an online A/B test.

**Residual risk: MEDIUM.**

---

### R6. Data quality failures in the ingestion pipeline
**Likelihood: Medium · Impact: High · Priority: HIGH**

Real interaction logs break in ways synthetic data never does: duplicate events
from client retries, missing `item_id` after a catalogue migration, timestamps
in mixed timezones, bot traffic inflating popularity, test accounts.

**Consequence:** bot traffic in particular is dangerous — it directly poisons
the popularity model and the item-item similarity matrix.

**Mitigation required before production:**
- Schema validation at ingestion (Great Expectations or equivalent)
- Bot / internal-account filtering **before** any aggregation
- Duplicate-event deduplication on (user, item, timestamp)
- Timezone normalisation to UTC at the boundary
- Referential integrity checks: every `item_id` must exist in the catalogue

**Currently implemented:** the loader parses timestamps to real datetimes so a
time-based split cannot silently degrade into string comparison, and popularity
features are reindexed against the full catalogue so zero-interaction items are
never dropped. Everything else on that list is **not yet built**.

**Residual risk: HIGH until the ingestion checks exist.**

---

### R7. Concept drift
**Likelihood: High · Impact: Medium · Priority: HIGH**

Customer preferences shift seasonally and structurally. A model trained on
winter data recommends coats in June. The current design trains once and would
degrade silently.

**Mitigation in place:** the retraining strategy in the architecture document
specifies per-component cadences and four drift triggers. **Not yet
implemented** as a running job.

**Residual risk: MEDIUM.**

---

### R8. Personal data and privacy
**Likelihood: Certain in production · Impact: High · Priority: HIGH**

The current schema carries `name`, `age`, `gender` and `location`. On real
customers these are personal data under GDPR / DPDP Act.

**Assessment of current use:**
- `name` — **not used by any model**. Present only for dashboard readability.
  Should be dropped or pseudonymised in production.
- `gender` — **not used as a model input.** Retained for fairness auditing only.
- `age`, `location` — used in the cold-start profile features.

**Required before production:**
- Data minimisation: drop `name` from the modelling path entirely
- Pseudonymised `user_id` with a separate identity mapping
- Retention policy on the interaction log
- A documented lawful basis for profiling, and an opt-out that actually
  disables personalisation rather than hiding it
- Right-to-erasure procedure covering embeddings, not just rows — deleting a
  customer's interactions does not remove their learned embedding

That last point is the one most often missed: an NCF embedding table retains a
learned representation of a customer after their raw rows are deleted. Erasure
requires retraining or explicit embedding removal.

**Residual risk: HIGH until the above is implemented.**

---

### R9. Training/serving skew
**Likelihood: Medium · Impact: High · Priority: HIGH**

Features computed one way in training and another way at serving produce silent
degradation that no offline metric catches.

**Mitigation in place:** the same `preprocessing.py` functions are used in
training, evaluation and serving. `HybridRecommender` is the single
implementation shared by FastAPI and Streamlit — the two application layers
cannot diverge because there is only one code path.

This was a real problem in the previous revision: fusion logic was duplicated
into `app_fastapi.py`, and a whole directory of module copies existed under
`reports/` purely to satisfy imports.

**Residual risk: LOW.**

---

### R10. Silent artifact loading failure
**Likelihood: Medium · Impact: High · Priority: HIGH**

**This is not hypothetical — it happened in the previous revision of this
project.** `app_fastapi.py` looked for NCF artifacts in `app/saved_models/`, a
directory that never existed, wrapped in a bare `try/except`. The service
started, reported healthy, and served every recommendation with the
deep-learning signal silently switched off.

**Mitigation in place:**
- Missing NCF artifacts now produce a loud console warning naming the exact file
  and the command to fix it.
- Missing core artifacts prevent startup; `/health` returns 503.
- Fusion weights are renormalised over the signals that actually produced
  scores, so a degraded hybrid is at least internally consistent.
- `/health` reports `ncf_available` explicitly.

**Residual risk: LOW.**

---

## 4. Risk summary

| ID | Risk | Priority | Residual |
|---|---|---|---|
| R1 | Synthetic data unrepresentative | CRITICAL | HIGH |
| R4 | Popularity feedback loop | CRITICAL | MEDIUM |
| R2 | Calibration deviations | HIGH | MEDIUM |
| R5 | Cold-start unverifiable | HIGH | MEDIUM |
| R6 | Ingestion data quality | HIGH | **HIGH** |
| R7 | Concept drift | HIGH | MEDIUM |
| R8 | Personal data / privacy | HIGH | **HIGH** |
| R9 | Training/serving skew | HIGH | LOW |
| R10 | Silent artifact failure | HIGH | LOW |
| R3 | Overfitting to generator | MEDIUM | MEDIUM |

**The two risks that must be closed before any production deployment are R6
(ingestion quality) and R8 (privacy).** Both are organisational and
infrastructural rather than modelling problems, and neither is solved by
improving the recommender.

---

## 5. Recommended next actions

1. Build ingestion validation with schema contracts and bot filtering (R6)
2. Complete a privacy impact assessment and drop `name` from the modelling path
   (R8)
3. Stand up the retraining scheduler with the documented drift triggers (R7)
4. Add exploration (epsilon-greedy or Thompson sampling) to break the popularity
   loop (R4)
5. Design the online A/B test that is the only way to validate cold start (R5)
