"""
Recommendation Evaluation Metrics
=================================
Enterprise-Grade Recommendation System with Deep Learning

Implements the four ranking metrics the business case marks as mandatory,
plus the beyond-accuracy metrics needed to judge whether a recommender is
actually good rather than merely accurate.

    Precision@K   of the K items shown, what fraction were relevant
    Recall@K      of all relevant items, what fraction appeared in the top K
    MAP@K         mean average precision - rewards putting hits near the top
    NDCG@K        discounted cumulative gain - position-weighted, the standard

Why both Precision and NDCG, when they measure similar things? Precision@10
treats a hit at position 1 and a hit at position 10 identically. On a storefront
they are worth very different amounts, because click-through decays sharply with
position. NDCG applies a logarithmic positional discount and is therefore the
metric that best tracks revenue.

Written from first principles rather than pulled from a library, because the
brief asks for custom metric implementations and because the exact handling of
edge cases - users with no relevant items, K larger than the candidate pool -
determines whether the reported numbers are trustworthy.
"""

import numpy as np
import pandas as pd


# =========================================================
# CORE RANKING METRICS
# =========================================================
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of the top-K recommendations that were relevant.

    The denominator is min(k, len(recommended)), not k. If the model could only
    produce 6 candidates, scoring it out of 10 penalises it for items it was
    never able to return and understates its precision.
    """
    if not recommended or not relevant:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)

    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of all relevant items captured within the top K.

    Note the ceiling: if a user has 40 relevant items in the test period,
    Recall@10 cannot exceed 0.25 no matter how perfect the model is. Recall
    should always be read alongside the relevant-item count per user, never
    on its own.
    """
    if not recommended or not relevant:
        return 0.0

    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Average precision for one user - precision recomputed at each hit position.

    Normalised by min(len(relevant), k) rather than by the hit count, which is
    the standard formulation. Dividing by hits alone would award a perfect 1.0
    to a model that found exactly one relevant item and placed it first, which
    is plainly not perfect performance.
    """
    if not recommended or not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for position, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / position

    denominator = min(len(relevant), k)
    return precision_sum / denominator if denominator > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int, relevance_scores: dict | None = None) -> float:
    """
    Normalised discounted cumulative gain.

    Supports graded relevance via `relevance_scores` (e.g. a 5-star rating is
    worth more than a 4-star one). When omitted, relevance is binary.

    The ideal DCG is computed from the best achievable ordering given what the
    user actually found relevant, so a user with only 2 relevant items is not
    penalised for failing to fill 10 slots.
    """
    if not recommended or not relevant:
        return 0.0

    def gain(item) -> float:
        if relevance_scores is None:
            return 1.0 if item in relevant else 0.0
        return float(relevance_scores.get(item, 0.0))

    dcg = sum(
        gain(item) / np.log2(position + 1)
        for position, item in enumerate(recommended[:k], start=1)
        if item in relevant
    )

    if relevance_scores is None:
        ideal_gains = [1.0] * min(len(relevant), k)
    else:
        ideal_gains = sorted(
            (relevance_scores.get(i, 0.0) for i in relevant), reverse=True
        )[:k]

    idcg = sum(g / np.log2(position + 1) for position, g in enumerate(ideal_gains, start=1))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """Did the top K contain at least one relevant item? (1.0 or 0.0)"""
    if not recommended or not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# =========================================================
# BEYOND-ACCURACY METRICS
# =========================================================
def catalogue_coverage(all_recommendations: list[list], total_items: int) -> float:
    """
    Share of the catalogue that ever appears in any recommendation list.

    A model can post excellent precision while only ever recommending 40 items
    out of 2,500. Commercially that is a failure: the remaining inventory is
    invisible and will never sell. Accuracy metrics alone cannot detect this.
    """
    if total_items == 0:
        return 0.0

    recommended_items = set()
    for recommendations in all_recommendations:
        recommended_items.update(recommendations)

    return len(recommended_items) / total_items


def long_tail_share(all_recommendations: list[list], long_tail_items: set) -> float:
    """Fraction of all recommended slots occupied by long-tail items."""
    total = sum(len(r) for r in all_recommendations)
    if total == 0:
        return 0.0

    tail_hits = sum(
        1 for recommendations in all_recommendations
        for item in recommendations if item in long_tail_items
    )
    return tail_hits / total


def intra_list_diversity(recommendations: list, similarity_df: pd.DataFrame) -> float:
    """
    Average pairwise dissimilarity within a single recommendation list.

    Guards against the classic failure where a model returns ten near-identical
    products. Each is individually relevant, so precision looks excellent, but
    the list gives the customer no real choice.
    """
    valid = [i for i in recommendations if i in similarity_df.index]
    if len(valid) < 2:
        return 0.0

    submatrix = similarity_df.loc[valid, valid].to_numpy()
    upper_triangle = submatrix[np.triu_indices(len(valid), k=1)]

    return float(1.0 - upper_triangle.mean())


def novelty(all_recommendations: list[list], popularity_counts: pd.Series) -> float:
    """
    Mean self-information of recommended items: -log2(P(item)).

    Higher means the system is surfacing less obvious products. A recommender
    that only returns best-sellers scores near zero here and is, in effect,
    an expensive way to reproduce a top-sellers list.
    """
    total_interactions = popularity_counts.sum()
    if total_interactions == 0:
        return 0.0

    scores = []
    for recommendations in all_recommendations:
        for item in recommendations:
            probability = popularity_counts.get(item, 0) / total_interactions
            if probability > 0:
                scores.append(-np.log2(probability))

    return float(np.mean(scores)) if scores else 0.0


# =========================================================
# PREDICTIVE METRICS
# =========================================================
def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(actuals)) ** 2)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actuals))))


# =========================================================
# AGGREGATION
# =========================================================
def evaluate_ranking(
    recommendations_by_user: dict[int, list],
    relevant_by_user: dict[int, set],
    k_values: list[int],
    relevance_scores_by_user: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """
    Compute all ranking metrics at each K, averaged over users.

    Only users who have at least one relevant item in the test period are
    scored. Including users with nothing to find would drag every metric toward
    zero by an amount that depends on the test split rather than on model
    quality, making runs incomparable.
    """
    evaluable_users = [
        user_id for user_id in recommendations_by_user
        if relevant_by_user.get(user_id)
    ]

    if not evaluable_users:
        return pd.DataFrame()

    rows = []

    for k in k_values:
        precisions, recalls, maps, ndcgs, hits = [], [], [], [], []

        for user_id in evaluable_users:
            recommended = recommendations_by_user[user_id]
            relevant = relevant_by_user[user_id]
            graded = (relevance_scores_by_user or {}).get(user_id)

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            maps.append(average_precision_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k, graded))
            hits.append(hit_rate_at_k(recommended, relevant, k))

        rows.append({
            "K": k,
            f"Precision@K": float(np.mean(precisions)),
            f"Recall@K": float(np.mean(recalls)),
            f"MAP@K": float(np.mean(maps)),
            f"NDCG@K": float(np.mean(ndcgs)),
            f"HitRate@K": float(np.mean(hits)),
            "evaluated_users": len(evaluable_users),
        })

    return pd.DataFrame(rows)
