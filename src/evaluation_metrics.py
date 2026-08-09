"""Recommendation Evaluation Metrics - ranking and beyond-accuracy measures."""

import numpy as np
import pandas as pd


# =========================================================
# CORE RANKING METRICS
# =========================================================
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the top-K recommendations that were relevant."""
    if not recommended or not relevant:
        return 0.0

    # Denominator is len(top_k), not k: do not penalise a model for
    # candidates it was never able to return.
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)

    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items captured within the top K."""
    if not recommended or not relevant:
        return 0.0

    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Average precision for one user, recomputed at each hit position."""
    if not recommended or not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for position, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / position

    # Normalise by min(relevant, k), not hits: dividing by hits alone would
    # score 1.0 for finding a single relevant item.
    denominator = min(len(relevant), k)
    return precision_sum / denominator if denominator > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int,
              relevance_scores: dict = None) -> float:
    """Normalised discounted cumulative gain, with optional graded relevance."""
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
    """Did the top K contain at least one relevant item?"""
    if not recommended or not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# =========================================================
# BEYOND-ACCURACY METRICS
# =========================================================
def catalogue_coverage(all_recommendations: list, total_items: int) -> float:
    """Share of the catalogue that ever appears in any recommendation."""
    if total_items == 0:
        return 0.0

    recommended_items = set()
    for recommendations in all_recommendations:
        recommended_items.update(recommendations)

    return len(recommended_items) / total_items


def long_tail_share(all_recommendations: list, long_tail_items: set) -> float:
    """Fraction of recommended slots occupied by long-tail items."""
    total = sum(len(r) for r in all_recommendations)
    if total == 0:
        return 0.0

    tail_hits = sum(
        1 for recommendations in all_recommendations
        for item in recommendations if item in long_tail_items
    )
    return tail_hits / total


def intra_list_diversity(recommendations: list, similarity_df: pd.DataFrame) -> float:
    """Average pairwise dissimilarity within one recommendation list."""
    valid = [i for i in recommendations if i in similarity_df.index]
    if len(valid) < 2:
        return 0.0

    submatrix = similarity_df.loc[valid, valid].to_numpy()
    upper_triangle = submatrix[np.triu_indices(len(valid), k=1)]

    return float(1.0 - upper_triangle.mean())


def novelty(all_recommendations: list, popularity_counts: pd.Series) -> float:
    """Mean self-information of recommended items: -log2(P(item))."""
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
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(actuals)) ** 2)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actuals))))


# =========================================================
# AGGREGATION
# =========================================================
def evaluate_ranking(recommendations_by_user: dict, relevant_by_user: dict,
                     k_values: list, relevance_scores_by_user: dict = None) -> pd.DataFrame:
    """Compute all ranking metrics at each K, averaged over users."""
    # Only score users who have something to find; including the rest drags
    # every metric down by an amount that depends on the split, not the model.
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
            "Precision@K": float(np.mean(precisions)),
            "Recall@K": float(np.mean(recalls)),
            "MAP@K": float(np.mean(maps)),
            "NDCG@K": float(np.mean(ndcgs)),
            "HitRate@K": float(np.mean(hits)),
            "evaluated_users": len(evaluable_users),
        })

    return pd.DataFrame(rows)
