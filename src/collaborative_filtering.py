"""Collaborative Filtering Improvements - de-biasing, re-ranking, cold start."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from content_based_nlp import recommend_for_new_user_by_profile
from data_loader import load_all
from preprocessing import load_pickle, save_pickle


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


# =========================================================
# SETTINGS
# =========================================================
COLD_START_MIN_INTERACTIONS = 3

# Accuracy vs diversity dial. Measured top-10 overlap against the
# uncorrected ranking: 0.10 -> 97%, 0.20 -> 92%, 0.35 -> 86%.
POPULARITY_PENALTY_ALPHA = 0.20

LONG_TAIL_BOOST = 1.15

# Retrieve this many by relevance, then diversify within that pool only.
CANDIDATE_POOL_SIZE = 200


# =========================================================
# IMPLICIT ITEM-ITEM SIMILARITY
# =========================================================
def create_item_similarity_from_implicit(implicit_matrix: pd.DataFrame) -> pd.DataFrame:
    """Item-item similarity from cart/purchase intent rather than ratings."""
    similarity = cosine_similarity(implicit_matrix.T).astype(np.float32)

    similarity_df = pd.DataFrame(
        similarity,
        index=implicit_matrix.columns,
        columns=implicit_matrix.columns,
    )

    save_pickle(similarity_df, "implicit_item_similarity.pkl")
    return similarity_df


# =========================================================
# POPULARITY DE-BIASING
# =========================================================
def build_popularity_lookup(item_popularity_features: pd.DataFrame,
                            penalty_alpha: float = POPULARITY_PENALTY_ALPHA) -> pd.DataFrame:
    """Precompute the per-item re-ranking multipliers."""
    lookup = item_popularity_features.copy()

    # Power form, not 1/(1+log1p(count)): that spans a ~5x range, wider than
    # the scores it multiplies, and collapses the ranking onto obscure items.
    lookup["popularity_penalty"] = 1.0 / np.power(
        lookup["interaction_count"] + 1.0, penalty_alpha
    )
    lookup["long_tail_boost"] = np.where(lookup["is_long_tail"] == 1, LONG_TAIL_BOOST, 1.00)

    return lookup.set_index("item_id")


RERANK_COLUMNS = [
    "interaction_count",
    "average_rating",
    "popularity_ratio",
    "conversion_rate",
    "is_long_tail",
    "popularity_penalty",
    "long_tail_boost",
]

RERANK_DEFAULTS = {
    "interaction_count": 0.0,
    "average_rating": 0.0,
    "popularity_ratio": 0.0,
    "conversion_rate": 0.0,
    "is_long_tail": 1.0,
    "popularity_penalty": 1.0,
    "long_tail_boost": 1.0,
}


def min_max_normalize(score_series: pd.Series) -> pd.Series:
    """Scale a score vector to [0, 1]."""
    scores = score_series.astype(float)

    if scores.empty:
        return scores

    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return pd.Series(0.0, index=scores.index)

    return (scores - lo) / (hi - lo)


def rerank_with_popularity_balance(score_series: pd.Series, popularity_lookup: pd.DataFrame,
                                   candidate_pool: int = CANDIDATE_POOL_SIZE) -> pd.DataFrame:
    """Retrieve by relevance, then diversify within the retrieved pool."""
    scores = score_series.astype(float)

    # Pool first: applied catalogue-wide, the penalty promotes irrelevant
    # obscure items above genuinely relevant ones.
    if candidate_pool is not None and len(scores) > candidate_pool:
        scores = scores.nlargest(candidate_pool)

    # Normalise before multiplying: SVD scores are signed, and a negative
    # score times a penalty below 1 moves UP, inverting the correction.
    recommendation_df = pd.DataFrame({
        "raw_score": min_max_normalize(scores),
        "original_score": scores,
    })

    available = [c for c in RERANK_COLUMNS if c in popularity_lookup.columns]
    recommendation_df = recommendation_df.join(popularity_lookup[available], how="left")

    for column, default in RERANK_DEFAULTS.items():
        if column in recommendation_df.columns:
            recommendation_df[column] = recommendation_df[column].fillna(default)
        else:
            recommendation_df[column] = default

    recommendation_df["adjusted_score"] = (
        recommendation_df["raw_score"]
        * recommendation_df["popularity_penalty"]
        * recommendation_df["long_tail_boost"]
    )

    return recommendation_df.sort_values("adjusted_score", ascending=False)


# =========================================================
# COLD-START DETECTION
# =========================================================
def get_cold_start_items(popularity_lookup: pd.DataFrame,
                         min_interactions: int = COLD_START_MIN_INTERACTIONS) -> list:
    """Item IDs with too little history to be scored collaboratively."""
    counts = popularity_lookup["interaction_count"]
    return counts.loc[counts < min_interactions].index.tolist()


def get_cold_start_users(user_engagement: pd.DataFrame, all_user_ids: np.ndarray,
                         min_interactions: int = COLD_START_MIN_INTERACTIONS) -> list:
    """User IDs needing fallback treatment."""
    # Reindex so zero-interaction users, absent from the frame, are included.
    counts = user_engagement["interaction_count"].reindex(all_user_ids, fill_value=0)
    return counts.loc[counts < min_interactions].index.tolist()


# =========================================================
# SCORING
# =========================================================
def recommend_svd_scores(user_id: int, user_item_matrix: pd.DataFrame,
                         predicted_ratings_df: pd.DataFrame, popularity_lookup: pd.DataFrame,
                         n_candidates: int = 200) -> pd.Series:
    """Latent-factor scores for unseen items, popularity-corrected."""
    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)

    seen = user_item_matrix.loc[user_id]
    predictions = predicted_ratings_df.loc[user_id].drop(
        seen[seen > 0].index, errors="ignore"
    )

    reranked = rerank_with_popularity_balance(predictions, popularity_lookup)
    return reranked["adjusted_score"].head(n_candidates)


def recommend_implicit_scores(user_id: int, implicit_matrix: pd.DataFrame,
                              item_similarity_df: pd.DataFrame, popularity_lookup: pd.DataFrame,
                              n_candidates: int = 200) -> pd.Series:
    """Item-item implicit CF scores, popularity-corrected."""
    if user_id not in implicit_matrix.index:
        return pd.Series(dtype=float)

    user_row = implicit_matrix.loc[user_id]
    interacted = user_row[user_row > 0].index.tolist()

    if not interacted:
        return pd.Series(dtype=float)

    scores = pd.Series(
        item_similarity_df[interacted].to_numpy().sum(axis=1),
        index=item_similarity_df.index,
    )
    scores = scores.drop(interacted, errors="ignore")

    reranked = rerank_with_popularity_balance(scores, popularity_lookup)
    return reranked["adjusted_score"].head(n_candidates)


def score_items_for_new_user(user_id: int, users_df: pd.DataFrame, items_df: pd.DataFrame,
                             popularity_lookup: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """Cold-start scoring from registration attributes alone."""
    user_row = users_df.loc[users_df["user_id"] == user_id]

    if user_row.empty:
        ranked = rerank_with_popularity_balance(
            popularity_lookup["popularity_ratio"], popularity_lookup
        )
        top_ids = ranked.head(n_top).index

        result = items_df.loc[items_df["item_id"].isin(top_ids)].copy()
        result["fallback_reason"] = "global_popularity"
        return result

    profile = user_row.iloc[0]
    result = recommend_for_new_user_by_profile(
        preferred_category=profile["preferred_category"],
        items_df=items_df,
        user_segment=profile["user_segment"],
        n_top=n_top,
    ).copy()
    result["fallback_reason"] = "segment_and_category_profile"

    return result


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Build CF artifacts and demonstrate de-biasing and cold start."""
    users_df, items_df, interactions_df = load_all()

    implicit_matrix = load_pickle("implicit_feedback_matrix.pkl")
    item_popularity = load_pickle("item_popularity_features.pkl")
    predicted_ratings_df = load_pickle("predicted_ratings.pkl")
    user_engagement = load_pickle("user_engagement_features.pkl")
    user_item_matrix = load_pickle("user_item_matrix.pkl")

    print("Building collaborative filtering improvements ...")

    implicit_similarity = create_item_similarity_from_implicit(implicit_matrix)
    print("  implicit item similarity :", implicit_similarity.shape)

    popularity_lookup = build_popularity_lookup(item_popularity)
    save_pickle(popularity_lookup, "popularity_lookup.pkl")
    print("  popularity lookup        :", popularity_lookup.shape)

    cold_items = get_cold_start_items(popularity_lookup)
    cold_users = get_cold_start_users(user_engagement, users_df["user_id"].to_numpy())
    print("  cold-start items         :", len(cold_items))
    print("  cold-start users         :", len(cold_users))

    active_user = int(user_item_matrix.index[0])
    raw = predicted_ratings_df.loc[active_user]
    reranked = rerank_with_popularity_balance(raw, popularity_lookup)

    print("\nPopularity de-biasing for user {}:".format(active_user))
    for k in (10, 20):
        before = raw.nlargest(k).index
        after = reranked.head(k).index
        print("  top-{:<3} mean exposure : {:>7,.0f} -> {:>7,.0f}".format(
            k,
            popularity_lookup.loc[before, "interaction_count"].mean(),
            popularity_lookup.loc[after, "interaction_count"].mean(),
        ))

    if cold_users:
        cold_user_id = int(cold_users[0])
        print("\nCold-start fallback for user {}:".format(cold_user_id))
        fallback = score_items_for_new_user(cold_user_id, users_df, items_df,
                                            popularity_lookup, 5)
        columns = [c for c in ["item_id", "title", "category", "price", "fallback_reason"]
                   if c in fallback.columns]
        print(fallback[columns].to_string(index=False))

    print("\nCollaborative filtering artifacts saved to", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
