"""
Baseline Recommendation Models (Machine Learning)
=================================================
Enterprise-Grade Recommendation System with Deep Learning

The four baselines the business case marks as mandatory:

    1. Popularity-based recommender
    2. User-based Collaborative Filtering
    3. Item-based Collaborative Filtering
    4. Matrix Factorisation (Truncated SVD)

These exist to establish the bar. A deep model that cannot beat a popularity
list is not worth its serving cost, and stating that comparison honestly is the
point of building them first.

Implemented with scikit-learn. The `surprise` library named as an alternative
in the brief has no wheel for Python 3.13, so TruncatedSVD provides the matrix
factorisation instead - the same latent-factor method, without the dependency.

Run:
    python models/baseline_recommenders.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Make the shared modules in src/ importable from this folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import load_all
from preprocessing import load_pickle, save_pickle


# =========================================================
# PATHS
# =========================================================
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


# =========================================================
# SETTINGS
# =========================================================
SVD_COMPONENTS = 50
DEFAULT_NEIGHBOURS = 50


# =========================================================
# 1. POPULARITY-BASED RECOMMENDER
# =========================================================
def build_popularity_model(interactions_df):
    """
    Rank the catalogue by a blended popularity score.

    Raw interaction count alone rewards items that are merely widely exposed.
    Blending in average rating and conversion rate means an item has to be both
    seen and actually liked to reach the top of the list, which is the sensible
    non-personalised default for a storefront.
    """
    popularity_df = interactions_df.groupby("item_id").agg(
        interaction_count=("item_id", "count"),
        avg_rating=("rating", "mean"),
        purchase_count=("purchase", "sum"),
        total_revenue=("revenue", "sum"),
    ).reset_index()

    popularity_df["avg_rating"] = popularity_df["avg_rating"].fillna(0.0)
    popularity_df["conversion_rate"] = (
        popularity_df["purchase_count"] / popularity_df["interaction_count"]
    )

    # Scale each component to [0, 1] so the weights mean what they say.
    def scale(series):
        lo, hi = series.min(), series.max()
        if hi > lo:
            return (series - lo) / (hi - lo)
        return pd.Series(0.0, index=series.index)

    popularity_df["popularity_score"] = (
        0.50 * scale(popularity_df["interaction_count"])
        + 0.30 * scale(popularity_df["avg_rating"])
        + 0.20 * scale(popularity_df["conversion_rate"])
    )

    popularity_df = popularity_df.sort_values(
        "popularity_score", ascending=False
    ).reset_index(drop=True)

    save_pickle(popularity_df, "popularity_model.pkl")
    return popularity_df


def recommend_popular(user_id, interactions_df, popularity_df, n_top=10):
    """Top-N most popular items the user has not already interacted with."""
    seen = interactions_df.loc[interactions_df["user_id"] == user_id, "item_id"].tolist()

    return (
        popularity_df.loc[~popularity_df["item_id"].isin(seen), "item_id"]
        .head(n_top)
        .tolist()
    )


# =========================================================
# 2. USER-BASED COLLABORATIVE FILTERING
# =========================================================
def create_user_similarity_matrix(user_item_matrix):
    """
    Cosine similarity between every pair of users.

    Note the cost: this is an O(users^2) dense matrix. At ~5,800 users it is
    ~133 MB in float32 and still tractable, but it is the first thing that
    breaks as the platform grows - which is why the scalability document
    recommends replacing it with approximate nearest neighbours beyond ~50k
    users rather than scaling the machine.
    """
    similarity = cosine_similarity(user_item_matrix).astype(np.float32)

    user_similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    save_pickle(user_similarity_df, "user_similarity.pkl")
    return user_similarity_df


def recommend_user_based(user_id, user_item_matrix, user_similarity_df,
                         n_top=10, n_neighbours=DEFAULT_NEIGHBOURS):
    """
    Score items by what similar users rated highly.

    Only the top `n_neighbours` most similar users contribute. Using all users
    would drown the signal in near-zero similarities and quietly collapse the
    output toward a popularity ranking.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError("user_id " + str(user_id) + " not present in the rating matrix.")

    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id, errors="ignore")
        .sort_values(ascending=False)
        .head(n_neighbours)
    )

    similarity_sum = similar_users.sum()
    weighted_scores = user_item_matrix.loc[similar_users.index].T.dot(similar_users)

    if similarity_sum > 0:
        weighted_scores = weighted_scores / similarity_sum

    already_seen = user_item_matrix.loc[user_id]
    weighted_scores = weighted_scores.drop(
        already_seen[already_seen > 0].index, errors="ignore"
    )

    return weighted_scores.sort_values(ascending=False).head(n_top).index.tolist()


# =========================================================
# 3. ITEM-BASED COLLABORATIVE FILTERING
# =========================================================
def create_item_similarity_matrix(user_item_matrix):
    """
    Cosine similarity between every pair of items.

    Preferred over the user-user matrix in production: the catalogue changes far
    more slowly than the user base, so this can be recomputed nightly instead of
    continuously.
    """
    item_similarity = cosine_similarity(user_item_matrix.T).astype(np.float32)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    save_pickle(item_similarity_df, "item_similarity.pkl")
    return item_similarity_df


def recommend_item_based(user_id, user_item_matrix, item_similarity_df, n_top=10):
    """Score candidates by similarity to the items this user already rated."""
    if user_id not in user_item_matrix.index:
        raise ValueError("user_id " + str(user_id) + " not present in the rating matrix.")

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]

    if rated_items.empty:
        return []

    # Vectorised: weight each rated item's similarity column by its rating.
    scores = item_similarity_df[rated_items.index].to_numpy() @ rated_items.to_numpy()
    scores = pd.Series(scores, index=item_similarity_df.index)

    scores = scores.drop(rated_items.index, errors="ignore")
    return scores.sort_values(ascending=False).head(n_top).index.tolist()


# =========================================================
# 4. MATRIX FACTORISATION (TRUNCATED SVD)
# =========================================================
def create_svd_model(user_item_matrix, n_components=SVD_COMPONENTS):
    """
    Factorise the rating matrix into latent user and item factors.

    The matrix is >99% empty, so the raw co-occurrence signal is far too thin
    for direct similarity. Compressing to `n_components` latent dimensions lets
    the model generalise across users who liked related-but-not-identical items.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T

    user_factors_df = pd.DataFrame(user_factors, index=user_item_matrix.index)
    item_factors_df = pd.DataFrame(item_factors, index=user_item_matrix.columns)

    save_pickle(svd, "svd_model.pkl")
    save_pickle(user_factors_df, "user_factors.pkl")
    save_pickle(item_factors_df, "item_factors.pkl")

    explained = float(svd.explained_variance_ratio_.sum())
    print("  SVD explained variance ({} components): {:.2%}".format(n_components, explained))

    return svd, user_factors_df, item_factors_df


def create_predicted_rating_matrix(user_item_matrix, user_factors_df, item_factors_df):
    """Reconstruct the dense score matrix from the latent factors."""
    predicted = user_factors_df.to_numpy() @ item_factors_df.to_numpy().T

    predicted_df = pd.DataFrame(
        predicted.astype(np.float32),
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )

    save_pickle(predicted_df, "predicted_ratings.pkl")
    return predicted_df


def recommend_svd(user_id, user_item_matrix, predicted_ratings_df, n_top=10):
    """Top-N unseen items by reconstructed latent score."""
    if user_id not in user_item_matrix.index:
        raise ValueError("user_id " + str(user_id) + " not present in the rating matrix.")

    already_seen = user_item_matrix.loc[user_id]
    predictions = predicted_ratings_df.loc[user_id].drop(
        already_seen[already_seen > 0].index, errors="ignore"
    )

    return predictions.sort_values(ascending=False).head(n_top).index.tolist()


# =========================================================
# PRESENTATION HELPER
# =========================================================
def describe_items(item_ids, items_df):
    """Map a ranked list of item_ids back to human-readable catalogue rows."""
    columns = [c for c in ["item_id", "title", "category", "brand", "price"]
               if c in items_df.columns]

    result = items_df.loc[items_df["item_id"].isin(item_ids), columns].copy()
    result["item_id"] = pd.Categorical(result["item_id"], categories=item_ids, ordered=True)
    return result.sort_values("item_id").reset_index(drop=True)


# =========================================================
# PIPELINE ENTRY POINT
# =========================================================
def main():
    users_df, items_df, interactions_df = load_all()
    user_item_matrix = load_pickle("user_item_matrix.pkl")
    normalized_matrix = load_pickle("normalized_user_item_matrix.pkl")

    print("Training baseline recommenders ...")

    popularity_df = build_popularity_model(interactions_df)
    print("  popularity model  :", popularity_df.shape)

    user_similarity_df = create_user_similarity_matrix(user_item_matrix)
    print("  user similarity   :", user_similarity_df.shape)

    item_similarity_df = create_item_similarity_matrix(user_item_matrix)
    print("  item similarity   :", item_similarity_df.shape)

    # SVD runs on the mean-centred matrix so latent factors capture relative
    # preference rather than each user's personal rating offset.
    svd, user_factors_df, item_factors_df = create_svd_model(normalized_matrix)
    predicted_ratings_df = create_predicted_rating_matrix(
        normalized_matrix, user_factors_df, item_factors_df
    )
    print("  predicted ratings :", predicted_ratings_df.shape)

    # ---- Demonstrate all four on the same user --------------------------
    sample_user_id = int(user_item_matrix.index[0])
    user_row = users_df.loc[users_df["user_id"] == sample_user_id].iloc[0]

    print("\nSample recommendations for user_id {}".format(sample_user_id))
    print("  segment={}, prefers={}\n".format(
        user_row["user_segment"], user_row["preferred_category"]))

    results = [
        ("Popularity", recommend_popular(sample_user_id, interactions_df, popularity_df, 5)),
        ("User-based CF", recommend_user_based(sample_user_id, user_item_matrix,
                                               user_similarity_df, 5)),
        ("Item-based CF", recommend_item_based(sample_user_id, user_item_matrix,
                                               item_similarity_df, 5)),
        ("SVD", recommend_svd(sample_user_id, user_item_matrix, predicted_ratings_df, 5)),
    ]

    for label, recs in results:
        print(label + ":")
        print(describe_items(recs, items_df).to_string(index=False))
        print()

    print("Baseline models saved to", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
