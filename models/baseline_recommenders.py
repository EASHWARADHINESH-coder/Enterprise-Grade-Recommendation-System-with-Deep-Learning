"""Baseline Recommendation Models - popularity, user-CF, item-CF, SVD."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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
def build_popularity_model(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Rank the catalogue by a blended popularity score."""
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

    def scale(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi > lo:
            return (series - lo) / (hi - lo)
        return pd.Series(0.0, index=series.index)

    # Scale each component so the weights mean what they say; unscaled,
    # interaction count would swamp average rating regardless of weight.
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


def recommend_popular(user_id: int, interactions_df: pd.DataFrame,
                      popularity_df: pd.DataFrame, n_top: int = 10) -> list:
    """Top-N most popular items the user has not seen."""
    seen = interactions_df.loc[interactions_df["user_id"] == user_id, "item_id"].tolist()

    return (
        popularity_df.loc[~popularity_df["item_id"].isin(seen), "item_id"]
        .head(n_top)
        .tolist()
    )


# =========================================================
# 2. USER-BASED COLLABORATIVE FILTERING
# =========================================================
def create_user_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between every pair of users."""
    # O(users^2): 133 MB here, 40 GB at 100k users. First thing to break at scale.
    similarity = cosine_similarity(user_item_matrix).astype(np.float32)

    user_similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    save_pickle(user_similarity_df, "user_similarity.pkl")
    return user_similarity_df


def recommend_user_based(user_id: int, user_item_matrix: pd.DataFrame,
                         user_similarity_df: pd.DataFrame, n_top: int = 10,
                         n_neighbours: int = DEFAULT_NEIGHBOURS) -> list:
    """Score items by what the most similar users rated highly."""
    if user_id not in user_item_matrix.index:
        raise ValueError("user_id " + str(user_id) + " not present in the rating matrix.")

    # Top-N neighbours only: all users would drown the signal in near-zero
    # similarities and collapse the output toward popularity.
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
def create_item_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between every pair of items."""
    item_similarity = cosine_similarity(user_item_matrix.T).astype(np.float32)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    save_pickle(item_similarity_df, "item_similarity.pkl")
    return item_similarity_df


def recommend_item_based(user_id: int, user_item_matrix: pd.DataFrame,
                         item_similarity_df: pd.DataFrame, n_top: int = 10) -> list:
    """Score candidates by similarity to the items this user rated."""
    if user_id not in user_item_matrix.index:
        raise ValueError("user_id " + str(user_id) + " not present in the rating matrix.")

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]

    if rated_items.empty:
        return []

    scores = item_similarity_df[rated_items.index].to_numpy() @ rated_items.to_numpy()
    scores = pd.Series(scores, index=item_similarity_df.index)

    scores = scores.drop(rated_items.index, errors="ignore")
    return scores.sort_values(ascending=False).head(n_top).index.tolist()


# =========================================================
# 4. MATRIX FACTORISATION (TRUNCATED SVD)
# =========================================================
def create_svd_model(user_item_matrix: pd.DataFrame, n_components: int = SVD_COMPONENTS):
    """Factorise the rating matrix into latent user and item factors."""
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


def create_predicted_rating_matrix(user_item_matrix: pd.DataFrame,
                                   user_factors_df: pd.DataFrame,
                                   item_factors_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the dense score matrix from the latent factors."""
    predicted = user_factors_df.to_numpy() @ item_factors_df.to_numpy().T

    predicted_df = pd.DataFrame(
        predicted.astype(np.float32),
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )

    save_pickle(predicted_df, "predicted_ratings.pkl")
    return predicted_df


def recommend_svd(user_id: int, user_item_matrix: pd.DataFrame,
                  predicted_ratings_df: pd.DataFrame, n_top: int = 10) -> list:
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
def describe_items(item_ids: list, items_df: pd.DataFrame) -> pd.DataFrame:
    """Map a ranked list of item_ids back to catalogue rows."""
    columns = [c for c in ["item_id", "title", "category", "brand", "price"]
               if c in items_df.columns]

    result = items_df.loc[items_df["item_id"].isin(item_ids), columns].copy()
    result["item_id"] = pd.Categorical(result["item_id"], categories=item_ids, ordered=True)
    return result.sort_values("item_id").reset_index(drop=True)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Train all four baselines and compare them on one user."""
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

    # SVD on the mean-centred matrix so factors capture relative preference.
    svd, user_factors_df, item_factors_df = create_svd_model(normalized_matrix)
    predicted_ratings_df = create_predicted_rating_matrix(
        normalized_matrix, user_factors_df, item_factors_df
    )
    print("  predicted ratings :", predicted_ratings_df.shape)

    sample_user_id = int(user_item_matrix.index[0])
    user_row = users_df.loc[users_df["user_id"] == sample_user_id].iloc[0]

    print("\nUser {} - {}, prefers {}\n".format(
        sample_user_id, user_row["user_segment"], user_row["preferred_category"]))

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
